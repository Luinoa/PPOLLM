import sys

import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, TaskType
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_chroma import Chroma

import os
import torch.nn as nn
import numpy as np

from torch.distributions.categorical import Categorical
import copy

root = os.path.dirname(os.path.abspath(__file__))


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LLMAgent(nn.Module):
    def __init__(self, normalization_mode='token', load_path=None, load_8bit=False, batch_size=2, inference=False):
        super().__init__()

        self.load_8bit = load_8bit
        self.base_model = 'Qwen/Qwen2-0.5B'
        self.lora_r = 8
        self.lora_alpha = 16
        # self.lora_dropout = 0.05
        self.lora_dropout = 0
        self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        self.batch_size = batch_size
        self.store = {}

        assert (
            self.base_model
        ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
        except:  # noqa: E722
            pass

        self.normalization_mode = normalization_mode

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

        self.llm = self._init_llm()
        # maybe we can change embedding model?
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.inference = inference
        if load_path:
            self.load(load_path)
        else:
            self.actor = self._init_actor().to(self.device)
            if not inference:
                self.critic = self._init_critic().to(self.device)
        if inference:
            self.actor.eval()
        
        # need to find some useful documents for ui testing
        docs = [Document(page_content="ui testing")]
        self.vector_store = Chroma.from_documents(documents=docs, embedding=self.embeddings)
        retriever = self.vector_store.as_retriever()

    def _init_llm(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            load_in_8bit=self.load_8bit,
            device_map="auto",
            cache_dir=os.path.join(root, f'weights/{self.base_model}')
        )

        if not self.load_8bit:
            model.half().to(self.device)
        else:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        return model

    def _init_actor(self, lora_weights=None):
        if lora_weights is None:
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(self.llm, config)

            model.print_trainable_parameters()

            old_state_dict = model.state_dict
            model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(model, type(model))
        else:
            model = PeftModel.from_pretrained(
                self.llm,
                lora_weights,
                torch_dtype=torch.float16,
            )

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        if not self.load_8bit:
            model.half()
        else:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        return model

    def _init_critic(self, critic_weights=None):
        critic = Critic(self.actor, self.tokenizer)
        if critic_weights is not None:
            critic.v_head.load_state_dict(torch.load(critic_weights, map_location="cpu"))
        return critic

    def save(self, epoch, exp_path):
        assert not self.inference
        print("save model")
        exp_path = os.path.join(exp_path, "epoch_{:04d}".format(epoch))

        os.makedirs(exp_path, exist_ok=True)
        # save lora
        self.actor.save_pretrained(exp_path)
        # save critic
        torch.save(self.critic.v_head.state_dict(), os.path.join(exp_path, "critic.pth"))

    @tool(response_format="content_and_artifact")
    def retrieve(self, query: str):
        """
        Retrieve information related to a query.
        """
        retrieved_docs = self.vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\n" f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def load(self, exp_path):
        print("load model")
        lora_weights = exp_path
        self.actor = self._init_actor(lora_weights).to(self.device)

        if not self.inference:
            critic_weights = os.path.join(exp_path, "critic.pth")
            self.critic = self._init_critic(critic_weights).to(self.device)

    def get_value(self, x):
        assert not self.inference
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with self.actor.disable_adapter():
            value = self.critic(input_ids, attention_mask=attention_mask)
        return value

    def get_action_and_value(self, text_obs, action=None, is_warmup=False, return_value=True):
        prompt = [o["prompt"] for o in text_obs]
        action_list = [o["action"] for o in text_obs]

        prompt_nums = len(prompt)
        action_nums = [len(item) for item in action_list]

        sequence = []
        for p, ac in zip(prompt, action_list):
            sequence += [p + " " + a for a in ac]

        # Construct input sequences: prompt + action for each option
        sequence = []
        for p, ac in zip(prompt, action_list):
            sequence += [p + " " + a for a in ac]

        # Flatten the action list for later normalization
        flat_action_list = [item for sublist in action_list for item in sublist]

        # Tokenize the flattened action list and calculate token length per action
        self.action_list_ids = self.tokenizer(flat_action_list, return_tensors="pt", padding=True)
        self.action_list_length = torch.sum(self.action_list_ids["attention_mask"], dim=-1) - 1  # exclude BOS

        # Prepare to store logits
        all_action_logits = []

        # Process in batches
        for i in range(0, len(sequence), self.batch_size):
            batch_seq = sequence[i:i + self.batch_size]
            batch_input = self.tokenizer(batch_seq, return_tensors="pt", padding=True).to(self.device)

            input_ids = batch_input["input_ids"]
            attention_mask = batch_input["attention_mask"]

            # Forward pass (no grad if warmup or inference)
            with torch.no_grad() if is_warmup or self.inference else torch.enable_grad():
                outputs = self.actor(input_ids, attention_mask=attention_mask)

            logits = torch.log_softmax(outputs.logits, dim=-1)

            # Shift inputs for token prediction
            logits = logits[:, :-1, :]
            input_ids = input_ids[:, 1:]
            gen_logits = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)

            # Slice logits to get action-specific scores
            sequence_length = torch.sum(attention_mask, dim=-1)
            batch_action_length = self.action_list_length[i:i + len(batch_seq)]
            batch_action_index = [[end - start, end] for start, end in zip(batch_action_length, sequence_length)]

            slices = [gen_logits[j, start - 1:end - 1] for j, (start, end) in enumerate(batch_action_index)]
            batch_action_logits = torch.stack([torch.sum(s) for s in slices])

            all_action_logits.append(batch_action_logits.detach())

        # Combine all logits from batches
        action_logits = torch.cat(all_action_logits, dim=0).to(self.device)

        if self.normalization_mode == 'token':
            action_logits = action_logits / self.action_list_length.to(self.device)
        elif self.normalization_mode == 'word':
            action_word_num = torch.tensor([len(action.split()) for action in flat_action_list]).to(self.device)
            action_logits = action_logits / action_word_num
        elif self.normalization_mode == 'sum':
            action_logits = action_logits
        else:
            assert 1 == 2

        actions = []
        log_probs = []
        entroy = []

        for i in range(prompt_nums):
            logits = action_logits[sum(action_nums[:i]):sum(action_nums[:i + 1])].reshape(-1, action_nums[i]).float()

            probs = Categorical(logits=logits)

            if action is None:
                cur_action = probs.sample()[0]
                cur_action = cur_action.view(-1)

            else:
                cur_action = action

            actions.append(cur_action)
            log_probs.append(probs.log_prob(cur_action))
            entroy.append(probs.entropy())

        action = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        entroy = torch.cat(entroy)

        if return_value and not self.inference:
            return action, log_probs, entroy, self.get_value(prompt)
        else:
            return action, log_probs, entroy, None

    # Normally generate texts
    def generate_text(
            self,
            prompt,
            max_new_tokens=30,
            temperature=1.0,
            top_p=0.9,
            do_sample=True,
            use_grad=False,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        context_manager = torch.enable_grad() if use_grad or self.inference else torch.no_grad()
        with context_manager:
            outputs = self.actor.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text


    def clean(self):
        torch.cuda.empty_cache()

## Note that the following code is modified from
## https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py
class Critic(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        # `OPT` models use word_embed_proj_dim as final output
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(
            self.config, "hidden_size") else self.config.word_embed_proj_dim if hasattr(
            self.config, "word_embed_proj_dim") else self.config.n_embd
        self.v_head_mlp1 = nn.Linear(self.config.n_embd, 1024, bias=False)
        self.v_head_mlp2 = nn.Linear(1024, 512, bias=False)
        self.v_head_mlp3 = nn.Linear(512, 1, bias=False)
        self.relu = nn.ReLU()
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                past_key_values=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        with torch.no_grad():
            transformer_outputs = self.rwtranrsformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_hidden_states=True)

        hidden_states = transformer_outputs[1][-1][:, -1, :].float()

        x = self.relu(self.v_head_mlp1(hidden_states))
        x = self.relu(self.v_head_mlp2(x))
        values = self.v_head_mlp3(x).squeeze(-1)
        return values