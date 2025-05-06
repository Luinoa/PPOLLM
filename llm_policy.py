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
    def __init__(self, normalization_mode='word',
                 load_path=None,
                 load_8bit=False,
                 batch_size=1,
                 inference=False,
                 base_model=None,
                 lora_r = 8,
                 ):
        super().__init__()

        self.load_8bit = load_8bit
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = 16
        # self.lora_dropout = 0.05
        self.lora_dropout = 0
        self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        self.batch_size = batch_size

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
        self.inference = inference
        if load_path:
            self.load(load_path)
        else:
            self.actor = self._init_actor().to(self.device)
            if not inference:
                self.critic = self._init_critic().to(self.device)
        if inference:
            self.actor.eval()

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

        """
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        """

        if not self.load_8bit:
            model.half()
        else:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        return model

    def _init_critic(self, critic_weights=None):
        critic = Critic(self.actor, self.tokenizer)
        if critic_weights is not None:
            ckpt = torch.load(critic_weights, map_location=self.device)
            # critic.v_head_mlp1.load_state_dict(ckpt["v_head_mlp1"])
            critic.v_head_mlp2.load_state_dict(ckpt["v_head_mlp2"])
            critic.v_head_mlp3.load_state_dict(ckpt["v_head_mlp3"])
        return critic

    def save(self, epoch, exp_path):
        assert not self.inference
        print("save model")

        os.makedirs(os.path.join(exp_path, "actor"), exist_ok=True)
        os.makedirs(os.path.join(exp_path, "critic"), exist_ok=True)
        # save lora
        self.actor.save_pretrained(os.path.join(exp_path, "actor"))
        # save critic
        torch.save({
            # "v_head_mlp1": self.critic.v_head_mlp1.state_dict(),
            "v_head_mlp2": self.critic.v_head_mlp2.state_dict(),
            "v_head_mlp3": self.critic.v_head_mlp3.state_dict()
        }, os.path.join(os.path.join(exp_path, "critic"), "critic.pth"))

    def load(self, exp_path):
        print("[INFO] Loading model...")
        lora_weights = os.path.join(exp_path, "actor")
        critic_weights = os.path.join(exp_path, "critic", "critic.pth")

        self.actor = self._init_actor(lora_weights).to(self.device)
        self.critic = self._init_critic(critic_weights).to(self.device)

    def get_value(self, x):
        assert not self.inference
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with self.actor.disable_adapter():
            value = self.critic(input_ids, attention_mask=attention_mask)
        return value

    def get_action_and_value(self, text_obs, action=None, return_value=True, no_grad=False):
        prompt = [o["prompt"] for o in text_obs]
        action_list = [o["action"] for o in text_obs]

        prompt_nums = len(prompt)
        action_nums = [len(item) for item in action_list]

        flat_action_list = [item for sublist in action_list for item in sublist]
        all_action_logits = []
        action_list_length = []

        for p, ac_list in zip(prompt, action_list):
            # Tokenize prompt
            prompt_ids = self.tokenizer(p, return_tensors="pt", add_special_tokens=False).to(self.device)

            # 先 forward prompt，获得 past_key_values
            with torch.no_grad() if self.inference or no_grad else torch.enable_grad():
                prompt_outputs = self.actor(**prompt_ids, use_cache=True)
            past_key_values = prompt_outputs.past_key_values

            for action_str in ac_list:
                # Tokenize action（不加 special tokens，保持拼接一致）
                action_ids = self.tokenizer(action_str, return_tensors="pt", add_special_tokens=False).to(self.device)
                action_input_ids = action_ids["input_ids"]  # [1, T]
                attention_mask = action_ids["attention_mask"]

                action_len = attention_mask.sum().item()
                action_list_length.append(action_len)

                # 用 past_key_values 推理 action（不再缓存）
                with torch.no_grad() if self.inference or no_grad else torch.enable_grad():
                    outputs = self.actor(
                        input_ids=action_input_ids,
                        past_key_values=past_key_values,
                        use_cache=False
                    )
                    logits = torch.log_softmax(outputs.logits, dim=-1)  # [1, T, V]

                # Shift logits and compute log probs
                shifted_logits = logits[:, :-1, :]  # ignore last prediction
                shifted_input_ids = action_input_ids[:, 1:]  # target tokens
                log_probs = torch.gather(shifted_logits, 2, shifted_input_ids[:, :, None]).squeeze(-1)  # [1, T-1]

                # Sum log probs (you can average later if needed)
                total_log_prob = log_probs.sum(dim=1).squeeze(0)  # scalar
                all_action_logits.append(total_log_prob)

        # Convert to tensor
        action_logits = torch.stack(all_action_logits).to(self.device)
        action_list_length = torch.tensor(action_list_length).to(self.device)

        if self.normalization_mode == 'token':
            action_logits = action_logits / action_list_length.to(self.device)
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
        # self.v_head_mlp1 = nn.Linear(self.config.n_embd, 1024, bias=False)
        self.v_head_mlp2 = nn.Linear(self.config.n_embd, 512, bias=False)
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

        # x = self.relu(self.v_head_mlp1(hidden_states))
        x = self.relu(self.v_head_mlp2(hidden_states))
        values = self.v_head_mlp3(x).squeeze(-1)
        return values