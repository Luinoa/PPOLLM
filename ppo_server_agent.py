# ppo_server_agent.py
import threading
import uuid
from typing import Dict, Optional, List, Union, Any, Tuple

import torch
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch.utils.tensorboard import SummaryWriter

from llm_policy import LLMAgent
from ppo_trainer import PPOTrainer


class LLMTaskSession:
    """Per-task session to accumulate interactions and trigger training if allowed."""

    def __init__(self, task_id: str, status: str = "attached"):
        self.task_id = task_id
        self.status = status  # May be useless
        self.trajectory = []
        self.total_steps = 0
        self.pending = None
        self.lock = threading.Lock()

    def store(self, obs, action, reward, done, value, logprob):
        self.trajectory.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "value": value,
            "logprob": logprob,
        })
        self.total_steps += 1
        if done:
            self.status = "done"

    def get_experience(self):
        """
        Prepare experience data for PPO training.
        Discard the final reward, keep the last transition's obs/done for next update.
        """
        # Do not include the last transition when computing rewards/returns
        obs = [x["obs"] for x in self.trajectory[:-1]]
        actions = torch.tensor([x["action"] for x in self.trajectory[:-1]])
        rewards = torch.tensor([x["reward"] for x in self.trajectory[:-1]])
        dones = torch.tensor([1.0 if x["done"] == True else 0.0 for x in self.trajectory[:-1]])
        values = torch.tensor([x["value"] if x["value"] is not None else 0.0 for x in self.trajectory[:-1]])
        logprobs = torch.stack([x["logprob"] for x in self.trajectory[:-1]])

        # Keep the final transition to serve as the seed for the next round
        last = self.trajectory[-1]

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "logprobs": logprobs,
            "next_obs": last["obs"],
            "next_done": torch.tensor(1.0 if last["done"] == True else 0.0),
        }

    def reset(self):
        """Reset the session while preserving the last transition as the new start."""
        if len(self.trajectory) > 0:
            # Keep the last step to continue from
            self.trajectory = [self.trajectory[-1]]
        else:
            self.trajectory = []
        self.total_steps = len(self.trajectory)
        self.status = "attached"


class PPOAgentServer:
    def __init__(self, args):
        """
        :param agent: An instance of LLMAgent (implementing get_action_and_value, save, etc.)
        :param trainer: PPOTrainer instance to perform updates (can be None for pure inference)
        :param inference_only: If True, disables training logic
        """
        self.args = args
        self.inference = args.inference
        self.sessions: Dict[str, LLMTaskSession] = {}
        self.lock = threading.Lock()
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.store = {}

        # Initialize the retriever
        markdown_path = "https://raw.githubusercontent.com/openatx/uiautomator2/master/README_CN.md"
        loader = UnstructuredMarkdownLoader(markdown_path)
        data = loader.load()
        assert len(data) == 1
        assert isinstance(data[0], Document)
        readme_content = data[0].page_content
        api_start_index = readme_content.find("API Documents")
        api_content = readme_content[api_start_index:] if api_start_index != -1 else ""

        api_document = Document(page_content=api_content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        all_splits = text_splitter.split_documents([api_document])
        vector_store = Chroma.from_documents(documents=all_splits, embedding=self.embeddings)
        self.retriever = vector_store.as_retriever()

        contextualize_q_system_prompt = (
            "Given a GUI Testing history which might reference context in the GUI Testing history, "
            "formulate a standalone question which can be understood without the GUI Testing history. Do NOT answer the question."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history")]
        )
        self.history_aware_retriever = create_history_aware_retriever(self.agent, self.retriever,
                                                                      contextualize_q_prompt)

        # Setup question-answer chain
        system_prompt = (
            "You are an expert in App GUI testing to guide the testing tool to enhance the coverage of "
            "functional scenarios in testing the App based on your extensive App testing experience."
            "I'll give you a obs and you need to polish the obs based on the history and action taken by the user."
            "Please provide me with the polished obs."
            "\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder("chat_history")])
        question_answer_chain = create_stuff_documents_chain(self.agent, qa_prompt)

        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)

        self.global_step = 1

        self.agent = LLMAgent(normalization_mode="word",
                              batch_size=args.forward_batch,
                              inference=args.inference,
                              base_model=args.model,
                              )

        if not self.inference:
            self.writer = SummaryWriter(f"{args.record_path}")
            self.trainer = PPOTrainer(self.agent, args, self.writer)

    def new_task(self) -> str:
        """Start a new task session."""
        task_id = str(uuid.uuid4())
        while task_id in self.sessions:
            task_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[task_id] = LLMTaskSession(task_id)
        return task_id

    def get_total_trajectory_size(self) -> int:
        """
        Computes the total trajectory size across all task sessions.
        The size of each trajectory is defined as len(trajectory) - 1.
        If the trajectory is empty, its size is 0.
        """
        total_size = 0
        for session in self.sessions.values():
            traj_len = len(session.trajectory)
            if traj_len > 0:
                total_size += traj_len - 1
        return total_size

    def get_total_ask_trajectory_size(self) -> int:
        """
        Computes the total ask_trajectory size across all task sessions.
        If a session has non-None `status`, the effective trajectory length is len + 1.
        The size of each trajectory is defined as (effective length - 1), and is 0 if effective length is 0.
        """
        total_size = 0
        for session in self.sessions.values():
            effective_len = len(session.trajectory)
            if session.status == "pending":
                effective_len += 1
            if effective_len > 0:
                total_size += effective_len - 1
        return total_size

    def ask_for_step(self, task_id: str) -> bool:
        """
        Check if the agent is ready to take a step for the given task_id.
        """
        if task_id not in self.sessions:
            raise ValueError(f"Unknown task_id {task_id}. Call new_task() first.")

        with self.lock:
            if self.get_total_ask_trajectory_size() > len(self.sessions) * self.args.training_batch:
                return False

            session = self.sessions[task_id]
            session.status = "pending"
            return True

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def rag_step(self, task_id: str, obs: Union[str, List[str]]) -> Tuple[str, Any]:
        """
        Take one RAG step given chat history. Returns the next action.
        """
        if task_id not in self.sessions:
            raise ValueError(f"Unknown task_id {task_id}. Call new_task() first.")

        session = self.sessions[task_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            history_messages_key="chat_history",
            output_messages_key="polished_obs",
            )

        # Use the observation and task ID to generate an enhanced action or answer using RAG
        rag_result = conversational_rag_chain.invoke(
            {"input": obs},
            config={"configurable": {"session_id": task_id}},
        )["polished_obs"]

        # Get action, logprob, and value estimate
        with torch.no_grad():
            if self.inference:
                action, logprob, _, value = self.agent.get_action_and_value([rag_result], return_value=False)
            else:
                action, logprob, _, value = self.agent.get_action_and_value([rag_result], return_value=True)
            self.agent.clean()

        action_sampled = action.cpu().numpy()[0]

        # Store the pending state for later use
        session.pending = {
            "obs": obs,
            "action": action,
            "logprob": logprob,
            "value": value
        }

        '''
        Optionally combine RAG result with sampled action
        This depends on how you intend to use the RAG result
        combined_result = {
            "action": action_sampled,
            "rag_result": rag_result,
        }
        '''

        return action_sampled

    def step(
            self, task_id: str, obs: Union[str, List[str]]
    ) -> Tuple[str, Any]:
        """
        Take one interaction step given prompt or observation. Returns the next action.
        """
        if task_id not in self.sessions:
            raise ValueError(f"Unknown task_id {task_id}. Call new_task() first.")

        session = self.sessions[task_id]

        # Get action, logprob, and value estimate
        with torch.no_grad():
            if self.inference:
                action, logprob, _, value = self.agent.get_action_and_value([obs], return_value=False)
            else:
                action, logprob, _, value = self.agent.get_action_and_value([obs], return_value=True)
            self.agent.clean()
        action_sampled = action.cpu().numpy()[0]

        session.pending = ({
            "obs": obs,
            "action": action,
            "logprob": logprob,
            "value": value
        })

        return action_sampled

    def feedback(
            self, task_id: str, reward: float, done: bool, next_obs: Optional[Union[str, List[str]]] = None
    ):
        """
        Provide feedback to the agent. If done, the task is closed.
        """
        if task_id not in self.sessions:
            raise ValueError(f"Unknown task_id {task_id}. Call new_task() first.")

        session = self.sessions[task_id]

        # Store feedback
        if session.pending is not None:
            obs = session.pending["obs"]
            action = session.pending["action"]
            logprob = session.pending["logprob"]
            value = session.pending["value"]
            with self.lock:
                session.store(obs, action, reward, done, value, logprob)
                session.pending = None
                session.status = "attached"

                if self.get_total_trajectory_size() >= len(self.sessions) * self.args.training_batch:
                    # Gather experiences from all sessions
                    experiences = self.gather_experiences()
                    print('Experiences gathered:', len(experiences))
                    # Perform PPO update and save
                    if not self.inference:
                        tmp_info = self.trainer.update(experiences, self.global_step)
                        print(f"Training step {self.global_step}: {tmp_info}")
                        self.agent.save(self.global_step, self.args.record_path)
                        self.global_step += 1
                        self.writer.flush()
        else:
            raise ValueError(f"Pending feedback not found for task_id {task_id}.")

    def gather_experiences(self):
        """Collect and reset all 'done' sessions."""
        all_experiences = []
        for task_id, session in list(self.sessions.items()):
            if session.total_steps > 1:
                exp = session.get_experience()
                all_experiences.append(exp)
                session.reset()
        return all_experiences

    def close_task(self, task_id: str):
        """Optionally remove a task to free memory."""
        if task_id in self.sessions:
            with self.lock:
                del self.sessions[task_id]

    def check_task(self, task_id: str):
        """Check if a task is valid."""
        return task_id in self.sessions
