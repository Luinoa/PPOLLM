# ppo_server_agent.py
import torch
import uuid
from typing import Dict, Optional, Tuple, List, Union

from onnxruntime.transformers.models.gpt2.parity_check_helper import inference


class LLMTaskSession:
    """Per-task session to accumulate interactions and trigger training if allowed."""
    def __init__(self, task_id: str, status: str = "attached"):
        self.task_id = task_id
        self.status = status
        self.trajectory = []

    def store(self, obs, action, reward, done, value, logprob):
        self.trajectory.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "value": value,
            "logprob": logprob
        })


class PPOAgentServer:
    def __init__(self, agent, trainer=None, inference= False):
        """
        :param agent: An instance of LLMAgent (implementing get_action_and_value, save, etc.)
        :param trainer: PPOTrainer instance to perform updates (can be None for pure inference)
        :param inference_only: If True, disables training logic
        """
        self.agent = agent
        self.trainer = trainer
        self.inference = inference
        self.sessions: Dict[str, LLMTaskSession] = {}

    def new_task(self) -> str:
        """Start a new task session."""
        task_id = str(uuid.uuid4())
        self.sessions[task_id] = LLMTaskSession(task_id)
        return task_id

    def step(
            self, task_id: str, obs: Union[str, List[str]]
    ) -> Tuple[str, Dict]:
        """
        Take one interaction step given prompt or observation. Returns the next action.
        """
        if task_id not in self.sessions:
            raise ValueError(f"Unknown task_id {task_id}. Call new_task() first.")

        session = self.sessions[task_id]

        # Get action, logprob, and value estimate
        with torch.no_grad() if self.inference else torch.enable_grad():
            action, logprob, _, value = self.agent.get_action_and_value([obs], return_value=False)
        if self.inference:
            self.agent.clean()
        action_sampled = action.cpu().numpy()[0]

        # TODO: Store transition for tracking purposes
        """
        if not self.inference
            session.store()
        """

        return action_sampled

    def close_task(self, task_id: str):
        """Optionally remove a task to free memory."""
        if task_id in self.sessions:
            del self.sessions[task_id]

    def check_task(self, task_id: str):
        """Check if a task is valid."""
        return task_id in self.sessions