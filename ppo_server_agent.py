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
        self.total_steps = 0
        self.pending = None

    def store(self, obs, action, reward, done, value, logprob):
        self.trajectory.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "value": value,
            "logprob": logprob
        })
        self.total_steps += 1


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
        with torch.no_grad():
            if self.inference:
                action, logprob, _, value = self.agent.get_action_and_value([obs], return_value=False)
            else:
                action, logprob, _, value = self.agent.get_action_and_value([obs], return_value=True)
            self.agent.clean()
        action_sampled = action.cpu().numpy()[0]

        # TODO: Store transition for tracking purposes
        if not self.inference:
            session.pending = ({
                "obs" : obs,
                "action": action,
                "logprob": logprob,
                "value": value
            })
            session.status = "pending" # Maybe useless?
        return action_sampled

    def feedback(
            self, task_id:str, reward: float, done: bool, next_obs: Optional[Union[str, List[str]]] = None
    ):
        """
        Provide feedback to the agent. If done, the task is closed.
        """
        if task_id not in self.sessions:
            raise ValueError(f"Unknown task_id {task_id}. Call new_task() first.")

        session = self.sessions[task_id]

        # Store feedback
        if session.pending:
            obs = session.pending["obs"]
            action = session.pending["action"]
            logprob = session.pending["logprob"]
            value = session.pending["value"]
            session.store(obs, action, reward, done, value, logprob)
            session.pending = None
        else:
            raise ValueError(f"Pending feedback not found for task_id {task_id}.")

    def close_task(self, task_id: str):
        """Optionally remove a task to free memory."""
        if task_id in self.sessions:
            del self.sessions[task_id]

    def check_task(self, task_id: str):
        """Check if a task is valid."""
        return task_id in self.sessions