# ppo_server_agent.py
import torch
import uuid
import threading
from typing import Dict, Optional, Tuple, List, Union
from ppo_trainer import PPOTrainer
from llm_policy import LLMAgent
from torch.utils.tensorboard import SummaryWriter


class LLMTaskSession:
    """Per-task session to accumulate interactions and trigger training if allowed."""
    def __init__(self, task_id: str, status: str = "attached"):
        self.task_id = task_id
        self.status = status # May be useless
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
        obs = torch.stack([x["obs"] for x in self.trajectory[:-1]])
        actions = torch.stack([x["action"] for x in self.trajectory[:-1]])
        rewards = torch.stack([x["reward"] for x in self.trajectory[:-1]])
        dones = torch.stack([x["done"] for x in self.trajectory[:-1]])
        values = torch.stack([x["value"] for x in self.trajectory[:-1]])
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
            "next_done": last["done"],
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
        self.inference = args.inference
        self.sessions: Dict[str, LLMTaskSession] = {}
        self.lock = threading.Lock()

        self.agent = LLMAgent(normalization_mode="word", batch_size=2, inference=args.inference)

        if not self.inference:
            self.writer = SummaryWriter(f"{args.record_path}")
            self.trainer = PPOTrainer(self.agent, args, self.writer)

    def new_task(self) -> str:
        """Start a new task session."""
        task_id = str(uuid.uuid4())
        self.sessions[task_id] = LLMTaskSession(task_id)
        return task_id

    def get_total_trajectory_size(self) -> int:
        """
        Computes the total trajectory size across all task sessions.
        The size of each trajectory is defined as len(trajectory) - 1.
        If the trajectory is empty, its size is 0.
        """
        total_size = 0
        with self.lock:
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
        with self.lock:
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
        :param task_id:
        :return:
        """
        if task_id not in self.sessions:
            raise ValueError(f"Unknown task_id {task_id}. Call new_task() first.")

        if self.get_total_ask_trajectory_size() > self.agent.batch_size:
            return False
        else:
            session = self.sessions[task_id]
            with self.lock:
                if self.get_total_ask_trajectory_size() > self.agent.batch_size:
                    return False
                else:
                    session.status = "pending"
            return True

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

        session.pending = ({
            "obs" : obs,
            "action": action,
            "logprob": logprob,
            "value": value
        })
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
        if session.pending is not None:
            obs = session.pending["obs"]
            action = session.pending["action"]
            logprob = session.pending["logprob"]
            value = session.pending["value"]
            if not self.inference:
                session.store(obs, action, reward, done, value, logprob)
            session.pending = None
            session.status = "attached"
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
            del self.sessions[task_id]

    def check_task(self, task_id: str):
        """Check if a task is valid."""
        return task_id in self.sessions