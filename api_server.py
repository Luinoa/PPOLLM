# ppo_server.py
# PPO + LangChain agent server for multi-task asynchronous interaction
import uuid

from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import uvicorn
from typing import Dict, Any


# Simulated PPO agent with get_action_and_value()
class DummyAgent:
    def get_action_and_value(self, obs):
        # Simulate a policy by returning a dummy action and value
        action = torch.tensor([0])  # Dummy action
        value = torch.tensor([0.5])  # Dummy value
        return action, torch.tensor([0.0]), None, value

    def store_transition(self, task_id, obs, action, reward, done, value):
        # Buffer logic placeholder
        print(f"[BUFFER] Stored transition for {task_id} | R: {reward}, D: {done}")

    def maybe_train(self, task_id):
        # Placeholder for training trigger
        print(f"[TRAIN] Check if training needed for {task_id}")


# Define request models
class StepRequest(BaseModel):
    task_id: str
    observation: Dict[str, Any]


class FeedbackRequest(BaseModel):
    task_id: str
    reward: float
    done: bool


class GenerationRequest(BaseModel):
    task_id: str
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True


class PlainTextResponse:
    task_id: str
    response: str


app = FastAPI()
agent = DummyAgent()

# In-memory task states
TASK_STATE = {}
BUFFER_SIZE = 2048  # max number of transitions per task

@app.post("/attach")
async def attach_task():
    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Initialize buffers for each task
    TASK_STATE[task_id] = {
        "attached": True,
        "step": 0,
        "obs": [None] * BUFFER_SIZE,
        "actions": torch.zeros(BUFFER_SIZE, dtype=torch.int64),
        "logprobs": torch.zeros(BUFFER_SIZE, dtype=torch.float32),
        "rewards": torch.zeros(BUFFER_SIZE, dtype=torch.float32),
        "dones": torch.zeros(BUFFER_SIZE, dtype=torch.float32),
        "values": torch.zeros(BUFFER_SIZE, dtype=torch.float32),
    }

    return {"status": "attached", "task_id": task_id}


@app.post("/detach")
async def detach_task(task_id: str):
    if task_id in TASK_STATE:
        del TASK_STATE[task_id]
        return {"status": "detached", "task_id": task_id}
    else:
        return {"error": "Task ID not found"}


@app.post("/step")
async def step(req: StepRequest):
    task_id = req.task_id
    obs = req.observation

    if task_id not in TASK_STATE:
        return {"error": "Invalid task ID"}

    # We treat observation as a dictionary containing prompt and optional previous actions
    obs_input = {
        "prompt": obs.get("prompt", ""),
        "history": obs.get("history", [])
    }

    action, logprob, _, value = agent.get_action_and_value(obs_input)

    state = TASK_STATE[task_id]
    step_idx = state["step"] % BUFFER_SIZE

    # Store data in buffers
    state["obs"][step_idx] = obs_input
    state["actions"][step_idx] = action
    state["logprobs"][step_idx] = logprob
    state["values"][step_idx] = value
    state["step"] += 1

    return {"action": action.item(), "value": value.item()}

"""
# Leave it commented out for now
@app.post("/generate", response_class=PlainTextResponse)
def generate_text(request: GenerationRequest):
    task_id = request.task_id
    response, logprob, value = agent.generate_text_with_value(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=request.do_sample
    )

    TASK_STATE[task_id]["prompt"] = request.prompt
    TASK_STATE[task_id]["response"] = response
    TASK_STATE[task_id]["logprob"] = logprob
    TASK_STATE[task_id]["value"] = value

    return {"task_id": task_id, "response": response}
"""


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    task_id = req.task_id
    reward = req.reward
    done = req.done

    if task_id not in TASK_STATE:
        return {"error": "Missing task state"}

    state = TASK_STATE[task_id]
    step_idx = (state["step"] - 1) % BUFFER_SIZE

    obs = state["obs"][step_idx]
    action = state["actions"][step_idx]
    value = state["values"][step_idx]

    # Store reward and done
    state["rewards"][step_idx] = reward
    state["dones"][step_idx] = float(done)

    # Store experience and maybe train
    agent.store_transition(task_id, obs, action, reward, done, value)
    agent.maybe_train(task_id)

    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)