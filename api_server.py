# ppo_server.py
# PPO + LangChain agent server for multi-task asynchronous interaction

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
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True

class PlainTextResponse:
    text: str

app = FastAPI()
agent = DummyAgent()

# In-memory task states
TASK_STATE = {}


@app.post("/step")
async def step(req: StepRequest):
    task_id = req.task_id
    obs = req.observation

    # Convert obs to tensor if needed
    obs_tensor = torch.tensor([obs["state"]])

    action, logprob, _, value = agent.get_action_and_value(obs_tensor)

    # Cache current obs and value for future feedback
    TASK_STATE[task_id] = {
        "obs": obs_tensor,
        "action": action,
        "value": value
    }

    return {"action": action.item(), "value": value.item()}

@app.post("/generate", response_class=PlainTextResponse)
def generate_text(request: GenerationRequest):
    response = agent.generate_text(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=request.do_sample
    )
    return response

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    task_id = req.task_id
    reward = req.reward
    done = req.done

    if task_id not in TASK_STATE:
        return {"error": "Missing task state"}

    # Retrieve stored obs, action, value
    state = TASK_STATE[task_id]
    obs = state["obs"]
    action = state["action"]
    value = state["value"]

    # Store experience and maybe train
    agent.store_transition(task_id, obs, action, reward, done, value)
    agent.maybe_train(task_id)

    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
