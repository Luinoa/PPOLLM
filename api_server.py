# ppo_server.py
# PPO + LangChain agent server for multi-task asynchronous interaction
import uuid

from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import uvicorn
from typing import Dict, Any

from llm_policy import LLMAgent
from ppo_server_agent import PPOAgentServer

# Define request models
class StepRequest(BaseModel):
    task_id: str
    obs: Dict[str, Any]

class FeedbackRequest(BaseModel):
    task_id: str
    next_obs: Dict[str, Any]
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

# Instantiate the PPO agent with the LLM agent (inference mode by default)
agent = LLMAgent(normalization_mode="word", batch_size=2, inference=True)
ppo_agent = PPOAgentServer(agent=agent)

# In-memory task states
TASK_STATE = {}
app = FastAPI()

@app.post("/attach")
async def attach_task():
    task_id = ppo_agent.new_task()
    TASK_STATE[task_id] = {"status": "attached", "task_id": task_id}
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

    response = ppo_agent.step(task_id, obs)
    return response


@app.post("/generate", response_class=PlainTextResponse)
def generate_text(request: GenerationRequest):
    task_id = request.task_id
    # Placeholder logic for generation (simulate response)
    response = f"[Mock] Generated response to: {request.prompt}"
    logprob = torch.tensor([0.0])
    value = torch.tensor([0.5])

    # Store the generated response in the task state
    if task_id in TASK_STATE:
        TASK_STATE[task_id]["response"] = response
        TASK_STATE[task_id]["logprob"] = logprob
        TASK_STATE[task_id]["value"] = value

    return {"task_id": task_id, "response": response}


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    task_id = req.task_id
    reward = req.reward
    done = req.done

    if task_id not in TASK_STATE:
        return {"error": "Missing task state"}

    # Process feedback
    ppo_agent.provide_feedback(task_id, reward, done)

    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true", help="Run in inference-only mode")
    args = parser.parse_args()

    agent.train_mode = not args.inference
    mode = "inference" if args.inference else "training"
    print(f"[MODE] Running in {mode} mode")

    uvicorn.run(app, host="0.0.0.0", port=8000)