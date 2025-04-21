# api_server.py
# PPO + LangChain agent server for multi-task asynchronous interaction
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any
from ppo_server_agent import PPOAgentServer

# Define request models
class StepRequest(BaseModel):
    task_id: str
    obs: Dict[str, Any]

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

class ActionResponse:
    task_id: str
    action: int

# In-memory task states
app = FastAPI()

@app.post("/attach")
async def attach_task():
    task_id = ppo_agent.new_task()
    return {"status": "attached", "task_id": task_id}


@app.post("/detach")
async def detach_task(task_id: str):
    if ppo_agent.check_task(task_id):
        ppo_agent.close_task(task_id)
        return {"status": "detached", "task_id": task_id}
    else:
        return {"error": "Task ID not found"}

@app.post("/ask")
async def ask_for_step(task_id: str):
    if not ppo_agent.check_task(task_id):
        return {"error": "Invalid task ID"}
    if ppo_agent.ask_for_step(task_id):
        return {"status": "ok"}
    else:
        return {"error": "This cannot happen"}

@app.post("/step")
async def step(req: StepRequest):
    task_id = req.task_id
    obs = req.obs

    if not ppo_agent.check_task(task_id):
        return {"error": "Invalid task ID"}

    response = ppo_agent.step(task_id, obs)
    return {"task_id": task_id, "status": "ok","action": int(response)}


@app.post("/generate", response_class=PlainTextResponse)
def generate_text(request: GenerationRequest):
    task_id = request.task_id
    # Placeholder logic for generation (simulate response)
    response = f"[Mock] Generated response to: {request.prompt}"

    return {"task_id": task_id, "response": response}


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    task_id = req.task_id
    reward = req.reward
    done = req.done

    if not ppo_agent.check_task(task_id):
        return {"error": "Missing task state"}

    # Process feedback
    ppo_agent.feedback(task_id, reward, done)

    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Algorithm specific arguments, mostly only useful in training
    parser.add_argument("--policy-learning-rate", type=float, default=1e-6,
                        help="the learning rate of the optimizer")
    parser.add_argument("--value-learning-rate", type=float, default=3e-5,
                        help="the learning rate of the optimizer")

    """
    # Not to consider this for now
    parser.add_argument("--anneal-lr", dest="anneal_lr", action="store_true", help="Enable learning rate annealing")
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false",
                        help="Disable learning rate annealing")
    parser.set_defaults(anneal_lr=True)
    """

    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--policy-minibatch-size", type=int, default=8,
                        help="the number of mini-batches")
    parser.add_argument("--value-minibatch-size", type=int, default=4,
                        help="the number of mini-batches")

    parser.add_argument("--norm-adv", dest="norm_adv", action="store_true", help="Enable advantages normalization")
    parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false", help="Disable advantages normalization")
    parser.set_defaults(norm_adv=True)

    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", dest="clip_vloss", action="store_true",
                        help="Use a clipped loss for the value function (default: True)")
    parser.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false",
                        help="Do not use a clipped loss for the value function")
    parser.set_defaults(clip_vloss=True)

    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")

    parser.add_argument('--gradient-checkpointing-steps', action='store', type=int, default=8,
                        help='The number of steps for gradient checkpointing')

    parser.add_argument('--record-path', action='store', type=str, default="weights/PPO",
                        help='The path to save the tensorboard results')


    parser.add_argument('-p', '--port', action='store', type=int, default=8000,
                        help="Port number for the server")
    parser.add_argument('--forward-batch', action='store', type=int, default=2,
                        help='The size of batches in each forward pass') # Forward in parallel, very memory sensitive

    parser.add_argument('--training-batch', action='store', type=int, default=8,
                        help='The size of training batches per session') # How many samples to train on per session (or per task)
    parser.add_argument("--update-epoches", type=int, default=1,
                        help="the number of epochs to update the policy")
    parser.add_argument("--warmup-updates", action="store", type=int, default=0,
                        help="The number of warmup updates before training starts") # Only for training critic

    parser.add_argument("-i", "--inference", dest="inference", action="store_true",
                        help="Run in inference-only mode")
    parser.add_argument("-t", "--training", dest="inference", action="store_false",
                        help="Run in training mode")
    parser.set_defaults(inference=True)

    args = parser.parse_args()

    # Instantiate the PPO agent with the LLM agent (inference mode by default)
    ppo_agent = PPOAgentServer(args)
    if args.inference:
        print("[INFO] Running in inference-only mode")
    else:
        print("[INFO] Running in training mode")

    print(f"[INFO] Starting server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)