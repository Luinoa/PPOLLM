# api_server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import logging

# Import your LLMAgent class (which is assumed to provide generate_text and get_action_and_value methods)
from llm_policy import LLMAgent

app = FastAPI(title="LLM API for Kea")

# Instantiate your agent (adjust parameters as needed)
agent = LLMAgent(normalization_mode="word", load_8bit=False, batch_size=2)
# Set the model to evaluation mode for inference
agent.actor.eval()
agent.critic.eval()

# Define a Pydantic model for the generate_text API request
class GenerateTextRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True

# Define the response model for the generate_text endpoint
class GenerateTextResponse(BaseModel):
    generated_text: str

# Request and response schemas
class SingleObservation(BaseModel):
    prompt: str
    action: List[str]
    source: str  # Used to route to different models

class ActionRequest(BaseModel):
    text_obs: SingleObservation

class ActionResponse(BaseModel):
    action: List[int]

# Define the response model for the action and value endpoint.
class ActionAndValueResponse(BaseModel):
    action: list         # The chosen action (e.g., index or action string)
    log_probs: list      # The log probabilities for the chosen actions
    entropy: list        # The entropy of the action distributions
    value: list          # The estimated state values

@app.post("/generate_text", response_model=GenerateTextResponse)
def generate_text_endpoint(request: GenerateTextRequest):
    """
    Endpoint to generate text based on the prompt.
    This calls the agent.generate_text() function.
    """
    try:
        generated_text = agent.generate_text(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        return GenerateTextResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_action", response_model=ActionResponse)
def get_action_only(request: ActionRequest):
    """
    Return only action from model; log other outputs like log_probs and value.
    """
    try:
        # Prepare input for batch model API
        text_obs_batch = [{
            "prompt": request.text_obs.prompt,
            "action": request.text_obs.action
        }]

        # Call the model
        action, log_probs, entropy, value = agent.get_action_and_value(
            text_obs_batch, return_value=True
        )

        # Logging (or store to DB, file, etc.)
        logging.info(f"Log probs: {log_probs.tolist()}")
        logging.info(f"Entropy: {entropy.tolist()}")
        if value is not None:
            logging.info(f"Value: {value.tolist()}")

        # Return only action to client
        return ActionResponse(action=action.tolist())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the API service using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
