import os
import subprocess

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from typing import List
import uuid
from src.custom_transformer import FeedBackModel
from src.predict import single_prediction, batch_prediction
from transformers import AutoTokenizer
import glob
import wandb
import torch
from src import config, device, wandb_api

local_model_dir = config["PATHS"]["ARTIFACT_PATH"]
config = config["MODEL_CONFIG"]
label_cols = config['label_cols'].split(',')

print(wandb_api)
if len(glob.glob(os.path.join(local_model_dir, "*/pytorch_model.bin"))) == 0:
    print("DOWNLOAD ARTIFACT")
    run_name = str(uuid.uuid4()).split('-')[0]
    wandb.login(key=wandb_api)
    # instantiate deafault run
    run = wandb.init(id=run_name, resume=True)
    # Indicate the artifact we want to use with the use_artifact method.
    artifact = run.use_artifact(config["artifact_path"], type='model')
    # download locally the model
    artifact_dir = artifact.download()
    # delete path
    wandb.Api().run(run.path).delete()
# have to kill the wandb process
# because some zombie processes from wandb will still be running
# and the CI jobs will fail
subprocess.run(["pkill",  "-9",  "wandb"])
########################
# LOAD the local model #
########################
# it is a pytorch model: loaded as follows
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
model = FeedBackModel(config['model_name'])
local_model = glob.glob(os.path.join(local_model_dir, "*/pytorch_model.bin"))[0]
model.load_state_dict(
    torch.load(
        local_model,
        map_location=torch.device(device)
    )
)

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])


class SingleRequestId(BaseModel):
    essay: str


class MultipleRequestId(BaseModel):
    essays: List[str]


class EssayScores(BaseModel):
    text: str
    cohesion: float
    syntax: float
    vocabulary: float
    phraseology: float
    grammar: float
    conventions: float


class EssaysScores(BaseModel):
    batch: List[EssayScores]


app = FastAPI()


# Defining path operation for root endpoint
@app.get('/')
def index():
    return {
        # Alice in wonderland
        "text_examples": ["It's no use going back to yesterday, because I was a different person then"]
    }


@app.post("/single_essay", response_model=EssayScores)
def single_essay_scoring(request: SingleRequestId):
    text = request.essay
    scores = single_prediction(text, tokenizer, model)
    response = dict(zip(label_cols, scores))
    response["text"] = request.essay
    return EssayScores(**response)


@app.post("/multiple_essays", response_model=EssaysScores)
def multiple_essays_scoring(request: MultipleRequestId):
    essays = request.essays
    scores_dict = batch_prediction(essays, tokenizer, model)
    response = []
    for i, text in enumerate(essays):
        element = dict(zip(label_cols, scores_dict[i]))
        element["text"] = text
        response.append(element)
    response = {
        "batch": response
    }
    return EssaysScores(**response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
