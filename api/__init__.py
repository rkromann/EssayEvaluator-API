import torch
from dotenv import load_dotenv, find_dotenv
import os
import configparser

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)
wandb_api = os.environ.get("WANDB_API")

config = configparser.ConfigParser()
config.read('config.ini')
