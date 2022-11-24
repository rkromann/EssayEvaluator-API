import configparser
import sys
import os
import torch
from dotenv import load_dotenv, find_dotenv

#sys.path.append(os.path.abspath('..'))

config = configparser.ConfigParser()
config.read('config.ini')

device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)
wandb_api = os.environ.get("WANDB_API")