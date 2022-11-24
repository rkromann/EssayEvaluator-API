from src.custom_transformer import EssayIterator
import torch
from . import config

config = config["MODEL_CONFIG"]

def single_prediction(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    with torch.no_grad():
        prediction = model(inputs["input_ids"], inputs["attention_mask"])
    return prediction.logits.squeeze(0).cpu().numpy()


def batch_prediction(test, tokenizer, model):
    # generate test embediings
    test_dataset = EssayIterator(test, tokenizer, is_train=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(config["train_batch_size"]),
        shuffle=False
    )
    input_ids, attention_mask = tuple(next(iter(test_dataloader)).values())
    input_ids = input_ids.to('cpu')
    attention_mask = attention_mask.to('cpu')
    # genreate predictions
    preds = model(input_ids, attention_mask)
    return preds.logits.detach().cpu().numpy()
