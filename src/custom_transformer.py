from . import config, device
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer, logging,
    AdamW, get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    Trainer, TrainingArguments
)
import random

from transformers.modeling_outputs import SequenceClassifierOutput
import pandas as pd

logging.set_verbosity_error()
logging.set_verbosity_warning()
config = config["MODEL_CONFIG"]
label_cols = config['label_cols'].split(',')


class EssayIterator(torch.utils.data.Dataset):
    def __init__(self, essays, tokenizer, labels=label_cols, is_train=True):
        self.essays = essays
        self.tokenizer = tokenizer
        self.max_seq_length = int(config["max_length"])
        self.labels = labels
        self.is_train = is_train

    def __getitem__(self, idx):
        if isinstance(self.essays, pd.DataFrame):
            essay = self.df.loc[idx, 'full_text']
        else:
            essay = list(self.essays)[idx]
        tokens = self.tokenizer(
            essay,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        result = {
            'input_ids': tokens['input_ids'].to(device).squeeze(),
            'attention_mask': tokens['attention_mask'].to(device).squeeze()
        }
        if self.is_train:
            result["labels"] = torch.tensor(
                self.essays.loc[idx, self.labels].to_list(),
            ).to(device)

        return result

    def __len__(self):
        return len(self.essays)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, len(label_cols))

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, attention_mask)
        out = self.drop(out)
        outputs = self.fc(out)
        return SequenceClassifierOutput(logits=outputs)


class RMSELoss(nn.Module):
    """
    Code taken from Y Nakama's notebook (https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
    """

    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, predictions, targets):
        loss = torch.sqrt(self.mse(predictions, targets) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        loss_func = RMSELoss(reduction='mean')
        loss = loss_func(outputs.logits.float(), inputs['labels'].float())
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    colwise_rmse = np.sqrt(np.mean((labels - predictions) ** 2, axis=0))
    res = {
        f"{analytic.upper()}_RMSE": colwise_rmse[i]
        for i, analytic in enumerate(label_cols)
    }
    res["MCRMSE"] = np.mean(colwise_rmse)
    return res
