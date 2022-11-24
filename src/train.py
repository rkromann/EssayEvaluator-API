from . import config, wandb_api
import gc

from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer, logging,
    AdamW, get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    Trainer, TrainingArguments
)
import random
import wandb
from src.custom_transformer import EssayIterator, FeedBackModel, CustomTrainer, compute_metrics
# Data Collator for Dynamic Padding

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
wandb.login(key=wandb_api)

# Data Collator for Dynamic Padding
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
# init predictions by fold
config = config["MODEL_CONFIG"]

predictions = {}
def cv_train(train, test):
    for fold in range(0, config['folds']):
        print('-'*50)
        print(f" ---- Fold: {fold} ----")
        print('-'*50)
        run = wandb.init(project="Feedback3-deberta",
                         config=config,
                         job_type='train',
                         group="FB3-BASELINE-MODEL",
                         tags=[config['model_name'], config['loss_type'], "10-epochs"],
                         name=f'FB3-fold-{fold}',
                         anonymous='must')
        # the reset index is VERY IMPORTANT for the Dataset iterator
        df_train = train[train.fold != fold].reset_index(drop=True)
        df_valid = train[train.fold == fold].reset_index(drop=True)
        # create iterators
        train_dataset = EssayIterator(df_train, tokenizer)
        valid_dataset = EssayIterator(df_valid, tokenizer)
        # init model
        model = FeedBackModel(config['model_name'])
        model.to(config['device'])

        # SET THE OPITMIZER AND THE SCHEDULER
        # no decay for bias and normalization layers
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": config['weight_decay'],
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_parameters, lr=config['learning_rate'])
        num_training_steps = (len(train_dataset) * int(config['epochs'])) // (int(config['train_batch_size']) * int(config['n_accumulate']))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1*num_training_steps,
            num_training_steps=num_training_steps
        )
        # DEFINE TrainingArguments
        training_args = TrainingArguments(
            output_dir=f"outputs-{fold}/",
            evaluation_strategy="epoch",
            per_device_train_batch_size=config['train_batch_size'],
            per_device_eval_batch_size=config['valid_batch_size'],
            num_train_epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            gradient_accumulation_steps=config['n_accumulate'],
            seed=config['seed'],
            #       group_by_length=True,
            max_grad_norm=config['max_grad_norm'],
            metric_for_best_model='eval_MCRMSE',
            load_best_model_at_end=True,
            greater_is_better=False,
            save_strategy="epoch",
            save_total_limit=1,
            report_to="wandb",
            label_names=["labels"]
        )
        # CREATE THE TRAINER
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=collate_fn,
            optimizers=(optimizer, scheduler),
            compute_metrics=compute_metrics
        )
        # LAUNCH THE TRAINER
        trainer.train()


        #INFERENCE
        test_dataset = EssayIterator(test, tokenizer, is_train=False)
        predictions[fold] = trainer.predict(test_dataset)
        # Save model artifact
        # create model artifact
        model_artifact = wandb.Artifact(f'FB3-fold-{fold}', type="model",
                                        description=f"MultilabelStratified - fold--{fold}")
        # save locally the model - it would create a local dir
        trainer.save_model(f'fold-{fold}')
        # add the local dir to the artifact
        model_artifact.add_dir(f'fold-{fold}')
        # log artifact
        # it would save the artifact version and declare the artifact as an output of the run
        run.log_artifact(model_artifact)

        run.finish()

        del model
        _ = gc.collect()
