import copy
import math
import itertools
from datasets import Dataset, load_dataset
import torch
import contextlib
import evaluate
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, BertTokenizerFast, DataCollatorWithPadding, AutoConfig, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from typing import Any
from utils import read_files_for_text_classification, instantiate_logger, load_config, get_default_device, get_lr
from custom_loss import WeightedMulticlassFocalLoss

# Instantiate logger
logger = instantiate_logger("Training")

# Load config file
logger.info("Load Config.")
config = load_config("config.yaml", "train")


class SentClassificationEncoder:
    def __init__(self, model_checkpoint, label2id, encoder_max_len: int = 512):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, do_lower_case=True)      # load the tokenizer
        self.tokenizer.padding_side = "right"                                                         # setting tokenizer padding in the right side
        self.label2id = label2id
        self.encoder_max_len = encoder_max_len

    def encode(self, example: dict[str, Any]) -> dict[str, Any]:
        """Encode data into tokenized format that can be accepted by HuggingFace model

        Parameters
        ----------
        example : dict[str, Any]
            Element of dataset.
        encoder_max_len : int, optional
            Maximum tokens length, by default 512.

        Returns
        -------
        output : dict[str, Any]
            Inputs for model to consume in the form of tokens.
        """
        
        texts = copy.copy(example['text'])
        labels = copy.copy(example['label'])

        for i in range(len(labels)):
            labels[i] = self.label2id[labels[i]]
            

        encoder_inputs = self.tokenizer(texts, is_split_into_words = False, truncation = True, max_length = self.encoder_max_len, return_overflowing_tokens = False)
        encoder_inputs["labels"] = labels 
        
        return encoder_inputs
    

# function to write the information about saved(best) model, feel free to modify this function based on your need
def write_best_model_info(epoch: int, monitor_val: float, model_info_path: str):
    """Function to save model information(epoch and validation score) during training process

    Parameters
    ----------
    epoch : int
        Current epoch
    monitor_val : float
        Validation score in current epoch
    """

    f = open(f"{model_info_path}", "w")
    f.write(f"epoch:{epoch}\n")
    f.write(f"monitor_val:{monitor_val}\n")
    f.close()


# fit function for training, it takes number of epoch, the model we want to fine tuned, train set loader, valid/dev set loader, optimizer 
# and the metric we want to monitor as the parameters, this function monitor 'accuracy' by default 
def fit(model, train_loader: DataLoader, dev_loader: DataLoader, opt: torch.optim.Optimizer, scheduler, saved_model_folder: str, 
        model_info_path: str, monitor: str = "accuracy"):
    """Training loop function

    Parameters
    ----------
    model : _type_
        The model used in the training.
    train_loader : DataLoader
        Dataloader for train data.
    dev_loader : DataLoader
        Dataloader for dev data.
    opt : torch.optim.Optimizer
        Optimizer used in the training process.
    scheduler : _type_
        Scheduler to control learning rate during training.
    saved_model_folder : str
        Folder location to save model.
    model_info_path : str
        File location to save model training info.
    monitor : str, optional
        Metric used to monitor model performance, options:["accuracy", "precision", "recall", "f1"], by default "accuracy".
    """
   
    # Prepare metric
    metric_accuracy = evaluate.load("accuracy")
    metric_f1_prec_recall = evaluate.combine([evaluate.load("f1"),
                                              evaluate.load("precision"),
                                              evaluate.load("recall")])
    
    monitor_val_max = 0
    gradient_accumulation_steps = config.gradient_accumulation_steps
    num_epochs = config.num_epochs

    training_iterator = itertools.cycle(train_loader)
    num_samples_in_epoch = len(train_loader)
    remainder = num_samples_in_epoch % gradient_accumulation_steps
    remainder = remainder if remainder != 0  else gradient_accumulation_steps
    total_train_updates = math.ceil(num_samples_in_epoch / gradient_accumulation_steps)

    for epoch in range(num_epochs):
        model.train()
        train_updates_pbar = tqdm(range(total_train_updates))
        total_train_loss = 0.0
        total_train_items = 0
        for update_step in train_updates_pbar:
            # In order to correctly the total number of non-padded tokens on which we'll compute the cross-entropy loss
            # we need to pre-load the full local batch - i.e the next per_device_batch_size * accumulation_steps samples
            batch_samples = []
            num_batches_in_step = gradient_accumulation_steps if update_step != (total_train_updates - 1) else remainder
            for _ in range(num_batches_in_step):
                batch_samples += [next(training_iterator)]

            # get local num items in batch 
            local_num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            # to compute it correctly in a multi-device DDP training, we need to gather the total number of items in the full batch.
            num_items_in_batches = accelerator.gather(local_num_items_in_batch).sum().item()
            total_train_items += num_items_in_batches

            for i, batch in enumerate(batch_samples):
                # if we perform gradient accumulation in a multi-devices set-up, we want to avoid unnecessary communications when accumulating
                # cf: https://muellerzr.github.io/blog/gradient_accumulation.html
                if (i < len(batch_samples) - 1 and accelerator.num_processes > 1):
                    ctx = model.no_sync
                else:
                    ctx = contextlib.nullcontext


                with ctx():
                    input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                    outputs = model(input_ids = input_ids, attention_mask = attention_mask)

                    # Unscaled loss sum for tracking total cross entrophy loss correctly
                    raw_loss = criterion_loss(outputs.logits, labels)

                    # We multiply by num_processes because the DDP calculates the average gradient across all devices whereas dividing by num_items_in_batch already takes into account all devices
                    # Same reason for gradient_accumulation_steps, but this times it's Accelerate that calculate the average gradient across the accumulated steps
                    # Scale loss for gradient step computation
                    scaled_loss = (raw_loss * gradient_accumulation_steps * accelerator.num_processes) / num_items_in_batches
                    accelerator.backward(scaled_loss)

                    # Accumulate global raw loss (gathering unscaled losses across processes)
                    gathered_loss = accelerator.gather(raw_loss.detach()).sum().item()
                    total_train_loss += gathered_loss

                    # Gather prediction and target across all devices for metrics
                    preds = outputs.logits.argmax(dim=1)
                    gathered_preds = accelerator.gather_for_metrics(preds).detach().cpu().tolist()
                    gathered_labels = accelerator.gather_for_metrics(labels).detach().cpu().tolist()

                    metric_accuracy.add_batch(predictions=gathered_preds, references=gathered_labels)
                    metric_f1_prec_recall.add_batch(predictions=gathered_preds, references=gathered_labels)

            # Sync gradients and perform optimization steps once every gradient_accumulation_steps
            opt.step()
            scheduler.step()
            opt.zero_grad()

            average_on_fly_loss =  total_train_loss / total_train_items
            train_updates_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1), average_on_fly_loss, get_lr(opt)))

        # Compute final epoch loss normalized by total item(data)
        average_epoch_train_loss = total_train_loss/ total_train_items
        train_metric_results = {**metric_accuracy.compute(), **metric_f1_prec_recall.compute(average = "macro")} # when we call compute(), the predictions and references are cleared
        print("(Epoch {}) TRAIN LOSS:{:.4f} ACC:{:.4f} PREC:{:.4f} REC:{:.4f} F1:{:.4f} LR:{:.8f}".format((epoch+1), average_epoch_train_loss, train_metric_results["accuracy"], 
                                                                                                          train_metric_results["precision"], train_metric_results["recall"], 
                                                                                                          train_metric_results["f1"], get_lr(opt)))   

        model.eval()
        pbar = tqdm(dev_loader, leave=True, total=len(dev_loader))
        with torch.no_grad():
            total_dev_loss = 0.0
            total_dev_items = 0
            for i, batch in enumerate(pbar):
                input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                
                outputs = model(input_ids = input_ids, attention_mask = attention_mask)
                dev_loss = criterion_loss(outputs.logits, labels)

                local_dev_tokens = (batch["labels"].ne(-100)).sum()

                total_dev_loss += accelerator.gather_for_metrics(dev_loss).sum().item()
                total_dev_items += accelerator.gather_for_metrics(local_dev_tokens).sum().item()

                dev_loss_avg = total_dev_loss/total_dev_items if total_dev_items > 0 else 0.0

                preds = outputs.logits.argmax(dim=1)
                gathered_preds = accelerator.gather_for_metrics(preds).detach().cpu().tolist()
                gathered_labels = accelerator.gather_for_metrics(labels).detach().cpu().tolist()

                metric_accuracy.add_batch(predictions=gathered_preds, references=gathered_labels)
                metric_f1_prec_recall.add_batch(predictions=gathered_preds, references=gathered_labels)
                pbar.set_description("(Epoch {}) DEV LOSS:{:.4f}".format((epoch+1), dev_loss_avg))

            average_epoch_dev_loss = total_dev_loss/total_dev_items if total_dev_items > 0 else 0.0
            dev_metric_results = {**metric_accuracy.compute(), **metric_f1_prec_recall.compute(average = "macro")}
            print("(Epoch {}) VALIDLOSS:{:.4f} ACC:{:.4f} PREC:{:.4f} REC:{:.4f} F1:{:.4f} LR:{:.8f}".format((epoch+1), average_epoch_dev_loss, dev_metric_results["accuracy"], 
                                                                                                             dev_metric_results["precision"], dev_metric_results["recall"], 
                                                                                                             dev_metric_results["f1"], get_lr(optimizer)))         

            if dev_metric_results[monitor] > monitor_val_max: 
                monitor_val_max = dev_metric_results[monitor] 
                model.save_pretrained(f"{saved_model_folder}/best_model_rel/")
                write_best_model_info(epoch+1, dev_metric_results[monitor], model_info_path=model_info_path)
            else: 
                pass


if __name__ == "__main__":

    # Load dataset
    logger.info("Load Data")
    data_files = {"train": config.dataset.train_file_path, 
                  "dev": config.dataset.dev_file_path 
                  }

    dataset = load_dataset('csv', data_files=data_files, sep = "\t", on_bad_lines = 'warn', header = 0 if config.dataset.header_exist else None, names = ["text", "label"])

    # Preprocess data
    logger.info("Preprocess data.")

    # label map, to mapping the labels into numerical values, useful in the preprocessing data and getting the output label
    unique_label = set(dataset['train']['label'])
    label2id = {tag: id for id, tag in enumerate(unique_label)}
    id2label = {id: tag for tag, id in label2id.items()}

    logger.info("Create Encoder.")
    encoder = SentClassificationEncoder(config.model_checkpoint, label2id)
    tokenized_dataset = dataset.map(encoder.encode, 
                                    batched = True, 
                                    remove_columns = dataset['train'].column_names
                                    )
    
    tokenized_dataset.set_format(type = 'torch', 
                                 columns = ['input_ids', 'attention_mask', 'labels'], 
                                 output_all_columns = True
                                 )

    # intatiate collator function, collate data into batch and padding based on the longest sequence on the batch, this way we can fine tune model faster compared
    # to padding all data into the same length 
    data_collator = DataCollatorWithPadding(tokenizer=encoder.tokenizer)

    # load dataset into dataloader, for more details about DataLoader, please check the pytorch documentation about DataLoader
    train_dl = DataLoader(tokenized_dataset['train'], 
                          batch_size=config.batch_size, 
                          collate_fn=data_collator, 
                          shuffle=True
                          )
    
    dev_dl = DataLoader(tokenized_dataset['dev'], 
                        batch_size=config.batch_size,   
                        collate_fn=data_collator, 
                        shuffle=False
                        )

    num_training_step = math.ceil(len(train_dl) / config.gradient_accumulation_steps) * config.num_epochs

    logger.info("Preprocess end.")

    # Load model
    logger.info("Load model")

    # set model config
    model_config = AutoConfig.from_pretrained(config.model_checkpoint,
                                              _num_labels=len(unique_label),
                                              id2label=id2label,
                                              label2id=label2id,
                                              finetuning_task = "text-classification"
                                              )

    model = AutoModelForSequenceClassification.from_pretrained(config.model_checkpoint, config=model_config)

    # calculate class weight since the data is imbalance, se will use this later for loss function so when training we give more weight in smaller class size 
    # Prepare training
    logger.info("Prepare for training.")

    target = torch.tensor(tokenized_dataset['train']['labels'])
    class_count = torch.bincount(target)
    class_weights = 1.0 / class_count
    class_weights = class_weights / class_weights.sum()

    # Create custom loss function and put the weights in it, so in training process we factor the class imbalance in the loss and update the model weights accordingly
    if config.focal_loss:
        logger.info("Loss is using focal loss")
        criterion_loss =  WeightedMulticlassFocalLoss(alpha=class_weights, reduction="sum")
    else:
        logger.info("Loss is using CE loss")
        criterion_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")  # create custom loss and put it in the device

    # set optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=50, num_training_steps=num_training_step+100) # +100 because I don't want the lr goes to 0

    # Set accelerator
    accelerator = Accelerator(mixed_precision= "fp16" if config.fp16 else "no", gradient_accumulation_steps=config.gradient_accumulation_steps)
    # load model, opt, dataloader to optimizer, optional: you can also load scheduler
    model, optimizer, scheduler, train_dl, dev_dl = accelerator.prepare(model, optimizer, scheduler, train_dl, dev_dl)

    # Training
    logger.info("Begin training loop.")

    # Training loop

    # training and saved the best model based on f1 score
    fit(model, train_dl, dev_dl, optimizer, scheduler, saved_model_folder=config.saved_model_folder, 
        model_info_path= config.model_info_path, monitor='f1')

    # save tokenizer at the end of training process
    encoder.tokenizer.save_pretrained(f"{config.saved_model_folder}/best_model_rel/")

    logger.info("Training end.")