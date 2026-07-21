import os
import copy
import yaml
from datasets import Dataset
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, BertTokenizerFast, DataCollatorWithPadding, AutoConfig
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Any
from utils import read_files_for_text_classification, instantiate_logger, get_default_device
from custom_loss import WeightedMulticlassFocalLoss

# Instantiate logger
logger = instantiate_logger("Training")


def encode(example: dict[str, Any], encoder_max_len: int = 512) -> dict[str, Any]:
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
    
    text = copy.copy(example['text'])
    label = copy.copy(example['label'])

    for i in range(len(label)):
        label[i] = label2id[label[i]]
        

    encoder_inputs = tokenizer(text, is_split_into_words = False, truncation = True, max_length = encoder_max_len, return_overflowing_tokens = False)
    input_ids = encoder_inputs['input_ids']
    input_attention = encoder_inputs['attention_mask']
    outputs = {'input_ids':input_ids, 'attention_mask': input_attention, "labels": label}
    
    return outputs


# function to get the learning rate value
def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate value of the optimizer

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer used for backprop

    Returns
    -------
    out : float
        Learning rate value
    """

    for param_group in optimizer.param_groups:
        return param_group['lr']
    

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

def fit(num_epochs: int, num_batch_per_epoch: int, model: AutoModelForSequenceClassification, train_loader: DataLoader, valid_loader: DataLoader, 
        opt: torch.optim.Optimizer, saved_model_folder: str, model_info_path: str, monitor: str = 'acc', fp16: bool = False):
    """Training loop function

    Parameters
    ----------
    num_epochs : int
        Number of epoch.
    num_batch_per_epoch : int
        Number of batch per epoch.
    model : AutoModelForSequenceClassification
        The model used in the training.
    train_loader : DataLoader
        Dataloader for train data.
    valid_loader : DataLoader
        Dataloader for validation data.
    opt : torch.optim.Optimizer
        Optimizer used in the training process.
    monitor : str, optional
        Metric used to monitor model performance, by default 'acc'.
    fp16 : bool, optional
        Using fp16 for automatic mixed precision(AMP) in training, by default False
    """
   
    monitor_val = 0
    monitor_val_max = 0

    for epoch in range(num_epochs):        
        model.train()
        train_loss = 0
        list_train_true_labels = []
        list_train_pred_label = []

        if num_batch_per_epoch is None:
            num_batch_per_epoch = len(train_loader) 

        train_pbar = tqdm(train_loader, leave=True, total=num_batch_per_epoch)

        if fp16:
            scaler = torch.amp.GradScaler(device.type)

        for i, batch_data in enumerate(train_pbar):
            input_ids, attention_mask, labels = batch_data["input_ids"], batch_data["attention_mask"], batch_data["labels"]
            list_train_true_labels += labels.tolist()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            opt.zero_grad()

            if fp16:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    output = model(input_ids = input_ids, attention_mask = attention_mask)
                    loss = criterion_loss(output.logits, labels)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                output = model(input_ids = input_ids, attention_mask = attention_mask)
                loss = criterion_loss(output.logits, labels)
                loss.backward()
                opt.step()

            train_loss += loss.item()
            logits = output.logits
            logits_labels = torch.argmax(logits, dim=1).tolist()
            list_train_pred_label += logits_labels

            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1), train_loss/(i+1), get_lr(opt)))

        acc = accuracy_score(list_train_true_labels, list_train_pred_label)
        precision, recall, f1, _ = precision_recall_fscore_support(list_train_true_labels, list_train_pred_label, average='macro')
        print("(Epoch {}) TRAIN LOSS:{:.4f} ACC:{:.4f} PREC:{:.4f} REC:{:.4f} F1:{:.4f} LR:{:.8f}".format((epoch+1), train_loss/(i+1), acc, precision, recall, 
                                                                                                          f1, get_lr(optimizer)))

        model.eval()
        pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
        with torch.no_grad():
            val_loss = 0
            list_val_true_labels = []
            list_val_pred_label = []
            for idx, data in enumerate(pbar):
                input_ids, attention_mask, labels = data["input_ids"], data["attention_mask"], data["labels"]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                list_val_true_labels += labels.tolist()
                labels = labels.to(device)
                opt.zero_grad()

                if fp16:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                        output = model(input_ids = input_ids, attention_mask = attention_mask)
                        loss = criterion_loss(output.logits, labels)    
                else:
                    output = model(input_ids = input_ids, attention_mask = attention_mask)
                    loss = criterion_loss(output.logits, labels)
                    
                logits = output.logits
                logits_labels = torch.argmax(logits, dim=1).tolist()
                list_val_pred_label += logits_labels
                val_loss += loss.item()
                pbar.set_description("(Epoch {}) VALID LOSS:{:.4f}".format((epoch+1), val_loss/(i+1)))

        acc = accuracy_score(list_val_true_labels, list_val_pred_label)
        precision, recall, f1, _ = precision_recall_fscore_support(list_val_true_labels, list_val_pred_label, average='macro')
        print("(Epoch {}) VALIDLOSS:{:.4f} ACC:{:.4f} PREC:{:.4f} REC:{:.4f} F1:{:.4f} LR:{:.8f}".format((epoch+1), val_loss/(i+1), acc, precision, recall, 
                                                                                                          f1, get_lr(optimizer)))        
        if monitor == "acc":
            monitor_val = acc
        elif monitor == "precision":
            monitor_val = precision
        elif monitor == "recall":
            monitor_val = recall
        elif monitor == "f1":
            monitor_val = f1
        else:
            monitor_val = acc

        if monitor_val > monitor_val_max: 
            monitor_val_max = monitor_val 
            model.save_pretrained(f"{saved_model_folder}/best_model_rel/")
            write_best_model_info(epoch+1, monitor_val, model_info_path=model_info_path)
        else: 
            pass



if __name__ == "__main__":

    # Load config file
    logger.info("Load Config.")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)["train"]


    batch_size = config["batch_size"]                      # the number of data for every batch in the training process
    encoder_max_len = config["encoder_max_length"]         # maximum vector length for every input 
    fp16 = config["fp16"]                                  # set mixed precision point, fp16 gives more speed if your card rtx 30 series above, if your gpu not support fp16 natively the training speed is slower
    train_file_path = config["file_path"]
    train_file_columns = config["file_columns"]
    model_checkpoint = config["model_checkpoint"]
    saved_model_folder = config["saved_model_folder"]
    model_info_path = config["model_info_path"]
    num_epochs = config["num_epochs"]                      # num of epoch
    num_batch_per_epoch = config["num_batch_per_epoch"]    # num of batch per epoch, set it None if we want to set one full loop of train data count as one epoch    
    focal_loss = config["focal_loss"]


    # create folder for saving model
    logger.info("Creating model folder.")
    try: 
        os.mkdir(saved_model_folder)
    except FileExistsError as fee:
        print("Folder already exist.")
        logger.info("Folder already exists.")

    # Load training data
    logger.info("Load training data.")

    train_data_df = read_files_for_text_classification(file_path=train_file_path, 
                                                       text_column_name=train_file_columns["text"], 
                                                       label_column_name=train_file_columns["label"]
                                                       )

    num_labels = len(train_data_df['text'].unique())

    # Preprocess data
    logger.info("Preprocess data.")

    # split train set to get the dev set  

    train_data_df, dev_data_df= train_test_split(train_data_df, 
                                                 test_size=0.3, 
                                                 stratify = train_data_df.label, 
                                                 random_state = 42
                                                 )

    # load the data into dataset object, this class from huggingface dataset library is useful to make it easier for preprocessing data
    # and transform it into the desireable input for the model

    dataset_train = Dataset.from_pandas(train_data_df[['text', 'label']])
    dataset_dev = Dataset.from_pandas(dev_data_df[['text', 'label']])

    logger.info("Loading tokenizer.")

    # Load the tokenizer used to transform the sentence into the desireable input for the bert model

    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, do_lower_case=True) # load the tokenizer
    tokenizer.padding_side = "right"                                                    # setting tokenizer padding in the right side

    # label map, to mapping the labels into numerical values, useful in the preprocessing data and getting the output label
    unique_label = set(train_data_df['label'].unique())
    label2id = {tag: id for id, tag in enumerate(unique_label)}
    id2label = {id: tag for tag, id in label2id.items()}

    tokenized_dataset_train = dataset_train.map(encode, 
                                                batched = True, 
                                                remove_columns = dataset_train.column_names)
    
    tokenized_dataset_train.set_format(type = 'torch', 
                                       columns = ['input_ids', 'attention_mask', 'labels'], 
                                       output_all_columns = True)
    
    tokenized_dataset_dev = dataset_dev.map(encode, 
                                            batched = True, 
                                            remove_columns = dataset_dev.column_names)
    
    tokenized_dataset_dev.set_format(type = 'torch', 
                                     columns = ['input_ids', 'attention_mask', 'labels'], 
                                     output_all_columns=True)

    # intatiate collator function, collate data into batch and padding based on the longest sequence on the batch, this way we can fine tune model faster compared
    # to padding all data into the same length 
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # load dataset into dataloader, for more details about DataLoader, please check the pytorch documentation about DataLoader
    train_dl = DataLoader(tokenized_dataset_train, 
                          batch_size=batch_size, 
                          collate_fn=data_collator, 
                          shuffle=True
                          )
    
    dev_dl = DataLoader(tokenized_dataset_dev, 
                        batch_size=batch_size,   
                        collate_fn=data_collator, 
                        shuffle=False
                        )

    logger.info("Preprocess end.")

    # Load model
    logger.info("Load model")

    # set model config
    model_config = AutoConfig.from_pretrained(model_checkpoint,
                                              _num_labels=len(unique_label),
                                              id2label=id2label,
                                              label2id=label2id,
                                              finetuning_task = "text-classification"
                                              )

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=model_config)

    # calculate class weight since the data is imbalance, se will use this later for loss function so when training we give more weight in smaller class size 

    # Prepare training
    logger.info("Prepare for training.")

    target = torch.tensor(tokenized_dataset_train['labels'])
    class_count = torch.bincount(target)
    class_weights = 1.0 / class_count
    class_weights = class_weights / class_weights.sum()

    # Create custom loss function and put the weights in it, so in training process we factor the class imbalance in the loss and update the model weights accordingly
    if focal_loss:
        logger.info("Loss is using focal loss")
        criterion_loss =  WeightedMulticlassFocalLoss(alpha=class_weights)
    else:
        logger.info("Loss is using CE loss")
        criterion_loss = nn.CrossEntropyLoss(weight=class_weights)  # create custom loss and put it in the device

    # get device
    device = get_default_device()

    # set optimizer and put model, criterion loss into device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model = model.to(device)
    criterion_loss = criterion_loss.to(device)

    # !!! don't forget to save the label map, so that we can use it to translate the model output, we will put these into the "model" folder
    with open(f"{saved_model_folder}/label2id_pkl", 'wb') as f1:
        pickle.dump(label2id, f1)
        
    with open(f'{saved_model_folder}/id2label_pkl', 'wb') as f2:
        pickle.dump(id2label, f2)  

    # Training
    logger.info("Begin training loop.")

    # Training loop

    # training and saved the best model based on f1 score
    fit(num_epochs, num_batch_per_epoch, model, train_dl, dev_dl, optimizer, saved_model_folder=saved_model_folder, 
        model_info_path= model_info_path, monitor='f1', fp16=fp16)

    # save tokenizer at the end of training process
    tokenizer.save_pretrained(f"{saved_model_folder}/best_model_rel/")

    logger.info("Training end.")