import numpy as np
import os
import evaluate 
import pickle
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from utils import read_wnut, instantiate_logger, load_config, calculate_class_weights
from transformers.tokenization_utils_base import BatchEncoding
from custom_class import FocalLossTrainer

logger = instantiate_logger("Token Class Training")

logger.info("load config")
config =  load_config("config.yaml", "train")

os.environ["TENSORBOARD_LOGGING_DIR"] = config.tensorboard_logging_dir


class WNUTDataset(torch.utils.data.Dataset):
    """Custom dataset class inherited from torch Dataset for WNUT data.

    Attributes
    ----------
    encodings : BatchEncoding
        Output of tokenizer function, encoded data of sentence and tags.
    labels : list[str]
        list of list of tags
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
# function to encode label data
def encode_tags(tags: list[list[str]], encodings: BatchEncoding):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

# need to create custom metrics for token classification task. Normally, it used seqeval metric 
# using seqeval metric from datasets library

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens) where the code is -100
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def save_base_model_and_wrap_with_peft(base_model, tokenizer, config):
    """Function to save base model and tokenizer following by creating peft model for LoRA training

    Parameters
    ----------
    base_model : _type_
        base model used for fine tuning. 
    tokenizer : _type_
        tokenizer of base model.
    config : _type_
        training config containing lora config.

    Returns
    -------
    out
       peft model that will be passed to Trainer. 
    """
    # save base model and tokenizer in one folder
    base_model.save_pretrained(config.base_model_save_path)
    tokenizer.save_pretrained(config.base_model_save_path)

    # Set LoRA config
    lora_config = LoraConfig(init_lora_weights=config.lora_config.init_lora_weights,
                             r=config.lora_config.r,
                             lora_alpha=config.lora_config.lora_alpha,
                             target_modules=config.lora_config.target_modules,
                             lora_dropout=config.lora_config.lora_dropout,
                             bias=config.lora_config.bias,
                             task_type=TaskType.TOKEN_CLS
                             )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model



if __name__ == "__main__":

    # read data
    logger.info("read training data")
    train_texts, train_tags = read_wnut(f'{config.train_data_path}')
    val_texts, val_tags = read_wnut(f'{config.valid_data_path}')

    # get unique tags, map tag to id and vice versa 
    unique_tags = set(tag for doc in train_tags + val_tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    
    # !!! don't forget to save the label map, so that we can use it to translate the model output, we will put these into the "idtag" folder
    idtag_dir = "idtag"
    with open(f"{idtag_dir}/tag2id_pkl", 'wb') as f1:
        pickle.dump(tag2id, f1)
        
    with open(f'{idtag_dir}/id2tag_pkl', 'wb') as f2:
        pickle.dump(id2tag, f2)  
    logger.info("preprocess training data done!")

    # load tokenizer
    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_checkpoint, do_lower_case=True, max_length = config.encoder_max_len)
    logger.info("load tokenizer done!")

    logger.info("begin encoding data")
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=False)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=False)
    
    # encode labels
    train_labels = encode_tags(train_tags, train_encodings)
    val_labels = encode_tags(val_tags, val_encodings)

    # calculate class weight using train labels
    class_weights = calculate_class_weights(train_labels)

    train_encodings.pop("offset_mapping") # we don't want to pass this to the model
    val_encodings.pop("offset_mapping")
    train_dataset = WNUTDataset(train_encodings, train_labels)
    val_dataset = WNUTDataset(val_encodings, val_labels)
    logger.info("encoding done!")

    logger.info("prepare fine tuning!")
    # Create data collator, this will automatically handle how we will treat the batch, 
    # in this case we do the padding for each batch following the longest sequence in the batch
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    logging_steps = int(np.floor(len(train_texts)/config.train_batch)) # train loss is logged every epoch

    training_args = TrainingArguments(
        output_dir= config.trainer_args.output_dir,             # output directory
        eval_strategy = config.trainer_args.eval_strategy ,
        num_train_epochs = config.trainer_args.num_train_epochs,               # total number of training epochs
        learning_rate = config.trainer_args.learning_rate,
        per_device_train_batch_size = config.train_batch,  
        per_device_eval_batch_size = config.eval_batch,
        gradient_accumulation_steps= config.trainer_args.gradient_accumulation_steps,
        average_tokens_across_devices= config.trainer_args.average_tokens_across_devices,   
        weight_decay = config.trainer_args.weight_decay,                # strength of weight decay
        load_best_model_at_end = config.trainer_args.load_best_model_at_end,      # load the best model after the end of training
        metric_for_best_model = config.trainer_args.metric_for_best_model,
        greater_is_better = config.trainer_args.greater_is_better,
        save_total_limit = config.trainer_args.save_total_limit,               # save only one model
        dataloader_drop_last = config.trainer_args.dataloader_drop_last,
        save_strategy = config.trainer_args.save_strategy,              # the value must be same with eval_strategy
        logging_strategy = config.trainer_args.logging_strategy, 
        remove_unused_columns= config.trainer_args.remove_unused_columns,  
        report_to= "tensorboard"
    )

    logger.info("load model for fine tuning")
    model_config = AutoConfig.from_pretrained(
        config.base_model_checkpoint,
        _num_labels=len(unique_tags),
        id2label=id2tag,
        label2id=tag2id
    )

    model = AutoModelForTokenClassification.from_pretrained(config.base_model_checkpoint, config=model_config)

    if config.lora_enable:
        logger.info("Save base model, tokenizer and create peft model")
        model = save_base_model_and_wrap_with_peft(model, tokenizer, config)

    # use focal loss in case of heavy data imbalance case.
    if config.focal_loss:
        trainer = FocalLossTrainer(
                                    class_weights=class_weights,         # class weight
                                    model=model,                         # the instantiated 🤗 Transformers model to be trained
                                    args=training_args,                  # training arguments, defined above
                                    train_dataset=train_dataset,         # training dataset
                                    eval_dataset=val_dataset,            # evaluation dataset
                                    compute_metrics=compute_metrics,     # custom metrics  
                                    processing_class = tokenizer,
                                    data_collator =  data_collator,
                                )
    else:
        trainer = Trainer(
                            model=model,                         # the instantiated 🤗 Transformers model to be trained
                            args=training_args,                  # training arguments, defined above
                            train_dataset=train_dataset,         # training dataset
                            eval_dataset=val_dataset,            # evaluation dataset
                            compute_metrics=compute_metrics,     # custom metrics  
                            processing_class = tokenizer,
                            data_collator =  data_collator
                        )

    # begin training
    logger.info("begin fine tuning!")
    trainer.train()
    logger.info("fine tuning process done")







    
    



    