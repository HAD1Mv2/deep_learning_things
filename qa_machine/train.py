import os
import copy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from utils import instantiate_logger, load_config


logger = instantiate_logger("Training")

logger.info("Load Train Config")
config = load_config("config.yaml", "train")

os.environ["TENSORBOARD_LOGGING_DIR"] = config.tensorboard_logging_dir

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_checkpoint)

# encode function for encoding data
def encode(example, encoder_max_len=config.encoder_max_len):
    
    texts = copy.copy(example['passage'])
    questions= copy.copy(example['question'])
    answers = copy.copy(example['seq_label'])
    answers_text = [None for i in range(len(texts))]
    
    # since the data type in string, we need to convert the data into list
    for i in range(len(texts)):
        t = texts[i].strip("']['").split("', '")
        a = answers[i].strip("']['").split("', '")
        q = questions[i].strip("']['").split(", ")
        q = [b.strip('\'"') for b in q]
        
        if len(t)!=len(a):
            t = texts[i].strip("']['").split(", ")
            t_swap = []
            for b in t:
                if b[0] == '"':
                    t_swap.append(b.strip('"'))
                else:
                    t_swap.append(b.strip("\'"))
            t = t_swap
            
        assert len(t)==len(a)
        
        answers[i] = a
        texts[i] = t
        questions[i] = q
        answers_text[i] = " ".join(list(np.array(t)[np.array(a) != 'O']))
        

    # encode after converting the data
    encoder_inputs = tokenizer(questions, texts, is_split_into_words=True, truncation="only_second", max_length=encoder_max_len, padding=False, 
                               return_overflowing_tokens=True, return_offsets_mapping=True, stride=config.doc_stride)

    input_ids = encoder_inputs['input_ids']
    input_attention = encoder_inputs['attention_mask']
    offset_mapping = encoder_inputs.pop("offset_mapping") 

    # get the start and end index position of the answers for the questions  
    start_answer_token_positions = []
    end_answer_token_positions = []
    
    for i in range(len(texts)):
        sequence_ids = encoder_inputs.sequence_ids(i)
        
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
            
        token_end_index = len(input_ids[i]) - 1
        while sequence_ids[token_end_index] != 1 :
            token_end_index -= 1
            
        start_token_answer = 0
        while answers[i][start_token_answer] == 'O':
            if offset_mapping[i][token_start_index + start_token_answer +1][0] == 0:
                start_token_answer += 1
            else:
                token_start_index += 1
        
        start_answer_token_positions.append(token_start_index + start_token_answer)
        
        end_token_answer = len(answers[i]) -1
        while answers[i][end_token_answer] == 'O':
            if offset_mapping[i][token_end_index][0] == 0:
                end_token_answer -= 1
                token_end_index -= 1
            else:
                token_end_index -= 1
                
        end_answer_token_positions.append(token_end_index + 1)
    
    outputs = {'input_ids':input_ids, 'attention_mask': input_attention, 
               "start_positions": start_answer_token_positions, "end_positions": end_answer_token_positions}
    
    return outputs


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
                             task_type=TaskType.QUESTION_ANS
                             )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model


if __name__ == "__main__":

    logger.info("Prepare dataset")
    data_files = {"train": config.train_data, "val": config.val_data}

    dataset = load_dataset('csv', data_files=data_files)

    # preprocess dataset, encode function will be applied on the fly
    dataset.set_transform(encode)

    logger.info("Training Preparation")
    if config.base_model_checkpoint in ["indobenchmark/indobert-base-p1", "indobenchmark/indobert-base-p2"]:
        # we need to overide the model config, because indobert-base model when loaded using AutoModelForQuestionAnswering, 
        # instead of creating 2 heads in the output layer, they create 5 heads. 
        model_config = AutoConfig.from_pretrained(config.base_model_checkpoint,
                                                  num_labels=2,
                                                  finetuning_task = "question answering"
                                                  )
        model = AutoModelForQuestionAnswering.from_pretrained(config.base_model_checkpoint, config = model_config)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(config.base_model_checkpoint)

    if config.lora_enable:
        logger.info("Save base model, tokenizer and create peft model")
        model = save_base_model_and_wrap_with_peft(model, tokenizer, config)

    # set config argument for Trainer object
    args = TrainingArguments(
                            output_dir= config.trainer_args.output_dir, 
                            eval_strategy = config.trainer_args.eval_strategy,   
                            save_strategy = config.trainer_args.save_strategy,
                            logging_strategy = config.trainer_args.logging_strategy, 
                            remove_unused_columns= config.trainer_args.remove_unused_columns,  
                            learning_rate= config.trainer_args.learning_rate,
                            per_device_train_batch_size= config.trainer_args.per_device_train_batch_size,
                            per_device_eval_batch_size= config.trainer_args.per_device_eval_batch_size,
                            num_train_epochs= config.trainer_args.num_train_epochs,
                            weight_decay= config.trainer_args.weight_decay,
                            gradient_accumulation_steps= config.trainer_args.gradient_accumulation_steps,
                            average_tokens_across_devices= config.trainer_args.average_tokens_across_devices,
                            load_best_model_at_end = config.trainer_args.load_best_model_at_end,
                            metric_for_best_model = config.trainer_args.metric_for_best_model,
                            save_total_limit = config.trainer_args.save_total_limit,   
                            report_to = config.trainer_args.report_to
                            )

    # set data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #set callback
    es_callback = EarlyStoppingCallback(early_stopping_patience=3)
    # create Trainer object
    trainer = Trainer(
                    model,
                    args,
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["val"],
                    data_collator=data_collator,
                    callbacks=[es_callback]
                    )

    logger.info("Begin Training")
    trainer.train()
    logger.info("End Training")