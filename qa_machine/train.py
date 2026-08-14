import copy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
import numpy as np
from utils import instantiate_logger, load_config


logger = instantiate_logger("Training")

logger.info("Load Train Config")
config = load_config("config.yaml", "train")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

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


if __name__ == "__main__":

    logger.info("Prepare dataset")
    data_files = {"train": config.train_data, "val": config.val_data}

    dataset = load_dataset('csv', data_files=data_files)

    # preprocess dataset, encode function will be applied on the fly
    dataset.set_transform(encode)

    logger.info("Training Preparation")
    model = AutoModelForQuestionAnswering.from_pretrained(config.model_checkpoint)

    # set config argument for Trainer object
    args = TrainingArguments(
                            output_dir= config.trainer_args.output_dir, 
                            eval_strategy = config.trainer_args.eval_strategy,   
                            save_strategy = config.trainer_args.save_strategy,
                            logging_strategy = config.trainer_args.logging_strategy, 
                            remove_unused_columns= config.trainer_args.remove_unused_columns,  
                            learning_rate= config.trainer_args.learning_rate,
                            per_device_train_batch_size= config.batch_size,
                            per_device_eval_batch_size= config.batch_size,
                            num_train_epochs= config.trainer_args.num_train_epochs,
                            weight_decay= config.trainer_args.weight_decay,
                            gradient_accumulation_steps= config.trainer_args.gradient_accumulation_steps,
                            average_tokens_across_devices= config.trainer_args.average_tokens_across_devices,
                            load_best_model_at_end = config.trainer_args.load_best_model_at_end,
                            metric_for_best_model = config.trainer_args.metric_for_best_model,
                            save_total_limit = config.trainer_args.save_total_limit,    
                            )

    # set data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #set callback
    eos_callback = EarlyStoppingCallback(early_stopping_patience=3)
    # create Trainer object
    trainer = Trainer(
                    model,
                    args,
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["val"],
                    data_collator=data_collator,
                    callbacks=[eos_callback]
                    )

    logger.info("Begin Training")
    trainer.train()
    logger.info("End Training")