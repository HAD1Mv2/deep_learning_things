import torch
import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils import load_config, instantiate_logger, get_default_device

logger = instantiate_logger("Summarize")

logger.info("Load Summarize Config")
config = load_config("config.yaml", "summarize")

def args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, default=None, help="a .txt file containing the articel to be summarize.")
    parser.add_argument("-o", "--outfile", type=str, default=None, help="file to save the summary, a .txt file.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    logger.info("Preparing ...")
    # load text file
    input_args = args_parser()
    with open(input_args.filename,"r", encoding="utf-8") as file:
        text = file.read()

    text = "summarize: " + text
    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
    device = get_default_device()
    model = T5ForConditionalGeneration.from_pretrained(config.model_checkpoint).to(device)

    encoding = tokenizer(text= text,  
                         truncation=True, 
                         max_length= config.encoder_max_len,
                         padding=False,
                         return_tensors= 'pt')

    logger.info("Summarize ...")
    with torch.no_grad():
        input_ids, attention_mask  = encoding["input_ids"], encoding["attention_mask"]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        summary_tokens = model.generate(input_ids=input_ids, 
                                 attention_mask=attention_mask, 
                                 max_length=170, 
                                 min_length=40, 
                                 length_penalty=2.0, 
                                 num_beams=4, 
                                 early_stopping=True)

        summary = tokenizer.decode(summary_tokens, skip_special_tokens=True)[0]
        
    logger.info("Here is the summary:")
    print(summary)

    logger.info("Saving summary to file.")
    with open(input_args.outfile, "w", encoding="utf-8") as file:
        file.write(summary)

    logger.info("End.")