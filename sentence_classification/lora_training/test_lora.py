import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from peft import PeftModel, AutoPeftModelForSequenceClassification
from utils import read_files_for_text_classification, get_default_device, instantiate_logger, load_config
from sklearn.metrics import classification_report

# Set logger 
logger = instantiate_logger("Test")

if __name__ == "__main__":

    logger.info("Load Config")
    config = load_config("config_lora.yaml", "test")

    logger.info("Preparation")
    logger.info("load device")
    # get device
    device = get_default_device()

    logger.info("Load test data.")
    test_data_df = read_files_for_text_classification(file_path=config.dataset.test_file_path, 
                                                      header_exist=config.dataset.header_exist
                                                      )    
    
    test_dataset = Dataset.from_pandas(test_data_df[["text", "label"]])
    print(len(test_dataset))

    logger.info("Prepare model")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_checkpoint)

    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_checkpoint)

    model = PeftModel.from_pretrained(base_model, config.adapter_path)
    model.eval()

    logger.info("Instantiate pipeline object")

    pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer, dtype=torch.float16 if config.fp16 else "auto", device = device)

    logger.info("Begin prediction loop")
    pred_result= []
    print("Streaming prediction progress")
    for out in tqdm(pipe(KeyDataset(test_dataset, "text"), batch_size=config.batch_size, truncation="only_first"), total = len(test_dataset)):
        pred_result.append(out)
    logger.info("Ending prediction loop")
    
    logger.info("Print report.\n")
    pred_result_df = pd.DataFrame.from_records(pred_result)

    print(classification_report(test_data_df['label'], pred_result_df['label']))

    logger.info("Test end")