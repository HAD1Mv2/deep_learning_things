import logging
import sys
import yaml
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from train import read_files_for_text_classification, get_default_device
from sklearn.metrics import classification_report

# Set logger config
logger = logging.getLogger("Test")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

if __name__ == "__main__":

    logger.info("Load Config.")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)["test"]

    batch_size = config["batch_size"]
    encoder_max_length = config["encoder_max_length"]
    fp16 = config["fp16"]
    test_file_path = config["file_path"]
    test_file_columns = config["file_columns"]
    model_checkpoint = config["model_checkpoint"]

    logger.info("Preparation.")
    logger.info("load device.")
    # get device
    device = get_default_device()

    logger.info("Load test data.")
    test_data_df = read_files_for_text_classification(file_path=test_file_path, 
                                                      text_column_name=test_file_columns["text"], 
                                                      label_column_name=test_file_columns["label"]
                                                      )    
    test_dataset = Dataset.from_pandas(test_data_df[["text", "label"]])

    logger.info("Instantiate pipeline obj.")
    pipe = pipeline("text-classification", model=model_checkpoint, device = device)

    logger.info("Begin prediction loop")
    pred_result= []
    print("Streaming prediction progress")
    for out in tqdm(pipe(KeyDataset(test_dataset, "text"), batch_size=batch_size, truncation="only_first"), total = len(test_dataset)):
        pred_result.append(out)
    logger.info("Ending prediction loop")
    
    logger.info("Print report.\n")
    pred_result_df = pd.DataFrame.from_records(pred_result)

    print(classification_report(test_data_df['label'], pred_result_df['label']))

    logger.info("Test end.")