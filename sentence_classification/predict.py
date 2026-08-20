import logging
import sys
import yaml
import argparse
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from utils import get_default_device

# Set logger config
logger = logging.getLogger("Predict")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_bulk", action="store_true", help="Do bulk prediction, filename required")
    parser.add_argument("--filename", type=str, default=None, help="Filename to read for bulk prediction, must be a tsv file")
    parser.add_argument("--header", type=str, default=None, help="Column name in tsv file containing text data to be predicted")
    parser.add_argument("--outfile", type=str, help="Filename for saving result in bulk prediction")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    input_args = args_parser()

    # Load config
    logger.info("Load Config.")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)["predict"]

    model_checkpoint = config["model_checkpoint"]
    batch_size = config["batch_size"]

    filename = input_args.filename
    do_bulk_predict = input_args.do_bulk
    header = input_args.header
    outfile = input_args.outfile

    try:
        file_column_name = input_args.column_name
    except:
        file_column_name = None

    logger.info("Load HF pipeline.")
    # get device
    device = get_default_device()

    # load pipe
    pipe = pipeline("text-classification", model=model_checkpoint, device = device)

    if do_bulk_predict :

        logger.info("Start bulk prediction process")
        logger.info(f"read {filename}")
        if header is None:
            predict_data_df = pd.read_csv(filename, sep = "\t", on_bad_lines = 'warn', header = header)
            predict_data_df.rename(columns = {predict_data_df.columns[0]: "text"}, inplace = True)
        else:
            predict_data_df = pd.read_csv(filename, sep = "\t", on_bad_lines = 'warn', header = 0)
            predict_data_df.rename(columns = {header: "text"}, inplace = True)
        predict_data_df = predict_data_df[["text"]]
        predict_dataset = Dataset.from_pandas(predict_data_df)

        logger.info("Begin prediction loop")
        predict_result= []
        print("Streaming prediction progress")
        for out in tqdm(pipe(KeyDataset(predict_dataset, "text"), batch_size=batch_size, truncation="only_first"), total = len(predict_dataset)):
            predict_result.append(out)
        logger.info("Ending prediction loop")

        predict_result_df = pd.DataFrame.from_records(predict_result)
        result_to_saved_df = pd.concat([predict_data_df, predict_result_df ], axis=1) 
        logger.info(f"save predictio result to {outfile}")
        result_to_saved_df.to_csv(outfile, sep="\t", index=False)
    else:
        text_to_pred = input("Input sentence: ")
        result = pipe(text_to_pred, truncation="only_first")
        print(f"The sentence sentiment is {result[0]['label']} with probability score {result[0]['score']}")

    logger.info("Quit.")