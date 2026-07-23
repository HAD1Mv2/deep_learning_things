import argparse
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from utils import get_default_device, load_config, instantiate_logger, save_to_jsonl

logger = instantiate_logger("Token Class Bulk Predict")

logger.info("load config")
config = load_config("config.yaml", "predict")

def args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, default=None, help="Input file for bulk token predictions, in txt file, one sentence per line")
    parser.add_argument("-o", "--outfile", type=str, help="Filename for saving result in bulk prediction, .jsonl format")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    input_args = args_parser()

    logger.info("Load pipeline object")
    device = get_default_device()

    token_classifier_pipe = pipeline("token-classification", model=config.model_checkpoint, aggregation_strategy="max", device=device)


    logger.info("Start bulk prediction process")
    logger.info(f"read {input_args.filename}")

    input_texts_dataset = load_dataset("text", data_files=input_args.filename)

    logger.info("Begin prediction loop")
    prediction_result= []
    print("Streaming prediction progress")
    for out in tqdm(token_classifier_pipe(KeyDataset(input_texts_dataset["train"], "text"), batch_size=config.batch_size, is_split_into_words=False, stride = 64), total = len(input_texts_dataset)):
        prediction_result.append(out)
    logger.info("Ending prediction loop")

    logger.info(f"save predictio result to {input_args.outfile}")
    save_to_jsonl(prediction_result, input_args.outfile)

    logger.info("Done")

