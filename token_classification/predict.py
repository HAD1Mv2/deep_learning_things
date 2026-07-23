import argparse
import nltk
from transformers import pipeline
from utils import get_default_device, load_config, instantiate_logger

logger = instantiate_logger("Token Class Predict")

logger.info("load config")
config =  load_config("config.yaml", "predict")

def args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Input text for token classification")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    input_args = args_parser()

    logger.info("Load pipeline object")
    device = get_default_device()

    token_classifier_pipe = pipeline("token-classification", model=config.model_checkpoint, aggregation_strategy="max", device=device)
    
    logger.info("Input Text")
    if input_args.text is None:
        text_to_pred = input("Input sentence: ")
    else:
        text_to_pred = input_args.text

    word_token = nltk.word_tokenize(text_to_pred)
    result = token_classifier_pipe(word_token, is_split_into_words=True, stride = 64)
    
    print(result)
    logger.info("Done")

