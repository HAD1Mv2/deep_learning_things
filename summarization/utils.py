import sys
import yaml
import logging
import torch
import itertools
import json
import numpy as np
from typing import Any
from types import SimpleNamespace
from logging import Logger
from transformers.tokenization_utils_base import BatchEncoding


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

    return optimizer.param_groups[0]['lr']

# set get device function
def get_default_device():
    """Pick device for training.

    Returns
    -------
    torch.device
        Device used for training, it will pick GPU or CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

    
def to_namespace(data):
    if isinstance(data, dict):
        # Recursively convert values, then wrap the dict in a SimpleNamespace 
        return SimpleNamespace(**{k: to_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        # Recursively convert items inside lists
        return [to_namespace(item) for item in data]
    else:
        # Return primitive types (strings, ints, etc.) as-is
        return data
    
def load_config(config_path: str, split: str) -> SimpleNamespace:
    """Load config. 

    Parameters
    ----------
    config_path : str
        Config file location.
    split : str
        Choose one of ["train", "test", "predict"].
    Returns
    -------
    SimpleNamespace
        Dictionary like object for easy access config's key value.
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)[split]

    config =  to_namespace(config)

    return config


def instantiate_logger(logger_name: str) -> Logger:
    """Instantiate Logger object.

    Parameters
    ----------
    logger_name : str
        Name of logger.

    Returns
    -------
    Logger
        Logger object.
    """

    # Set logger config
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


# encode function to preprocess the data
class SummarizeEncoder:
    def __init__(self, tokenizer, encoder_max_len: int = 512):
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
    
    def encode(self, example) -> BatchEncoding:
        """Method for encoding text dataset

        Parameters
        ----------
        example : 
            Element of dataset obj.

        Returns
        -------
        _type_
            _description_
        """

        # flatten the paragraph and summary data from dataset
        paragraph_wordpiece = ['summarize', ':' ] + list(itertools.chain.from_iterable(itertools.chain.from_iterable(example['paragraphs'])))
        summary_wordpiece = list(itertools.chain.from_iterable(example['summary']))

        # we need to put 'summarize: ' at the beginning of every paragraph, since that what the documentation tell to, you can change to another signature though
        encodings= self.tokenizer(text= paragraph_wordpiece, # the paragraph in dataset is in the form of list of sentence, and the sentence is in the form of list of words
                            text_target= summary_wordpiece, 
                            is_split_into_words=True, 
                            truncation=True, 
                            max_length= self.encoder_max_len,
                            padding=False)

        return encodings


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        # Convert numpy floats to python floats
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        # Convert numpy ints to python ints
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        # Convert numpy arrays to python lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_to_jsonl(obj_to_save: list[Any], outfile: str, num_obj_to_save: int =100):

    with open(outfile, "w", encoding="utf-8") as f:
        for i, item in enumerate(obj_to_save):
            if i >= num_obj_to_save:
                break
            f.write(json.dumps(item, cls=NumpyEncoder) + "\n")