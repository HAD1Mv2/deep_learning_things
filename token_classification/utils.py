import re
import sys
import yaml
import torch
import logging
import json
import numpy as np
from pathlib import Path
from typing import Any
from logging import Logger

from types import SimpleNamespace

def read_file_txt(filename: str) -> list[str]:
    """Read text file in lines.

    Parameters
    ----------
    filename : str
        File to read.

    Returns
    -------
    list[str]
        List of text.
    """
    with open(filename) as f:
        result = f.readlines()

    result = list(map(str.strip, result))
    return result

    
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


# function to read data
def read_wnut(file_path:str) -> tuple[list[list[str]], list[list[str]]]:
    """Read Training data.

    Parameters
    ----------
    file_path : str
        File path.

    Returns
    -------
    tuple[list[str], list[str]]
        Tuple of sentence tokens and tags list.
    """

    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


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
    

def save_to_jsonl(obj_to_save: list[Any], outfile: str):

    with open(outfile, "w", encoding="utf-8") as f:
        for item in obj_to_save:
            f.write(json.dumps(item, cls=NumpyEncoder) + "\n")


def calculate_class_weights(train_labels: list[list[int]]) -> torch.tensor:
    """Calculate class weight based on class frequncy in train data.

    Parameters
    ----------
    train_labels : list[list[int]]
        List of list of token labels, label in numerical integer from 0 to num_classes-1.

    Returns
    -------
    torch.tensor
        Torch tensor containing class weight.
    """

    unrolled_train_labels = [element for label in train_labels for element in label if element != -100]
    unrolled_train_labels = torch.tensor(unrolled_train_labels)
    class_count = torch.bincount(unrolled_train_labels)
    class_weights = 1.0 / class_count
    class_weights = class_weights / class_weights.sum()
    return class_weights 