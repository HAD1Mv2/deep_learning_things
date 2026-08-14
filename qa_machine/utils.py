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

# get device function
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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

def read_json(filename: str):

    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def save_to_jsonl(obj_to_save: list[Any], outfile: str, num_obj_to_save: int =100):

    with open(outfile, "w", encoding="utf-8") as f:
        for i, item in enumerate(obj_to_save):
            if i >= num_obj_to_save:
                break
            f.write(json.dumps(item, cls=NumpyEncoder) + "\n")