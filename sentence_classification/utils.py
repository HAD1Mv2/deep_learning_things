import sys
import yaml
import logging
import pandas as pd
import torch
from logging import Logger
from types import SimpleNamespace


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

def read_files_for_text_classification(file_path: str, header_exist: bool) -> pd.DataFrame:
    """Read files used for train model in text classification task, must be a tsv file. The file assumed already cleaned. 

    Parameters
    ----------
    file_path : str
        File location.
    text_column_name : str
        Name of column contains input text/sentece data.
    label_column_name : str
        Name of column contains label data.

    Returns
    -------
    output : pd.DataFrame
        Pandas dataframe of the data.
    """

    data_df = pd.read_csv(file_path, sep = "\t", on_bad_lines = 'warn', header = 0 if header_exist else None, names = ['text', 'label'])

    return data_df

# set get device function
def get_default_device() -> torch.device:
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