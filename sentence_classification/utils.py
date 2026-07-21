import sys
import logging
import pandas as pd
import torch
from logging import Logger

def read_files_for_text_classification(file_path: str, text_column_name: str, label_column_name: str) -> pd.DataFrame:
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

    data_df = pd.read_csv(file_path, sep = "\t", on_bad_lines = 'warn', header = None, names = [text_column_name, label_column_name])
    data_df.rename(columns = {text_column_name: "text", label_column_name: "label"}, inplace = True)

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

