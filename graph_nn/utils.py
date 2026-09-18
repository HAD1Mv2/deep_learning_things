import logging
import sys
from logging import Logger
from types import SimpleNamespace

import torch
import yaml
from module_architecture import (
    GraphPredictorModel,
    HeteroGNNEncoder,
    HeteroGNNEncoderEdge,
    MLPPredictor,
)


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
def get_device() -> torch.device:
    """Get machine device, priority cuda >> mps >> cpu.

    Returns
    -------
    torch.device
        Device for run the model.
    """
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    return torch.device(device_name)


# function to get the learning rate value
def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate value of the optimizer

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer used for backprop.

    Returns
    -------
    out : float
        Learning rate value.
    """

    return optimizer.param_groups[0]['lr']

def load_model(gnn_encoder_path: str, predictor_path: str, use_edge_attr: bool, hidden_channels: int) -> GraphPredictorModel:
    """Load GNN Predictor model

    Parameters
    ----------
    gnn_encoder_path : str
        Path to gnn encoder. 
    predictor_path : str
        Path to predictor component.
    use_edge_attr : bool
        Whether load model that incorporate edge features.
    hidden_channels : int
        Dimension of hidden layers. 

    Returns
    -------
    GraphPredictorModel
        Link regression model.
    """
    if use_edge_attr:
        gnn_encoder = HeteroGNNEncoderEdge(hidden_channels)
    else:
        gnn_encoder = HeteroGNNEncoder(hidden_channels)
    
    predictor = MLPPredictor(hidden_channels)

    gnn_encoder_state_dict = torch.load(gnn_encoder_path, weights_only=True)
    gnn_encoder.load_state_dict(gnn_encoder_state_dict)

    predictor_state_dict = torch.load(predictor_path, weights_only=True)
    predictor.load_state_dict(predictor_state_dict)

    model = GraphPredictorModel(gnn_encoder, predictor)

    return model 
