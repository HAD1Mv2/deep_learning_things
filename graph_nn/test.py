import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from tqdm.auto import tqdm
from utils import get_device, instantiate_logger, load_config, load_model

# Instantiate logger
logger = instantiate_logger("Test")

# Load config file
logger.info("Load Config.")
config = load_config("config.yaml", "test")


def test(model, test_loader):

    device = get_device()
    print(f"Device using: '{device}'")
    model = model.to(device)

    model.eval()
    test_pbar = tqdm(test_loader, leave=True, total=len(test_loader))
    with torch.no_grad():
        total_test_loss = 0
        total_test_examples = 0
        for _, sampled_data in enumerate(test_pbar):
            sampled_data = sampled_data.to(device, non_blocking=True)
            output = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data['user', 'rates', 'anime'].edge_label_index, sampled_data.edge_attr_dict)
            loss = F.mse_loss(output, sampled_data['user', 'rates', 'anime'].edge_label)
            total_test_loss += loss.item() * output.numel()
            total_test_examples += output.numel()
            test_pbar.set_description("TEST || ")

        test_mse_loss = total_test_loss / total_test_examples

        print(f"TEST || MSE Loss: {test_mse_loss:.4f}")


if __name__ == "__main__":

    logger.info("Load test PyG data object")
    test_data = torch.load(config.test_g_path, weights_only=False)

    # create test loader
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[-1, -1],
        edge_label_index=(("user", "rates", "anime"), test_data["user", "rates", "anime"].edge_label_index),
        edge_label=test_data["user", "rates", "anime"].edge_label,
        batch_size=config.batch_size,
        pin_memory=True
    )   

    # load model
    logger.info("Load Model")
    model = load_model(config.gnn_encoder_path, config.predictor_path, config.use_edge_attr, config.hidden_channels)

    logger.info("Test model")
    test(model, test_loader)

    logger.info("End")
    
