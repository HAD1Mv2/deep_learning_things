import torch
import torch.nn.functional as F
from module_architecture import (
    GraphPredictorModel,
    HeteroGNNEncoder,
    HeteroGNNEncoderEdge,
    MLPPredictor,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import LinkNeighborLoader
from tqdm.auto import tqdm
from utils import get_device, get_lr, instantiate_logger, load_config

# Instantiate logger
logger = instantiate_logger("Training")

# Load config file
logger.info("Load Config.")
config = load_config("config.yaml", "train")


def fit(model, epochs, learning_rate, weight_decay, train_loader, val_loader):

    device = get_device()
    print(f"Device using: '{device}'")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience=2, threshold=1.0e-2, threshold_mode='rel')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_examples = 0
        train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
        for _, sampled_data in enumerate(train_pbar):
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device, non_blocking=True)
            output = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data['user', 'rates', 'anime'].edge_label_index, sampled_data.edge_attr_dict)
            loss = F.mse_loss(output, sampled_data['user', 'rates', 'anime'].edge_label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * output.numel()
            total_train_examples += output.numel()
            train_pbar.set_description(f"TRAIN || (Epoch {epoch+1}) Lr :{get_lr(optimizer):.4f}")

        print(f"TRAIN || (Epoch {epoch+1}), MSE Loss: {total_train_loss / total_train_examples:.4f}, lr: {get_lr(optimizer)}")

        model.eval()
        val_pbar = tqdm(val_loader, leave=True, total=len(val_loader))
        with torch.no_grad():
            total_val_loss = 0
            total_val_examples = 0
            for _, sampled_data in enumerate(val_pbar):
                sampled_data = sampled_data.to(device, non_blocking=True)
                output = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data['user', 'rates', 'anime'].edge_label_index, sampled_data.edge_attr_dict)
                loss = F.mse_loss(output, sampled_data['user', 'rates', 'anime'].edge_label)
                total_val_loss += loss.item() * output.numel()
                total_val_examples += output.numel()
                val_pbar.set_description(f"VALIDATION || (Epoch {epoch+1})")

            val_mse_loss = total_val_loss / total_val_examples
            scheduler.step(val_mse_loss)
            print(f"VALIDATION || (Epoch {epoch+1}), MSE Loss: {val_mse_loss:.4f}")

if __name__ == "__main__":

    # load graph data
    logger.info("Load train and test PyG data object")

    train_data = torch.load("data/hetero_graph/train_g.pt", weights_only=False)
    val_data = torch.load("data/hetero_graph/val_g.pt", weights_only=False)

    # create train loader
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=config.num_neighbors,
        edge_label_index=(("user", "rates", "anime"), train_data["user", "rates", "anime"].edge_label_index),
        edge_label=train_data["user", "rates", "anime"].edge_label,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True
    )

    # create val loader
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=config.num_neighbors,
        edge_label_index=(("user", "rates", "anime"), val_data["user", "rates", "anime"].edge_label_index),
        edge_label=val_data["user", "rates", "anime"].edge_label,
        batch_size=config.batch_size,
        pin_memory=True
    )    

    # create model
    logger.info("Instatiate Model")
    if config.use_edge_attr:
        gnn_encoder = HeteroGNNEncoderEdge(config.hidden_channels)
    else:
        gnn_encoder = HeteroGNNEncoder(config.hidden_channels)
    
    predictor = MLPPredictor(config.hidden_channels)
    model = GraphPredictorModel(gnn_encoder, predictor)

    # train model
    logger.info("Training")
    fit(model, config.epochs, config.optim.learning_rate, config.optim.weight_decay, train_loader, val_loader)

    # save model
    logger.info("Training Finish, save models")
    torch.save(gnn_encoder.state_dict(), config.gnn_encoder_save_path)
    torch.save(predictor.state_dict(), config.predictor_save_path)

    logger.info("End")