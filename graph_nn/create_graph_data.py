import os
from pathlib import Path

import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

if __name__ == "__main__":

    anime_node_dataset_path = "data/processed/anime-dataset-2023-sample.csv"
    user_node_dataset_path = "data/processed/users-details-2023-sample.csv" 
    user_anime_edge_data_path = "data/processed/users-score-2023-sample.csv"

    anime_node_df = pd.read_csv(anime_node_dataset_path)
    user_node_df = pd.read_csv(user_node_dataset_path)
    user_anime_edge_df = pd.read_csv(user_anime_edge_data_path)

    # Create node index for anime
    anime_node_df.reset_index(names="anime_node_id", inplace=True) 

    # Create node index for user
    user_node_df.reset_index(names="user_node_id", inplace=True)

    # Preprocess data, modify this section to preprocess data based on your need
    # ===========================================================================
    # Preprocess anime_df to prepare anime features 
    genre_one_hot = anime_node_df.Genres.str.get_dummies(sep=", ").add_prefix("genre_") 
    type_one_hot = anime_node_df.Type.str.get_dummies(sep=", ").add_prefix("type_") 

    genre_columns_to_drop = ['genre_Award Winning', 'genre_UNKNOWN']
    genre_one_hot.drop(columns = genre_columns_to_drop , inplace = True) 

    anime_node_df = pd.concat([anime_node_df.drop(columns=["Genres"]), genre_one_hot], axis=1)
    anime_node_df = pd.concat([anime_node_df.drop(columns=["Type"]), type_one_hot], axis=1)

    # scale down Score and Episode value
    anime_node_df['Score'] = anime_node_df['Score']/10.0
    anime_node_df['Episodes'] = anime_node_df['Episodes']/100.0

    # Preprocess user node data to get the feature
    user_gender_one_hot = user_node_df.Gender.str.get_dummies(sep=", ").add_prefix("gender_") 
    user_node_df = pd.concat([user_node_df.drop(columns=["Gender"]), user_gender_one_hot], axis=1)

    # preprocess user_anime_edge_df
    # scale down rating score from range (1., 10.) to range (0.1, 1.)
    user_anime_edge_df['rating'] = user_anime_edge_df['rating']/10.0 

    # ============================================================================

    # Create edge_index in COO format
    # pairing source and target nodes index (creating edge index)
    user_anime_edge_df = pd.merge(user_anime_edge_df, user_node_df[["Mal ID", "user_node_id"]], how="left", left_on="user_id", right_on="Mal ID")
    user_anime_edge_df = pd.merge(user_anime_edge_df, anime_node_df[["anime_id", "anime_node_id"]], how="left", on="anime_id")
    edge_source_user_id = torch.from_numpy(user_anime_edge_df.user_node_id.values)
    edge_end_anime_id = torch.from_numpy(user_anime_edge_df.anime_node_id.values)
    edge_index_user_to_anime = torch.stack([edge_source_user_id, edge_end_anime_id], dim=0)

    # Create feature vectors, modify this code to suit your need regarding the feature you want to use
    # =============================================================================
    # Node anime feature vectors
    # For the sake of simplicity, I only use num of episode, genre and type anime as feature vector
    
    anime_features_columns = [
                            'Episodes', 'genre_Action', 'genre_Adventure', 'genre_Avant Garde', 'genre_Boys Love', 'genre_Comedy', 
                            'genre_Drama', 'genre_Ecchi', 'genre_Erotica', 'genre_Fantasy', 'genre_Girls Love', 'genre_Gourmet','genre_Hentai', 
                            'genre_Horror', 'genre_Mystery', 'genre_Romance','genre_Sci-Fi', 'genre_Slice of Life', 'genre_Sports',
                            'genre_Supernatural', 'genre_Suspense', 'type_Movie','type_Music', 'type_ONA', 'type_OVA', 
                            'type_Special', 'type_TV'
                            ]

    anime_features = torch.from_numpy(anime_node_df[anime_features_columns].values).to(torch.float)  

    # Node user feature vectors
    # for the sake of simplicity, user feature will use gender as feature
    user_features= torch.from_numpy(user_node_df[['gender_Female', 'gender_Male','gender_Non-Binary']].values).to(torch.float)

    # Create edge label vectors
    ratings = torch.Tensor(user_anime_edge_df['rating'].to_numpy()/10.)
    # =============================================================================

    # Create HeteroData obj.

    data = HeteroData()

    # Save node indices:
    data["user"].node_id = torch.arange(len(user_node_df))
    data["anime"].node_id = torch.arange(len(anime_node_df))

    # Add the node features and edge indices:
    data["user"].x = user_features
    data["anime"].x = anime_features
    data["user", "rates", "anime"].edge_index = edge_index_user_to_anime 
    data['user', 'rates', 'anime'].edge_label = ratings  # [num_ratings]
    data['user', 'rates', 'anime'].edge_attr = ratings.reshape(-1, 1)  # [num_ratings, 1]

    # We also need to make sure to add the reverse edges from movies to users
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    data = T.ToUndirected()(data)

    # With the above transformation we also got reversed labels for the edges.
    # We are going to remove them:
    del data['anime', 'rev_rates', 'user'].edge_label

    # check whether the heterodata is valid
    data.validate(raise_on_error=True)

    # Create train, val, test split

    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.05,
        disjoint_train_ratio=0.2,
        neg_sampling_ratio=0.0,
        edge_types=[('user', 'rates', 'anime')],
        rev_edge_types=[('anime', 'rev_rates', 'user')],
    )(data)    

    # save train, val and test graph data
    dir_path = Path("data/hetero_graph")
    if not dir_path.is_dir():
        os.mkdir(dir_path)

    torch.save(train_data, "data/hetero_graph/train_g.pt")
    torch.save(val_data, "data/hetero_graph/val_g.pt")
    torch.save(test_data, "data/hetero_graph/test_g.pt")