# Since anime 2023 dataset is huge, and it will takes a long time to train on the whole data, 
# we need to sample the dataset enough to train the model within a reasonable time
# This script is used to sampling data from anime dataset

import pandas as pd
from utils import load_config

config = load_config("config.yaml", "sampling_data")

if __name__ == "__main__":

    user_details_path = "data/anime-dataset-2023/users-details-2023.csv"
    anime_data_path = "data/anime-dataset-2023/anime-dataset-2023.csv"
    user_score_path = "data/anime-dataset-2023/users-score-2023.csv"

    user_details_df = pd.read_csv(user_details_path)
    anime_df = pd.read_csv(anime_data_path)
    users_score_df = pd.read_csv(user_score_path)

    # remove users with zero days watch
    user_details_df = user_details_df[user_details_df["Days Watched"] > 0.0]

    # sampling user data
    user_details_df_sample = user_details_df.sample(n=config.num_user_sample, random_state=config.random_seed)

    # filter user score data, taking data where user_id exist in sampled user
    users_score_df = users_score_df[users_score_df["user_id"].isin(user_details_df_sample["Mal ID"])]

    # filter user data that don't exist in user score data (remove isolated vertices)
    user_details_df_sample = user_details_df_sample[user_details_df_sample["Mal ID"].isin(users_score_df["user_id"])]

    # filter anime data so it matched with user score data
    anime_df = anime_df[anime_df["anime_id"].isin(users_score_df["anime_id"])]

    # drop anime data columns that not used  
    anime_df.drop(labels=['English name', 'Other name', 'Synopsis', 'Aired', 
                        'Premiered', 'Status', 'Producers', 'Licensors', 
                        'Source', 'Duration', 'Rating', 'Rank', 'Popularity', 
                        'Favorites', 'Scored By', 'Members', 'Image URL', 
                        'Studios'], 
                axis=1, 
                inplace=True)
    # replace unknown values with big negative value
    anime_df["Score"] = anime_df["Score"].replace("UNKNOWN", -10).astype(float)
    anime_df["Episodes"] = anime_df["Episodes"].replace("UNKNOWN", -100).astype(float)

    # filter user score df one more time, since there are some cases where anime_id in user score not exist in anime df data
    users_score_df= users_score_df[users_score_df["anime_id"].isin(anime_df["anime_id"])]

    # save anime sample data
    anime_sample_path = "data/processed/anime-dataset-2023-sample.csv"
    anime_df.to_csv(anime_sample_path, index=False)

    # save user details data
    sample_user_details_path = "data/processed/users-details-2023-sample.csv"
    user_details_df_sample.to_csv(sample_user_details_path, index=False) 

    # save ratings data
    users_score_df_sample_path = "data/processed/users-score-2023-sample.csv"
    users_score_df.to_csv(users_score_df_sample_path, index=False)