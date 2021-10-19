import torch
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

def get_opt(name):
    if name.lower() == 'adam':
        return torch.optim.Adam
    elif name.lower() == 'sgd':
        return torch.optim.SGD
    elif name.lower() == 'sparseadam':
        return torch.optim.SparseAdam
    else:
        raise NotImplementedError('Optimazer can be only adam, sgd or sparseadam')

def get_edges(df):
    df = df[df['rating'] > 3.5]

    user_id_embeddings = {id:i for i,id in enumerate(df['userID'].unique())}
    movie_id_embeddings = {id:i for i,id in enumerate(df['movieID'].unique())}

    src = [user_id_embeddings[x] for x in df['userID']]
    dst = [movie_id_embeddings[x] for x in df['movieID']]
    edge_index = torch.tensor([src,dst])
    edge_attr = torch.from_numpy(df['rating'].values).to(torch.float)
    
    #RandomLinkSplit acts different than expected when manual assigning edge labels
    #edge_label = torch.from_numpy(df['rating'].apply(lambda x : 1 if x >= 3.5 else 0).values).to(torch.long)
    
    return user_id_embeddings, movie_id_embeddings, edge_index, edge_attr, []

def load_prepare_data(movies_path,connection_path,negative_sample_ratio):
    movies_df = pd.read_pickle(movies_path)
    user_movies_egdes = pd.read_pickle(connection_path)
    user_movies_egdes = user_movies_egdes[user_movies_egdes['movieID'].isin(movies_df['movieID'].values)]
    movies_feature = movies_df[movies_df['movieID'].isin(user_movies_egdes['movieID'].values)]
    movies_feature.drop(columns = ['movieID'], inplace = True)
    movies_feature = torch.from_numpy(movies_df.values).float()
    user_id_embeddings, movie_id_embeddings, edge_index, edge_attr, edge_label = get_edges(user_movies_egdes)

    data = HeteroData()
    data['user'].num_nodes = len(user_id_embeddings) 
    data['user'].x = torch.ones(data['user'].num_nodes,1)
    data['movie'].x = movies_feature 
    data['user','rated', 'movie'].edge_index = edge_index

    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)
    data = T.NormalizeFeatures()(data)

    del data['user','rev_rated','movie'].edge_label

    transformer = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        add_negative_train_samples = True,
        neg_sampling_ratio=negative_sample_ratio,
        edge_types=[('user','rated','movie')],
        rev_edge_types=[('user','rev_rated','movie')],
        is_undirected = True,
    )
    train_data, val_data, test_data = transformer(data)

    return user_id_embeddings,movie_id_embeddings,train_data, val_data, test_data

