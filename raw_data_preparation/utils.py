import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer,util
import csv

NUM_REVIEWS = 20

def encode_variable(values):
    encoding_dict = {}
    encoded_values = []
    i = 0
    for x in values:
        if not x in encoding_dict:
            encoding_dict[x] = i
            i+=1
        encoded_values.append(encoding_dict[x])
    
    return encoding_dict, pd.Series(encoded_values)

def weighted_rating(x, m=30, C=None):
    if C == None:
        raise ValueError('Argument C (Mean rating of all movies) cannot be none')

    v,R = x
    return (v/(v+m) * R) + (m/(m+v) * C)


def calculate_imdb_mean_rating(row):
    ratings = []
    if row['rtAllCriticsRating'] != 0:
        ratings.append(row['rtAllCriticsRating'])
    if row['rtTopCriticsRating'] != 0:
        ratings.append(row['rtTopCriticsRating'])
    if row['rtAudienceRating'] != 0:
        ratings.append(row['rtAudienceRating'])
    
    return np.mean(ratings)

def calculate_user_mean_rating(x):
    return pd.Series((x['movieID'].iloc[0],(round(np.mean(x['rating'])/5,2)), len(x['rating'])),index=['movieID','mean_rating','platform_count'])

def fill_nan_location(x,countries):
    if pd.isna(x['location1']):
        if x['Animation'] != 1:
            x['location1'] = countries[countries['movieID'] == x['movieID']]['country'].values[0]
        else:
            x['location1'] = 'other'
    return x

def process_title(movies, model):
    movie_embeddings = model.encode(movies['title'].tolist(), batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    clusters = util.community_detection(movie_embeddings, min_community_size=5, threshold=0.60)
    cluster_map = dict()
    [cluster_map.update({movies.iloc[x]['title']:i}) for i,cluster in enumerate(clusters) for x in cluster]
    last_cluster = len(clusters) + 1

    return movies['title'].apply(lambda x : cluster_map[x] if x in cluster_map else last_cluster)

def load_process_movies(movies_path,user_movies_path):
    movies = pd.read_csv(movies_path, sep='\t',quoting= csv.QUOTE_NONE, engine = 'python')
    sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    movies = movies.replace('\\N',0.0)

    movies['rtAllCriticsRating'] = movies['rtAllCriticsRating'].astype(np.float32)
    movies['rtTopCriticsRating'] = movies['rtTopCriticsRating'].astype(np.float32)
    movies['rtAudienceRating'] = movies['rtAudienceRating'].astype(np.float32)
    movies['rtAllCriticsNumReviews'] = movies['rtAllCriticsNumReviews'].astype(np.int)
    movies['rtAudienceNumRatings'] = movies['rtAudienceNumRatings'].astype(np.int)
    movies['rtTopCriticsNumReviews'] = movies['rtTopCriticsNumReviews'].astype(np.int)

    movies['rtAllCriticsRating'] = movies['rtAllCriticsRating'].apply(lambda x: round(x/10,2))
    movies['rtTopCriticsRating']  = movies['rtTopCriticsRating'].apply(lambda x: round(x/10,2))
    movies['rtAudienceRating']  = movies['rtAudienceRating'].apply(lambda x : round(x/5,2))

    movies['imdb_count'] = movies[['rtTopCriticsNumReviews','rtAudienceNumRatings','rtAllCriticsNumReviews']].sum(axis=1)
    movies = movies.loc[movies['imdb_count']>NUM_REVIEWS]
    movies['rating'] = movies.apply(lambda row : calculate_imdb_mean_rating(row), axis=1)
    movies['weighted_imdb_rating'] = weighted_rating((movies['imdb_count'],movies['rating']),m = NUM_REVIEWS, C= np.mean(movies['rating']))
    movies.drop(columns=['rtAllCriticsRating','rating','rtAllCriticsNumReviews','rtAllCriticsNumFresh','rtAllCriticsNumRotten','rtTopCriticsRating','rtTopCriticsNumReviews','rtTopCriticsNumFresh','rtTopCriticsNumRotten','rtAudienceRating','rtAudienceNumRatings','imdbID','spanishTitle', 'imdbPictureURL', 'rtID', 'rtAllCriticsScore','rtTopCriticsScore', 'rtAudienceScore', 'rtPictureURL'],inplace = True)

    user_movies = pd.read_csv(user_movies_path, sep='\t',quoting= csv.QUOTE_NONE, engine = 'python')
    df = user_movies.groupby('movieID').apply(calculate_user_mean_rating).reset_index(drop = True)
    df = df.loc[df['platform_count']>10]
    movies.rename(columns = {'id': 'movieID'}, inplace = True)
    merged_df = pd.merge(df,movies,on = 'movieID')
    merged_df['title_cluster'] = process_title(merged_df, sentence_transformer)
    merged_df.drop(columns= ['title','platform_count','mean_rating'], inplace = True)

    return merged_df

def process_locations(locations_path,countries_path,df):
    locations = pd.read_csv(locations_path,sep='\t',quoting= csv.QUOTE_NONE, engine = 'python')
    countries = pd.read_csv(countries_path,sep='\t',quoting= csv.QUOTE_NONE, engine = 'python')
    locations.drop(['location2','location3','location4'],axis = 1, inplace=True)
    locations = pd.merge(df,locations,on= 'movieID')
    locations = locations.apply(lambda x: fill_nan_location(x,countries),axis = 1)

    merged_locations = locations.groupby('movieID')['location1'].apply(pd.Series.mode).reset_index()
    locations_dict, merged_locations['location1'] = encode_variable(locations['location1'].values)
    merged_locations.rename(columns = {'location1':'frequent_location'}, inplace=True)
    merged_locations.drop(columns='level_1', inplace=True)
    merged_locations = pd.merge(df, merged_locations, on = 'movieID')

    return locations_dict, merged_locations

def process_genres(genres_path, df):
    genres = pd.read_csv(genres_path,sep='\t', quoting= csv.QUOTE_NONE,engine = 'python')
    genres = genres.groupby('movieID')['genre'].apply(list).reset_index()
    genres = genres.drop('genre', 1).join(pd.get_dummies(pd.DataFrame(genres.genre.tolist()).stack()).astype(int).sum(level=0))
    
    return pd.merge(df, genres, on = 'movieID')

def process_directors(directors_path, df):
    directors = pd.read_csv(directors_path,sep='\t',quoting= csv.QUOTE_NONE, engine = 'python')
    most_frequent = directors['directorID'].value_counts()[:10].index.tolist()
    directors['directorID'] = directors['directorID'].apply(lambda x : x if x in most_frequent else 'other')
    directors.drop(['directorName'],axis = 1, inplace=True)
    directors_dict, directors['directorID'] = encode_variable(directors['directorID'].values)

    return directors_dict, pd.merge(df, directors, on = 'movieID')

def process_actors(actors_path, df):
    actors = pd.read_csv(actors_path, sep='\t',quoting= csv.QUOTE_NONE, engine = 'python')
    actors.drop(['actorName','actorID'], axis=1, inplace=True)
    actors = actors.groupby('movieID')['ranking'].apply(np.mean).reset_index()
    actors.rename(columns = {'ranking':'mean_actors_ranking'},inplace = True)

    return pd.merge(df, actors, on = 'movieID')


def process_user_movie_connection(user_movies_path):
    user_movies = pd.read_csv(user_movies_path, sep='\t',quoting= csv.QUOTE_NONE, engine = 'python')

    return user_movies[['userID','movieID','rating']]