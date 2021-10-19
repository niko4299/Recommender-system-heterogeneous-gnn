import pickle
from utils import load_process_movies, process_actors, process_genres, process_locations, process_directors, process_user_movie_connection

MOVIES_PATH = 'raw_data_preparation/data/movies.dat'
ACTORS_PATH = 'raw_data_preparation/data/movie_actors.dat'
COUNTRIES_PATH = 'raw_data_preparation/data/movie_countries.dat'
GENRES_PATH = 'raw_data_preparation/data/movie_genres.dat'
LOCATIONS_PATH = 'raw_data_preparation/data/movie_locations.dat'
TAGS_PATH = 'raw_data_preparation/data/movie_tags.dat'
DIRECTORS_PATH = 'raw_data_preparation/data/movie_directors.dat'
USER_MOVIES_PATH  = 'raw_data_preparation/data/user_ratedmovies.dat'


if __name__ == '__main__':
    movies = load_process_movies(MOVIES_PATH,USER_MOVIES_PATH)
    user_movie_connection = process_user_movie_connection(USER_MOVIES_PATH)
    merged_df = process_genres(GENRES_PATH,movies)
    locations_map, merged_df = process_locations(LOCATIONS_PATH,COUNTRIES_PATH,merged_df)
    directors_map, merged_df = process_directors(DIRECTORS_PATH,merged_df)
    merged_df = process_actors(ACTORS_PATH,merged_df)
    merged_df.drop_duplicates(subset=['movieID'],inplace=True)
    merged_df.to_pickle("dataset/final_movies_dataset.pkl")
    
    user_movie_connection = user_movie_connection[user_movie_connection['movieID'].isin(movies['movieID'].values)]
    movies = movies[movies['movieID'].isin(user_movie_connection['movieID'].values)]

    user_movie_connection.to_pickle('dataset/user_movie_connection.pkl')

    with open('dataset/locations_encoding_map.pickle', 'wb') as handle:
        pickle.dump(locations_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('dataset/directors_encoding_map.pickle', 'wb') as handle:
        pickle.dump(directors_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    