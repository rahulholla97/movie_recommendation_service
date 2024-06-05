import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

movies_df = pd.read_csv('rotten_tomatoes_movies.csv')
reviews_df = pd.read_csv('rotten_tomatoes_movie_reviews.csv')

movies_df.fillna('', inplace=True)
reviews_df.fillna('', inplace=True)

if 'releaseDateTheaters' in movies_df.columns:
    movies_df['releaseDateTheaters'] = pd.to_datetime(movies_df['releaseDateTheaters'], errors='coerce')

if 'releaseDateStreaming' in movies_df.columns:
    movies_df['releaseDateStreaming'] = pd.to_datetime(movies_df['releaseDateStreaming'], errors='coerce')

movies_df['combined_features'] = (
    movies_df['genre'] + ' ' + 
    movies_df['director'] + ' ' + 
    movies_df['writer'] + ' ' + 
    movies_df['ratingContents']
)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])

nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)


def get_recommendations(title, nn=nn, movies_df=movies_df, tfidf_matrix=tfidf_matrix):
    idx = movies_df[movies_df['title'].str.contains(title, case=False)].index[0]
    distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=11)
    movie_indices = indices.flatten()[1:]
    movie_distances = distances.flatten()[1:]

    recommendations = pd.DataFrame({
        'Index': movie_indices,
        'Title': movies_df['title'].iloc[movie_indices].values,
        'Genre': movies_df['genre'].iloc[movie_indices].values,
        'Director': movies_df['director'].iloc[movie_indices].values,
        'Writer': movies_df['writer'].iloc[movie_indices].values,
        'RatingContents': movies_df['ratingContents'].iloc[movie_indices].values,
        'Distance': movie_distances
    })
    
    return recommendations

st.title("Movie Recommendation System")
user_input = st.text_input("Enter a movie title:", "")

if user_input:
    recommendations = get_recommendations(user_input)
    st.dataframe(recommendations)
