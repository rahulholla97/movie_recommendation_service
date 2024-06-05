import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


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

k_folds = 5
kf = KFold(n_splits=k_folds)

mse_scores = []

for i, (train_index, test_index) in enumerate(tqdm(kf.split(tfidf_matrix), total=k_folds, desc="Cross-validation")):
    X_train, X_test = tfidf_matrix[train_index], tfidf_matrix[test_index]
    nn.fit(X_train)    
    predictions = []

    for idx in test_index:
        distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=11)
        movie_indices = indices.flatten()[1:]
        predictions.append(movie_indices)
    
    predictions_flat = [item for sublist in predictions for item in sublist]    
    mse = mean_squared_error(test_index, predictions_flat)
    mse_scores.append(mse)
    print(f"MSE for fold {i+1}: {mse}")

average_mse = sum(mse_scores) / len(mse_scores)
print("Average Mean Squared Error:", average_mse)
