## Movie Recommendation System

Movie recommender systems have emerged as a pivotal technology in recent years, revolutionizing the entertainment industry and enhancing user experiences. With an abundance of movies available across various platforms, it has become increasingly challenging for viewers to discover films that resonate with their preferences. Movie recommender systems employ sophisticated algorithms and data analysis techniques to offer tailored suggestions based on user behavior, preferences, ratings, and other relevant factors.

These recommendations not only help viewers save time and effort in searching for movies but also enhance their overall entertainment experience. By providing tailored suggestions, movie recommender systems expose users to a broader range of films, including hidden gems and lesser-known titles, which they might not have otherwise come across. This not only encourages diversity in movie consumption but also supports filmmakers and content creators by promoting a wider viewership.

### Objective:

The objective of this project is to develop a content-based movie recommender system. By analyzing various movie features such as audience scores, genres, directors, and critic reviews, the system aims to provide personalized movie recommendations to users.

### Dataset:

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews/data)

- Get the dataset by one of the following ways:
  - Run the following commands in your terminal and unzip the package:
    - ```pip install kaggle``` and then ```kaggle datasets download -d andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews```

  - If you are using **linux / mac** use the **rotten_tomatoes_movie.tar.xz** available in the repo and unzip it.

  - If you are using **windows / mac** use the **rotten_tomatoes_movie.7z** available in the repo and unzip it.
  
- Make sure to move the unzipped dataset files into the same directory as the source code.

The dataset comprises two main components:

**1. Rotten Tomatoes Movies Dataset (rotten_tomatoes_movies.csv):**
   - **id:** Unique identifier for each movie.
   - **title:** The title of the movie.
   - **audienceScore:** The average score given by regular viewers.
   - **tomatoMeter:** The percentage of positive reviews from professional critics.
   - **rating:** The movie's age-based classification (e.g., G, PG, PG-13, R).
   - **releaseDateTheatres:** The date the movie was released in theaters.
   - **releaseDateStreaming:** The date the movie became available for streaming.
   - **runtimeMinutes:** The duration of the movie in minutes.
   - **genre:** The movie's genre(s).
   - **originalLanguage:** The original language of the movie.
   - **director:** The movie's director.
   - **writer:** The writer(s) responsible for the movie's screenplay.
   - **boxOffice:** The movie's total box office revenue.
   - **distributor:** The company responsible for distributing the movie.
   - **soundMix:** The audio format(s) used in the movie.


**2. Rotten Tomatoes Movie Reviews Dataset (rotten_tomatoes_movie_reviews.csv):**
   - **id:** Unique identifier for each movie (matches the id in rotten_tomatoes_movies.csv).
   - **reviewId:** Unique identifier for each critic review.
   - **creationDate:** The date the review was published.
   - **criticName:** The name of the critic who wrote the review.
   - **isTopCritic:** A boolean value indicating if the critic is considered a top critic.
   - **originalScore:** The score provided by the critic.
   - **reviewState:** The status of the review (e.g., fresh, rotten).
   - **publicationName:** The name of the publication where the review was published.
   - **reviewText:** The full text of the critic review.
   - **scoreSentiment:** The sentiment of the critic's score (e.g., positive, negative, neutral).
   - **reviewUrl:** The URL of the original review on Rotten Tomatoes.

### Implementation:

Using machine learning algorithms and techniques, the recommendation system analyzes the provided dataset to generate personalized movie recommendations for users. By considering various movie attributes and user preferences, the system aims to enhance the overall movie-watching experience by suggesting relevant and engaging content.

