import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import altair as alt

# Load data data from GitHub
def load_data_from_repo():
    movies_url = "data/movies.csv"
    ratings_url = "data/ratings.csv"

    # Load the CSV files from the 'data/' folder
    ratings = pd.read_csv(ratings_url, usecols=["userId", "movieId", "rating"])
    movies = pd.read_csv(movies_url, usecols=["movieId", "title", "genres"])

    # Handle genres: split by '|', expand genres and make a new row for each genre
    movies['genres'] = movies['genres'].str.split('|')
    genres_expanded = movies.explode('genres').reset_index(drop=True)
  
    # Merge the ratings with movies data
    data = pd.merge(ratings, genres_expanded, on="movieId")
    
    return data, movies, genres_expanded  # Return genres_expanded along with other data


# Load the data
data, movies, genres_expanded = load_data_from_repo()  # Expand the genres as well

# Create the user-item matrix (ratings for each user on each movie)
user_item_matrix = data.pivot_table(index="userId", columns="title", values="rating")
user_item_matrix.fillna(0, inplace=True)

# Calculate cosine similarity on the original user-item matrix
movie_similarity = cosine_similarity(user_item_matrix.T) # Transpose to compare movies
movie_titles = user_item_matrix.columns
movie_indices = {title: idx for idx, title in enumerate(movie_titles)}

# Normalize movie titles (case insensitive, ignore year in parentheses)
def normalize_title(title):
    title = re.sub(r"\(\d{4}\)", "", title)
    return title.strip().lower()

normalized_titles = {normalize_title(title): title for title in movie_titles}

# Function to find similar movies using cosine similarity with filters
def find_filtered_similar_movies(movie_title, top_n=10, min_rating=3.5, min_num_ratings=5, genres=None):
    normalized_input = normalize_title(movie_title)
  
    if normalized_input not in normalized_titles:
        return ["Movie not found."]
    
    matched_title = normalized_titles[normalized_input]
    movie_index = movie_indices[matched_title]
    similar_movies = movie_similarity[movie_index]
    similar_movie_indices = np.argsort(similar_movies)[::-1][1:] # Get indices sorted by similarity, excluding self

    # Filter by average rating, genres, and number of ratings
    filtered_movies = []
    for idx in similar_movie_indices:
        movie_name = movie_titles[idx]
        avg_rating = data[data['title'] == movie_name]['rating'].mean()
        num_ratings = len(data[data['title'] == movie_name])  # Count the number of ratings
        movie_genres = set(movies[movies['title'] == movie_name]['genres'].iloc[0])
        
        if avg_rating >= min_rating and num_ratings >= min_num_ratings and (genres is None or movie_genres.intersection(genres)):
            filtered_movies.append(movie_name)
        if len(filtered_movies) >= top_n:
            break
    
    return filtered_movies

# Function to display star rating
def display_star_rating(rating):
    rounded_up_rating = np.ceil(rating * 2) / 2 # Always round up to the next 0.5
    full_stars = int(np.floor(rounded_up_rating)) # Get the integer part
    half_star = 1 if rounded_up_rating - full_stars == 0.5 else 0 # Check for a half-star
    return '⭐' * full_stars + ('✩' if half_star else '')

# Streamlit UI
st.title("Movie Recommendation System")

# Movie recommendation dropdown
selected_movie = st.selectbox("Choose a movie:", movie_titles)

# Checkbox for optional filters
filter_options = st.checkbox("Additional options", value=False)

# Default recommendations without filters
if not filter_options:
    # Get recommendations for the selected movie without any filters
    recommendations = find_filtered_similar_movies(selected_movie, top_n=10, min_rating=0, genres=None, min_num_ratings=0)
    
    # Display recommendations without any filters
    st.subheader(f"Recommendations for '{selected_movie}'")
    for recommended_movie in recommendations:
        
        # Get average rating for each recommended movie
        avg_rating = data[data['title'] == recommended_movie]['rating'].mean()
      
        # Display the movie, average rating, and star rating
        st.write(f"**{recommended_movie}**")
        st.write(f"Average Rating: {avg_rating:.2f} ({display_star_rating(avg_rating)})")
        st.write("---")  # Line break between recommendations

# If the user opts to filter
if filter_options:
    st.sidebar.header("Filter Options")
    min_avg_rating = st.sidebar.slider("Min average rating:", min_value=0.5, max_value=5.0, value=3.5, step=0.5)
    min_num_ratings = st.sidebar.slider("Min number of ratings:", min_value=1, max_value=100, value=5, step=1)
    
    available_genres = genres_expanded['genres'].unique() # Now we have access to genres_expanded
    selected_genres = st.sidebar.multiselect("Choose Genre(s):", options=available_genres, default=available_genres)

    # Get recommendations for the selected movie with applied filters
    filtered_recommendations = find_filtered_similar_movies(
      selected_movie, top_n=10, min_rating=min_avg_rating, genres=set(selected_genres), min_num_ratings=min_num_ratings
    )

    # Display filtered recommendations
    st.subheader(f"Filtered Recommendations for '{selected_movie}'")
  
    if filtered_recommendations:
        for recommended_movie in filtered_recommendations:
            # Get average rating for each recommended movie
            avg_rating = data[data['title'] == recommended_movie]['rating'].mean()
          
            # Display the movie, average rating, and star rating
            st.write(f"**{recommended_movie}**")
            st.write(f"Average Rating: {avg_rating:.2f} ({display_star_rating(avg_rating)})")
            st.write("---")  # Line break between recommendations
    else:
        st.write("No movies found matching the criteria.")
      
# Movie rating distribution visualization dropdown
st.subheader("Rating Frequency")
selected_movie_for_viz = st.selectbox("Choose a movie to be visualized:", movie_titles)


# Get the rating distribution for the selected movie
movie_ratings = data[data['title'] == selected_movie_for_viz]

# Calculate the average rating for the selected movie
avg_rating_for_viz = movie_ratings['rating'].mean()

# Count the ratings and get the value counts
movie_ratings_full = movie_ratings['rating'].value_counts().reset_index()
movie_ratings_full.columns = ['Rating', 'Count']

# Ensure that all ratings are included in the final DataFrame (0.5 to 5.0 scale)
all_ratings = pd.DataFrame({'Rating': np.arange(0.5, 5.5, 0.5)})
movie_ratings_full = all_ratings.merge(movie_ratings_full, on='Rating', how='left').fillna(0)

# Create the Altair bar chart for the rating distribution
rating_chart = alt.Chart(movie_ratings_full).mark_bar().encode(
    x=alt.X('Rating:O', title='Rating'),
    y=alt.Y('Count:Q', title='Rating entries', 
            scale=alt.Scale(domain=[0, movie_ratings_full['Count'].max()], nice=True)),
    color=alt.Color('Rating:O', legend=None)
).properties(title=f"Average Rating: {avg_rating_for_viz:.2f}")

# Show the Altair chart
st.altair_chart(rating_chart, use_container_width=True)

st.markdown("""
### Dataset Information  

This recommendation system is based on the **MovieLens Latest Datasets**. The dataset consists of:  

- **100,000 ratings** applied to **9,000 movies** by **600 users**.  
- The dataset includes only movies **up to 2018**.  

The current implementation uses the **small dataset** to ensure functionality before scaling to a larger, more up-to-date dataset in future updates.  

**Source:** [MovieLens Dataset](https://grouplens.org/datasets/movielens/latest/)
""")
