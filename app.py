import os
import requests
import zipfile
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Dataset download function
def download_and_extract_data():
    url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_file = "ml-latest-small.zip"
    folder = "ml-latest-small"
    
    if not os.path.exists(folder):  # Check if the dataset folder already exists
        response = requests.get(url, stream=True)
        with open(zip_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall()

# Download the dataset if not present
download_and_extract_data()

# Load and process data
ratings = pd.read_csv("ml-latest-small/ratings.csv", usecols=["userId", "movieId", "rating"])
movies = pd.read_csv("ml-latest-small/movies.csv", usecols=["movieId", "title"])
data = pd.merge(ratings, movies, on="movieId")
user_item_matrix = data.pivot_table(index="userId", columns="title", values="rating")
user_item_matrix.fillna(0, inplace=True)
movie_user_matrix = user_item_matrix.T
movie_similarity = cosine_similarity(movie_user_matrix)
movie_indices = pd.Series(data=range(len(movie_user_matrix.index)), index=movie_user_matrix.index)

# Function to find similar movies
def find_similar_movies(movie_title, top_n=10):
    # Normalize movie titles (case insensitive, ignore year in parentheses)
    normalized_titles = {
        title.lower().rsplit(" (", 1)[0]: title for title in movie_indices.index
    }
    normalized_input = movie_title.strip().lower()

    if normalized_input not in normalized_titles:
        return ["Movie not found."]
    
    # Match normalized title back to actual title
    matched_title = normalized_titles[normalized_input]
    movie_index = movie_indices[matched_title]
    similar_movies = movie_similarity[movie_index]
    similar_movie_indices = np.argsort(similar_movies)[::-1][1:top_n+1]
    return movie_indices.iloc[similar_movie_indices].index.tolist()

# Streamlit UI
st.title("Lucas’ Mattscheiben Empfehlungen")
movie_title = st.text_input("Welches gute Stück haben wir denn schon gesehen wa ? :")

if st.button("Bekomme Vorschläge"):
    if movie_title:
        recommendations = find_similar_movies(movie_title)
        if recommendations[0] == "Film nicht gefunden.":
            st.write(f"Keine Vorschläge gefunden für  '{movie_title}'. Probier es stattdessden doch einfach mal mit Gemütlichkeit. Mit Ruhe und Gemütlichkeit. Oder lies doch mal einfahc ein Buch")
        else:
            st.write("**Vorschläge:**")
            for movie in recommendations:
                st.write(f"- {movie}")
    else:
        st.write("Bite gib ein Film ein.")
