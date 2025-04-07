# 🎬 Movie Recommendation System

## 🔗 Streamlit App  
👉 [Click here to try the app](https://movie-recommendation-25.streamlit.app/)

## 🌟 Table of Contents
- [🌟 Table of Contents](#-table-of-contents)
- [🔮 Project Overview](#-project-overview)
- [🔄 Data Pipeline & Preprocessing](#-data-pipeline--preprocessing)
- [💡 App Features](#-app-features)
- [⚠️ Limitations & Future Improvements](#️-limitations--future-improvements)
- [📄 Dataset Information](#-dataset-information)
- [📢 Disclaimer](#-disclaimer)

---

## 🔮 Project Overview

This Streamlit app is a content-based **movie recommendation system** using collaborative filtering techniques on user ratings. It allows users to:

- Select a movie and receive **similar recommendations** based on cosine similarity  
- Apply **filters** such as genre, minimum average rating, and number of ratings  
- Explore **visualizations** of rating distributions

---

## 🔄 Data Pipeline & Preprocessing

- Ratings and metadata are loaded from two CSVs: `movies.csv` and `ratings.csv`
- Genres are extracted and expanded for filtering functionality
- A **user-item matrix** is created (users as rows, movies as columns)
- **Cosine similarity** is computed between movie vectors to find similar titles

---

## 💡 App Features

### 🎬 Movie Recommendation
- Choose a movie and get **10 similar titles**
- Optionally filter by:
  - Minimum average rating
  - Minimum number of ratings
  - Genres

### ⭐ Star Ratings
- Each recommended title displays:
  - Average rating
  - Visualized with full and half **star ratings**

### 📊 Rating Distribution Visualizer
- Pick any movie and view its **rating frequency distribution** 

---

## ⚠️ Limitations & Future Improvements

### Current Limitations
- Uses the **MovieLens small dataset**, limited to **100K ratings up to 2018**
- Recommendations are based on **collaborative filtering only** (not hybrid)
- Deployment constraints limit the size of the dataset used in this demo

### Future Improvements
- Upgrade to the **full or latest MovieLens dataset** once deployment constraints allow
- Add user login and interaction tracking for **personalized recommendations**

---

## 📄 Dataset Information

This system uses the [**MovieLens Latest Small Dataset**](https://grouplens.org/datasets/movielens/latest/), maintained by GroupLens.

**Summary:**
- ~100,000 ratings from ~600 users on ~9,000 movies
- Includes metadata such as genres
- Filtered to movies **released up to 2018**

The current version uses this small dataset to test the architecture and features, but the aim is to transition to a **larger and more up-to-date** dataset in future releases.

---

## 📢 Disclaimer

This project is intended for **educational and non-commercial purposes only**.  
All data is sourced from publicly available datasets maintained by [GroupLens](https://grouplens.org).  
The project does **not offer any official movie streaming or proprietary film data**.
