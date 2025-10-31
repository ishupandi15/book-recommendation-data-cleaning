# Step 2 – Data Analysis

## Task 1: Building a Recommender System
- Implemented User–User Collaborative Filtering with cosine similarity (K=10).
- Generated top-5 unread book recommendations for each user.
- Output: `Book_Recommendations1.csv`

## Task 2: Discovering User Groups
- Clustered users using KMeans, Hierarchical, and DBSCAN.
- Listed top 10 popular books per cluster (K=10).
- Output: `clusters_top_books_k10.txt`

## Task 3: Estimating User Ages
- Trained Linear, Polynomial, and Decision Tree regression models.
- Used 5-fold cross-validation and selected the model with lowest RMSE.
- Predicted missing user ages and exported final CSV.
- Output: `User_Age_Predictions.csv`
