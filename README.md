# 📘 Book Recommendation Data Cleaning & Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-yellow?logo=pandas)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

### 🏫 **Arizona State University — IFT 520: Analyzing Big Data**
**Instructor:** Dr. Asmaa Elbadrawy  

---

## 🚀 Overview
A two-phase academic project focused on cleaning, analyzing, and modeling large-scale book rating data.  
The goal was to build a **personalized recommendation engine**, discover **user behavior clusters**, and **predict missing demographic data** using machine-learning models.

---

## 🧩 Project Structure

<img width="298" height="503" alt="image" src="https://github.com/user-attachments/assets/613127ee-123f-4697-9427-c20a6f43950f" />


---

## ⚙️ Technologies & Methods

| **Category** | **Tools / Methods** |
|---------------|----------------------|
| **Languages** | Python 3.10 + |
| **Libraries** | Pandas • NumPy • SciPy • scikit-learn |
| **Data Handling** | Sparse Matrices • Data Cleaning • Feature Mapping |
| **Modeling** | User–User Collaborative Filtering • Cosine Similarity |
| **Clustering** | KMeans • Agglomerative (Ward) • DBSCAN |
| **Regression** | Linear • Polynomial • Decision Tree • K-Fold CV |
| **Outputs** | CSV • libsvm • Text Reports |


---

## 🧱 Phase 1 – Data Preparation
- Processed raw book-ratings data into a sparse matrix (User × Book).  
- Exported dataset into **.libsvm** format for machine-learning pipelines.  
- Implemented with Pandas, SciPy, and scikit-learn.  
**Output:** `ratings3-2.libsvm`  

---

## 📊 Phase 2 – Data Analysis & Modeling

### 🔹 Task 1 – Collaborative Filtering
- Built a **User–User Recommender** using cosine similarity (K = 10).  
- Generated top-5 unread book recommendations per user.  
**Output:** `Book_Recommendations1.csv`

### 🔹 Task 2 – Discovering User Groups
- Clustered users with **KMeans**, **Agglomerative (Ward)**, and **DBSCAN**.  
- Analyzed most-frequent books per cluster to interpret reading trends.  
**Output:** `clusters_top_books_k10.txt`

### 🔹 Task 3 – Predicting User Ages
- Trained **Linear**, **Polynomial**, and **Decision Tree** regressors with 5-fold CV.  
- Selected model with lowest RMSE and predicted ages for users missing demographic info.  
**Output:** `User_Age_Predictions.csv`

---

## 📈 Results Summary

| 📄 **Output File** | 🧠 **Purpose** |
|--------------------|----------------|
| `ratings3-2.libsvm` | Clean, sparse matrix of user–book ratings |
| `Book_Recommendations1.csv` | Personalized top-5 book recommendations |
| `clusters_top_books_k10.txt` | Top-10 popular books per user cluster |
| `User_Age_Predictions.csv` | Estimated ages for users missing demographic data |


---

## 🧠 Key Learnings
- Built an end-to-end ML pipeline from preprocessing → modeling → evaluation.  
- Applied collaborative filtering and clustering for user-behavior analysis.  
- Used regression and cross-validation for demographic prediction.  
- Gained experience in reproducible data-science documentation and GitHub collaboration.

---

## 📜 License
This project is released under the **MIT License**.  
You are free to use, modify, and share with proper attribution.

---

## 🔗 Connect / Portfolio
**Ishwariya Pandi** – *Graduate Student, Arizona State University*  
📎 [GitHub Repository](https://github.com/ishupandi15/book-recommendation-data-cleaning)

