# 📘 Book Recommendation Data Cleaning & Analysis

**Course:** IFT 520 – Analyzing Big Data  
**Institution:** Arizona State University  
**Instructor:** Dr. Asmaa Elbadrawy  

---

## 🧱 Overview
Two-phase project that cleans and analyzes book ratings data, builds personalized recommendation models, clusters user groups, and predicts missing user ages using regression models.

---

## 📂 Structure

book-recommendation-data-cleaning/
├── data/
├── step-1/
├── step-2/
│ ├── task1_recommender.py
│ ├── task2_clustering.py
│ ├── task3_estimate_ages.py
│ └── outputs/
├── Project_Step1-2.docx
├── Project_Step2_IFT511-1.docx
├── Book_Recommendations1.csv
├── ratings3-2.libsvm
└── README.md


---

## ⚙️ Tools
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, SciPy, scikit-learn  
- **Methods:** Collaborative Filtering, Clustering (KMeans, Ward, DBSCAN), Regression (Linear, Tree, Polynomial)

---

## 📈 Results
| Step | Output | Description |
|------|---------|-------------|
| 1 | `ratings3-2.libsvm` | Cleaned dataset in libsvm format |
| 2 – Task 1 | `Book_Recommendations1.csv` | Top 5 book recommendations |
| 2 – Task 2 | `clusters_top_books_k10.txt` | Top 10 books per cluster |
| 2 – Task 3 | `User_Age_Predictions.csv` | Predicted user ages |

---

## 🧠 Learnings
- Built end-to-end data pipelines for recommender systems.  
- Implemented clustering to identify user behavior segments.  
- Applied regression to estimate missing demographic data.  
- Produced reproducible academic code and documentation.

---

## 🧾 License
MIT License – free for educational use.
