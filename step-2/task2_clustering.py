import os, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"), delimiter=';', on_bad_lines='skip')
ratings.rename(columns={'User-ID':'UserID','ISBN':'BookID','Rating':'Rating'}, inplace=True)
ratings["UserID"], ratings["BookID"] = ratings["UserID"].astype(str), ratings["BookID"].astype(str)
ratings["Rating"] = pd.to_numeric(ratings["Rating"], errors="coerce")
ratings.dropna(subset=["Rating"], inplace=True)

TOP_U, TOP_B = 500, 500
ratings = ratings[ratings["UserID"].isin(ratings["UserID"].value_counts().nlargest(TOP_U).index)]
ratings = ratings[ratings["BookID"].isin(ratings["BookID"].value_counts().nlargest(TOP_B).index)]

u_map = {u:i for i,u in enumerate(ratings["UserID"].unique())}
b_map = {b:i for i,b in enumerate(ratings["BookID"].unique())}
ratings["u_idx"] = ratings["UserID"].map(u_map)
ratings["b_idx"] = ratings["BookID"].map(b_map)

M = csr_matrix((ratings["Rating"], (ratings["u_idx"], ratings["b_idx"])),
               shape=(len(u_map), len(b_map)))
M_bin = M.copy(); M_bin.data = np.ones_like(M_bin.data)

def top_books(labels, method, out_file="clusters_top_books_k10.txt"):
    with open(os.path.join(OUT_DIR, out_file), "a") as f:
        for c in range(10):
            idx = np.where(labels==c)[0]
            if len(idx)==0: continue
            cols = M_bin[idx].nonzero()[1]
            freq = Counter(cols).most_common(10)
            f.write(f"\n{method} - Cluster {c}\n")
            for b_idx, count in freq:
                f.write(f"Book {b_idx}, Frequency {count}\n")

km = KMeans(n_clusters=10, random_state=42, n_init="auto")
km_labels = km.fit_predict(M)
top_books(km_labels, "KMeans")

agg = AgglomerativeClustering(n_clusters=10, linkage="ward")
agg_labels = agg.fit_predict(M.toarray())
top_books(agg_labels, "Agglomerative")

for eps in [0.6,0.7,0.8]:
    db = DBSCAN(eps=eps, min_samples=5, metric="cosine")
    db_labels = db.fit_predict(M)
    with open(os.path.join(OUT_DIR, "dbscan_summary.txt"), "a") as f:
        f.write(f"eps={eps}, clusters={len(set(db_labels[db_labels>=0]))}\n")

print("✅ Clustering results saved to outputs/")
