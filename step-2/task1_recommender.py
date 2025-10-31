import os, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ratings_path = os.path.join(DATA_DIR, "Ratings.csv")
books_path   = os.path.join(DATA_DIR, "Booksv.csv")
out_path     = os.path.join(OUT_DIR, "Book_Recommendations1.csv")

r = pd.read_csv(ratings_path, delimiter=';', on_bad_lines='skip')
r.columns = [c.strip() for c in r.columns]
r.rename(columns={'User-ID':'UserID','ISBN':'BookID','Rating':'Rating'}, inplace=True)
r["UserID"], r["BookID"] = r["UserID"].astype(str), r["BookID"].astype(str)
r["Rating"] = pd.to_numeric(r["Rating"], errors="coerce")
r.dropna(subset=["Rating"], inplace=True)

TOP_U, TOP_B = 500, 500
top_u = r["UserID"].value_counts().nlargest(TOP_U).index
top_b = r["BookID"].value_counts().nlargest(TOP_B).index
r = r[r["UserID"].isin(top_u) & r["BookID"].isin(top_b)]

u_map = {u:i for i,u in enumerate(r["UserID"].unique())}
b_map = {b:i for i,b in enumerate(r["BookID"].unique())}
u_rev = {i:u for u,i in u_map.items()}
b_rev = {i:b for b,i in b_map.items()}

r["u_idx"] = r["UserID"].map(u_map)
r["b_idx"] = r["BookID"].map(b_map)
M = csr_matrix((r["Rating"], (r["u_idx"], r["b_idx"])),
               shape=(len(u_map), len(b_map)))

model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=11)
model.fit(M)
dist, idx = model.kneighbors(M)

recs = []
for u in range(M.shape[0]):
    neigh, sims = idx[u][1:], 1 - dist[u][1:]
    u_r = M[u].toarray().ravel()
    seen = set(np.where(u_r > 0)[0])
    cand = set(M[neigh].nonzero()[1]) - seen
    scores = {}
    for b in cand:
        neigh_r = M[neigh, b].toarray().ravel()
        mask = neigh_r > 0
        if mask.any():
            scores[b] = np.dot(sims[mask], neigh_r[mask]) / sims[mask].sum()
    for b, sc in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:5]:
        recs.append({"User_ID": u_rev[u], "Book_ID": b_rev[b], "Recommendation_Score": round(sc,3)})

recs = pd.DataFrame(recs)
books = pd.read_csv(books_path, delimiter=';', on_bad_lines='skip')
books.rename(columns={'ISBN':'Book_ID','Book-Title':'Book_Title'}, inplace=True)
out = recs.merge(books[["Book_ID","Book_Title"]], on="Book_ID", how="left")
out.to_csv(out_path, index=False)
print(f"✅ Saved recommendations -> {out_path}")
