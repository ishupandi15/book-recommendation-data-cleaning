import os, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"), delimiter=';', on_bad_lines='skip')
users   = pd.read_csv(os.path.join(DATA_DIR, "Users.csv"), delimiter=';', on_bad_lines='skip')
books   = pd.read_csv(os.path.join(DATA_DIR, "Booksv.csv"), delimiter=';', on_bad_lines='skip')

ratings.rename(columns={'User-ID':'UserID','ISBN':'BookID','Rating':'Rating'}, inplace=True)
users.rename(columns={'User-ID':'UserID','Age':'Age'}, inplace=True)
books.rename(columns={'ISBN':'BookID','Book-Title':'Book_Title'}, inplace=True)

ratings["UserID"], ratings["BookID"] = ratings["UserID"].astype(str), ratings["BookID"].astype(str)
ratings["Rating"] = pd.to_numeric(ratings["Rating"], errors="coerce")
users["UserID"], users["Age"] = users["UserID"].astype(str), pd.to_numeric(users["Age"], errors="coerce")

TOP_U, TOP_B = 500, 500
ratings = ratings[ratings["UserID"].isin(ratings["UserID"].value_counts().nlargest(TOP_U).index)]
ratings = ratings[ratings["BookID"].isin(ratings["BookID"].value_counts().nlargest(TOP_B).index)]

u_map = {u:i for i,u in enumerate(ratings["UserID"].unique())}
b_map = {b:i for i,b in enumerate(ratings["BookID"].unique())}
ratings["u_idx"], ratings["b_idx"] = ratings["UserID"].map(u_map), ratings["BookID"].map(b_map)

M = csr_matrix((ratings["Rating"], (ratings["u_idx"], ratings["b_idx"])),
               shape=(len(u_map), len(b_map))).toarray()

u_df = pd.DataFrame({"UserID":[u for u in u_map.keys()]})
u_df = u_df.merge(users[["UserID","Age"]], on="UserID", how="left")

known = u_df["Age"].notna()
X, y = M[known.values], u_df.loc[known,"Age"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
def rmse_cv(model): return -cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error").mean()

lin = make_pipeline(StandardScaler(), LinearRegression())
poly = make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression())
tree = DecisionTreeRegressor(random_state=42, max_depth=12)

rmse_lin, rmse_poly, rmse_tree = rmse_cv(lin), rmse_cv(poly), rmse_cv(tree)
best = min([(lin, rmse_lin), (poly, rmse_poly), (tree, rmse_tree)], key=lambda x:x[1])[0]
print(f"✅ RMSE — Linear:{rmse_lin:.3f}, Poly:{rmse_poly:.3f}, Tree:{rmse_tree:.3f}")

best.fit(X, y)
missing = ~known.values
pred = best.predict(M[missing])

out = []
for uid, p in zip(u_df.loc[missing,"UserID"], pred):
    read = ratings[ratings["UserID"]==uid]["BookID"].tolist()
    titles = [books.loc[books["BookID"]==b,"Book_Title"].values[0] for b in read if b in books["BookID"].values]
    out.append({"User ID":uid, "Estimated Age":int(np.clip(p,5,100)), "List of Book Titles read by the user separated by commas":", ".join(titles)})

pd.DataFrame(out).to_csv(os.path.join(OUT_DIR,"User_Age_Predictions.csv"),sep=';',index=False)
print("✅ Saved predictions -> outputs/User_Age_Predictions.csv")
