import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="👕",
    layout="wide",
)

# Expected structure:
# project/
#   app.py
#   data/
#     users.csv
#     products.csv
#     interaction_events.csv
DATA_PATH = "data"
USERS_FILE = os.path.join(DATA_PATH, "users.csv")
PRODUCTS_FILE = os.path.join(DATA_PATH, "products.csv")
INTERACTIONS_FILE = os.path.join(DATA_PATH, "interaction_events.csv")
ARTIFACT_FILE = os.path.join(DATA_PATH, "fashion_recommender_artifacts.joblib")

EVENT_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "wishlist": 3.0,
    "add_to_cart": 4.0,
    "purchase": 5.0,
    "not_interested": -3.0,
}
CONTENT_WEIGHT = 0.45
COLLABORATIVE_WEIGHT = 0.40
POPULARITY_WEIGHT = 0.15


def parse_listish(value):
    """Convert comma/pipe/semicolon-separated preference values to text."""
    if pd.isna(value):
        return ""
    return str(value).replace("|", " ").replace(",", " ").replace(";", " ").strip()


def normalize_scores(scores):
    """Min-max normalize a numeric score array safely."""
    scores = np.asarray(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-8:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


@st.cache_data(show_spinner="Loading fashion data...")
def load_data():
    missing = [
        p for p in (USERS_FILE, PRODUCTS_FILE, INTERACTIONS_FILE)
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing data file(s): " + ", ".join(missing)
            + ". Put users.csv, products.csv, and interaction_events.csv "
              "inside the data/ folder next to app.py."
        )

    users = pd.read_csv(USERS_FILE, low_memory=False)
    products = pd.read_csv(PRODUCTS_FILE, low_memory=False)

    interaction_columns = [
        "user_id", "product_id", "event_type", "timestamp",
        "time_spent_seconds", "scroll_depth", "quantity", "source", "session_id"
    ]
    header = pd.read_csv(INTERACTIONS_FILE, nrows=0).columns.tolist()
    usecols = [c for c in interaction_columns if c in header]
    interactions = pd.read_csv(
        INTERACTIONS_FILE,
        usecols=usecols or None,
        low_memory=False,
    )

    for col in ("user_id", "product_id"):
        if col in interactions.columns:
            interactions[col] = interactions[col].astype("string").str.strip()
    if "user_id" in users.columns:
        users["user_id"] = users["user_id"].astype("string").str.strip()
    if "product_id" in products.columns:
        products["product_id"] = products["product_id"].astype("string").str.strip()

    if "timestamp" in interactions.columns:
        interactions["timestamp"] = pd.to_datetime(
            interactions["timestamp"], errors="coerce", utc=True
        )
    if "event_type" in interactions.columns:
        interactions["event_type"] = (
            interactions["event_type"].astype("string").str.lower().str.strip()
        )

    # Avoid duplicate product IDs causing ambiguous joins and mismatched vectors.
    products = products.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    users = users.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    return users, products, interactions


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        st.error(f"{label} is missing required column(s): {', '.join(missing)}")
        st.stop()


def weighted_interactions(interactions):
    events = interactions.copy()
    events["event_type"] = events["event_type"].astype(str).str.lower().str.strip()
    events["weight"] = events["event_type"].map(EVENT_WEIGHTS).fillna(0.0)
    if "quantity" in events.columns:
        qty = pd.to_numeric(events["quantity"], errors="coerce").fillna(1).clip(1, 5)
        mask = events["event_type"].isin(["purchase", "add_to_cart"])
        events.loc[mask, "weight"] *= qty[mask]
    return events


@st.cache_resource(show_spinner="Loading trained Colab model artifacts...")
def load_model_artifact():
    """Load the exact content/SVD artifacts exported from the Colab notebook."""
    if not os.path.exists(ARTIFACT_FILE):
        raise FileNotFoundError(
            f"Model artifact not found: {ARTIFACT_FILE}. Export the trained model "
            "from Colab and place fashion_recommender_artifacts.joblib inside data/."
        )

    artifact = joblib.load(ARTIFACT_FILE)
    required = [
        "products", "tfidf_matrix", "product_to_index", "vectorizer",
        "cf_user_to_index", "cf_user_factors", "cf_item_factors", "cf_products"
    ]
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ValueError(
            "The Colab artifact is missing required keys: " + ", ".join(missing)
            + ". Re-run the export cell in Colab."
        )

    catalog = artifact["products"].copy().reset_index(drop=True)
    catalog["product_id"] = catalog["product_id"].astype(str).str.strip()
    product_ids = [str(pid).strip() for pid in artifact["cf_products"]]

    if len(catalog) != artifact["tfidf_matrix"].shape[0]:
        raise ValueError("Artifact catalog and TF-IDF matrix rows do not align.")
    if len(product_ids) != artifact["tfidf_matrix"].shape[0]:
        raise ValueError("Artifact product IDs and TF-IDF matrix rows do not align.")
    if len(product_ids) != artifact["cf_item_factors"].shape[0]:
        raise ValueError("Artifact product IDs and SVD item factors do not align.")

    cf_model = {
        "ready": True,
        "message": "",
        "user_to_row": {str(k): int(v) for k, v in artifact["cf_user_to_index"].items()},
        "user_ids": [str(uid) for uid in artifact.get("cf_users", artifact["cf_user_to_index"].keys())],
        "product_ids": product_ids,
        "user_factors": artifact["cf_user_factors"],
        "item_factors": artifact["cf_item_factors"],
        "components": int(getattr(artifact.get("cf_model"), "n_components", artifact["cf_item_factors"].shape[1])),
        "explained_variance": float(getattr(artifact.get("cf_model"), "explained_variance_ratio_", np.array([])).sum()) if artifact.get("cf_model") is not None else None,
        "matrix_shape": tuple(artifact.get("cf_matrix_shape", (len(artifact.get("cf_users", artifact["cf_user_to_index"])), len(product_ids)))),
        "nnz": artifact.get("cf_nnz"),
    }

    product_to_index = {str(k): int(v) for k, v in artifact["product_to_index"].items()}
    return catalog, artifact["tfidf_matrix"], product_to_index, artifact["vectorizer"], cf_model


def get_popular_products(interactions, product_catalog, top_n=20):
    events = weighted_interactions(interactions)
    events = events[events["weight"] > 0]
    popular = (
        events.groupby("product_id", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "interaction_score"})
        .sort_values("interaction_score", ascending=False)
    )
    popular["product_id"] = popular["product_id"].astype(str)
    catalog = product_catalog.copy()
    catalog["product_id"] = catalog["product_id"].astype(str)
    popular = popular[popular["product_id"].isin(set(catalog["product_id"]))]
    popular = popular.merge(catalog, on="product_id", how="inner")

    if "availability" in popular.columns:
        unavailable = popular["availability"].astype(str).str.lower().isin(
            ["false", "0", "no", "out_of_stock", "unavailable"]
        )
        popular = popular.loc[~unavailable]
    return popular.head(top_n).reset_index(drop=True)


def recommend_similar_products(product_id, product_catalog, matrix, product_to_index, top_n=10):
    product_id = str(product_id)
    if product_id not in product_to_index:
        return pd.DataFrame()
    idx = product_to_index[product_id]
    scores = cosine_similarity(matrix[idx], matrix).ravel()
    scores[idx] = -1
    top_indices = scores.argsort()[::-1][:top_n]
    result = product_catalog.iloc[top_indices].copy()
    result["similarity_score"] = scores[top_indices]
    return result.sort_values("similarity_score", ascending=False).reset_index(drop=True)


def recommend_for_user(
    user_id, users, interactions, product_catalog, matrix,
    product_to_index, vectorizer, cf_model, top_n=10,
    content_weight=CONTENT_WEIGHT,
    collaborative_weight=COLLABORATIVE_WEIGHT,
    popularity_weight=POPULARITY_WEIGHT,
):
    user_id = str(user_id).strip()
    weights = np.array(
        [content_weight, collaborative_weight, popularity_weight], dtype=float
    )
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Recommendation weights must be non-negative and sum to > 0.")
    weights = weights / weights.sum()

    catalog = product_catalog.copy().reset_index(drop=True)
    catalog["product_id"] = catalog["product_id"].astype(str)
    product_ids = catalog["product_id"].tolist()
    history = interactions[interactions["user_id"].astype(str).str.strip() == user_id].copy()
    if "timestamp" in history.columns:
        history = history.sort_values("timestamp")

    seen = set(history["product_id"].dropna().astype(str).str.strip())
    disliked = set(
        history.loc[history["event_type"].astype(str).str.lower() == "not_interested",
                    "product_id"].dropna().astype(str).str.strip()
    )

    user_rows = users[users["user_id"].astype(str).str.strip() == user_id]
    content_scores = np.zeros(len(product_ids), dtype=np.float32)

    positive = weighted_interactions(history)
    positive = positive[positive["weight"] > 0]
    valid_seeds = []
    if not positive.empty:
        if "timestamp" in positive.columns:
            positive = positive.sort_values("timestamp")
        valid_seeds = [
            pid for pid in positive["product_id"].astype(str).drop_duplicates().tail(5)
            if pid in product_to_index
        ]

    if valid_seeds:
        seed_indices = [product_to_index[pid] for pid in valid_seeds]
        similarities = cosine_similarity(matrix[seed_indices], matrix)
        content_scores = similarities.max(axis=0).astype(np.float32)
    elif not user_rows.empty:
        user = user_rows.iloc[0]
        preference_fields = [
            "preferred_categories", "preferred_brands", "preferred_colors",
            "preferred_materials", "preferred_styles", "preferred_occasions",
            "preferred_sizes",
        ]
        profile_text = " ".join(
            parse_listish(user.get(field, ""))
            for field in preference_fields
            if field in users.columns and pd.notna(user.get(field, ""))
        ).lower().strip()
        if profile_text:
            profile_vec = vectorizer.transform([profile_text])
            content_scores = cosine_similarity(profile_vec, matrix).ravel().astype(np.float32)

    content_scores = normalize_scores(content_scores)

    # SVD scores align to the same catalog order as the content model.
    collaborative_scores = np.zeros(len(product_ids), dtype=np.float32)
    if cf_model["ready"] and user_id in cf_model["user_to_row"]:
        row = cf_model["user_to_row"][user_id]
        raw_cf = cf_model["user_factors"][row] @ cf_model["item_factors"].T
        collaborative_scores = normalize_scores(raw_cf)

    # Popularity scores use positive weighted events only.
    weighted = weighted_interactions(interactions)
    positive_events = weighted[weighted["weight"] > 0].copy()
    positive_events["product_id"] = positive_events["product_id"].astype(str)
    popularity_map = (
        positive_events.groupby("product_id")["weight"].sum().to_dict()
    )
    raw_popularity = np.array(
        [np.log1p(max(float(popularity_map.get(pid, 0.0)), 0.0)) for pid in product_ids],
        dtype=np.float32,
    )
    popularity_scores = normalize_scores(raw_popularity)

    final_scores = (
        weights[0] * content_scores
        + weights[1] * collaborative_scores
        + weights[2] * popularity_scores
    )

    ranked = pd.DataFrame({
        "product_id": product_ids,
        "content_score": content_scores,
        "collaborative_score": collaborative_scores,
        "popularity_score": popularity_scores,
        "recommendation_score": final_scores,
    })

    # Do not recommend already interacted-with or explicitly disliked items.
    ranked = ranked[~ranked["product_id"].isin(seen | disliked)]
    ranked = ranked.merge(catalog, on="product_id", how="inner")

    if not user_rows.empty and "price" in ranked.columns:
        user = user_rows.iloc[0]
        minimum = pd.to_numeric(pd.Series([user.get("preferred_min_price")]), errors="coerce").iloc[0]
        maximum = pd.to_numeric(pd.Series([user.get("preferred_max_price")]), errors="coerce").iloc[0]
        price = pd.to_numeric(ranked["price"], errors="coerce")
        if pd.notna(minimum):
            ranked = ranked[price >= minimum]
        if pd.notna(maximum):
            ranked = ranked[pd.to_numeric(ranked["price"], errors="coerce") <= maximum]

    if "availability" in ranked.columns:
        availability = ranked["availability"].astype(str).str.lower()
        unavailable = availability.isin(["false", "0", "no", "out_of_stock", "unavailable"])
        ranked = ranked[~unavailable]

    return ranked.sort_values("recommendation_score", ascending=False).head(top_n).reset_index(drop=True)


# Load source CSVs for user history, activity analytics, and profile preferences.
try:
    users, csv_products, interactions = load_data()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

require_columns(users, ["user_id"], "Users data")
require_columns(csv_products, ["product_id"], "Products data")
require_columns(interactions, ["user_id", "product_id", "event_type"], "Interactions data")

# Use the exact product catalog, TF-IDF matrix, vectorizer, and SVD factors
# exported from Colab. No TF-IDF or SVD training happens in this app.
try:
    product_catalog, tfidf_matrix, product_to_index, content_vectorizer, cf_model = load_model_artifact()
except Exception as exc:
    st.error(f"Could not load the trained Colab model: {exc}")
    st.stop()

# The artifact catalog is the model's aligned catalog; CSV products remain useful for overview.
products = csv_products

st.title("👕 Fashion Recommendation System")
st.caption("Hybrid recommendations combine content similarity, collaborative filtering, and product popularity.")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home", "Dataset Overview", "Popular Products", "Similar Products",
        "User Recommendations", "Interaction Analytics", "About"
    ],
)

if menu == "Home":
    st.header("🏠 Project Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Users", f"{users['user_id'].nunique():,}")
    c2.metric("Products", f"{products['product_id'].nunique():,}")
    c3.metric("Interactions", f"{len(interactions):,}")

    c4, c5, c6 = st.columns(3)
    c4.metric("SVD model", "Ready" if cf_model["ready"] else "Fallback")
    c5.metric("SVD components", str(cf_model.get("components", 0)))
    c6.metric("Hybrid weights", "45 / 40 / 15")

    event_counts = interactions["event_type"].value_counts()
    st.subheader("Interaction types")
    st.bar_chart(event_counts)

    st.markdown("""
    **Recommendation methods**
    - **Content-based filtering (45%)** — matches product attributes to products the user engaged with, or their profile preferences.
    - **Collaborative filtering (40%)** — uses TruncatedSVD on the user–product interaction matrix.
    - **Popularity (15%)** — boosts products with stronger positive engagement.

    The final list excludes products already interacted with and products marked `not_interested`,
    then applies user price preferences and availability filtering where those fields exist.
    """)

    if cf_model["ready"]:
        nnz_text = (
            f" with {cf_model['nnz']:,} non-zero user-product pairs"
            if cf_model.get("nnz") is not None else ""
        )
        st.success(
            f"Loaded Colab SVD factors for {cf_model['matrix_shape'][0]:,} users × "
            f"{cf_model['matrix_shape'][1]:,} products{nnz_text}."
        )
    else:
        st.warning(
            "Collaborative filtering is unavailable: " + cf_model["message"]
            + " Content and popularity scores will still be used."
        )

elif menu == "Dataset Overview":
    st.header("📊 Dataset Overview")
    for label, df in [("Users", users), ("Products", products), ("Interactions", interactions)]:
        st.subheader(label)
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        st.dataframe(df.head(10), use_container_width=True)

elif menu == "Popular Products":
    st.header("🔥 Popular Products")
    top_n = st.slider("Number of products", 5, 50, 20)
    popular = get_popular_products(interactions, product_catalog, top_n)
    st.dataframe(popular, use_container_width=True)
    if not popular.empty:
        chart_col = "product_name" if "product_name" in popular.columns else "product_id"
        st.bar_chart(popular.set_index(chart_col)["interaction_score"])

elif menu == "Similar Products":
    st.header("👕 Similar Product Recommendations")
    options = product_catalog[["product_id"]].copy()
    options["display"] = (
        product_catalog.get("product_name", product_catalog["product_id"])
        .fillna(product_catalog["product_id"].astype(str)).astype(str)
    )
    options = options.drop_duplicates("product_id").reset_index(drop=True)
    selected_id = st.selectbox(
        "Choose a product",
        options["product_id"].astype(str).tolist(),
        format_func=lambda pid: (
            f"{options.loc[options['product_id'].astype(str) == pid, 'display'].iloc[0]} ({pid})"
        ),
    )
    recs = recommend_similar_products(
        selected_id, product_catalog, tfidf_matrix, product_to_index, top_n=10
    )
    st.subheader("Recommended similar products")
    st.dataframe(recs, use_container_width=True)

elif menu == "User Recommendations":
    st.header("👤 Personalized Hybrid Recommendations")
    user_ids = users["user_id"].dropna().astype(str).drop_duplicates().tolist()
    if not user_ids:
        st.warning("No users are available in users.csv.")
        st.stop()

    selected_user = st.selectbox("Choose a user", user_ids[:10000])
    top_n = st.slider("Number of recommendations", 5, 20, 10, step=5)

    # Make weights adjustable for demonstrations.
    with st.expander("Advanced: adjust hybrid scoring weights"):
        st.caption("Weights are normalized automatically if they do not add up to 100%.")
        content_weight = st.slider("Content-based weight", 0, 100, 45, 5) / 100
        collaborative_weight = st.slider("Collaborative-filtering weight", 0, 100, 40, 5) / 100
        popularity_weight = st.slider("Popularity weight", 0, 100, 15, 5) / 100

    st.write(f"Selected user: **{selected_user}**")

    history = interactions[interactions["user_id"].astype(str) == selected_user].copy()
    if "timestamp" in history.columns:
        history = history.sort_values("timestamp", ascending=False)
    history = history.head(10).merge(
        products.drop_duplicates("product_id"),
        on="product_id", how="left", suffixes=("_event", "_product")
    )
    st.subheader("Recent interactions")
    st.dataframe(history, use_container_width=True)

    recs = recommend_for_user(
        selected_user, users, interactions, product_catalog,
        tfidf_matrix, product_to_index, content_vectorizer, cf_model,
        top_n=top_n,
        content_weight=content_weight,
        collaborative_weight=collaborative_weight,
        popularity_weight=popularity_weight,
    )

    st.subheader("Recommended products")
    if recs.empty:
        st.info("No products matched this user's filters. Try another user or adjust the user's price preferences.")
    else:
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Recommendations", len(recs))
        metric2.metric("Unique categories", recs["category"].nunique() if "category" in recs.columns else "N/A")
        metric3.metric("Mean final score", f"{recs['recommendation_score'].mean():.3f}")

        # Product cards with image links when provided in the dataset.
        card_cols = st.columns(2)
        for i, (_, row) in enumerate(recs.iterrows()):
            with card_cols[i % 2]:
                with st.container(border=True):
                    if "image_url" in recs.columns and pd.notna(row.get("image_url")) and str(row.get("image_url")).strip():
                        st.image(str(row["image_url"]), use_container_width=True)
                    st.markdown(f"**{row.get('product_name', row['product_id'])}**")
                    st.caption(
                        f"Product ID: {row['product_id']} · "
                        f"Category: {row.get('category', 'N/A')} · "
                        f"Brand: {row.get('brand', 'N/A')}"
                    )
                    if "price" in recs.columns and pd.notna(row.get("price")):
                        st.write(f"Price: {row['price']}")
                    score_a, score_b = st.columns(2)
                    score_a.metric("Final score", f"{row['recommendation_score']:.3f}")
                    score_b.metric("Content", f"{row['content_score']:.3f}")
                    score_c, score_d = st.columns(2)
                    score_c.metric("Collaborative", f"{row['collaborative_score']:.3f}")
                    score_d.metric("Popularity", f"{row['popularity_score']:.3f}")
                    if "product_url" in recs.columns and pd.notna(row.get("product_url")):
                        st.markdown(f"[View product]({row['product_url']})")

        st.subheader("Score breakdown")
        score_cols = [
            "product_id", "product_name", "category", "brand", "price",
            "content_score", "collaborative_score", "popularity_score",
            "recommendation_score"
        ]
        score_cols = [col for col in score_cols if col in recs.columns]
        st.dataframe(
            recs[score_cols].style.format({
                "content_score": "{:.3f}",
                "collaborative_score": "{:.3f}",
                "popularity_score": "{:.3f}",
                "recommendation_score": "{:.3f}",
            }, na_rep="N/A"),
            use_container_width=True,
        )

elif menu == "Interaction Analytics":
    st.header("📈 Interaction Analytics")
    event_counts = (
        interactions["event_type"].value_counts()
        .rename_axis("event_type").reset_index(name="count")
    )
    st.subheader("Events by type")
    st.dataframe(event_counts, use_container_width=True)
    st.bar_chart(event_counts.set_index("event_type")["count"])

    if "timestamp" in interactions.columns:
        dated = interactions.dropna(subset=["timestamp"]).copy()
        if not dated.empty:
            dated["date"] = dated["timestamp"].dt.date
            daily = dated.groupby("date").size().rename("interactions")
            st.subheader("Interactions over time")
            st.line_chart(daily)

    st.subheader("Most active users")
    active = (
        interactions["user_id"].value_counts().head(20)
        .rename_axis("user_id").reset_index(name="interaction_count")
    )
    st.dataframe(active, use_container_width=True)

elif menu == "About":
    st.header("ℹ️ About This Project")
    st.markdown("""
    **Project:** Fashion Recommendation System  
    **Authors:** Manushree Bartaria & Mahi Agrawal

    This application uses the supplied CSV datasets and combines:
    - Weighted popularity-based ranking
    - Content-based product similarity using TF-IDF and cosine similarity
    - Collaborative filtering using TruncatedSVD on a user-product interaction matrix
    - Hybrid ranking with configurable content, collaborative, and popularity weights

    **Default hybrid weights:** 45% content-based, 40% collaborative filtering, 15% popularity.

    This implementation does **not** use reinforcement learning. The SVD model is a
    baseline collaborative-filtering method trained when the app starts and cached
    for the current Streamlit process.
    """)
