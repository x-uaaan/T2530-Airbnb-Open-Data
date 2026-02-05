import json
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import seaborn as sns
import pyfpgrowth
import matplotlib.pyplot as plt

#--------------------------------------------------------------------- Page configuration ---------------------------------------------------------------------#

st.set_page_config(page_title="Airbnb Analytics System", layout="wide")
sns.set(style="whitegrid")
st.title("🏠 Airbnb Analytics System")

#-------------------------------------------------------------------------- Load Data -------------------------------------------------------------------------#
 
if "data" not in st.session_state:
    df = pd.read_csv("Airbnb_Open_Data_cleaned.csv")
    st.session_state.data = df

df = st.session_state.data
st.success("Retail data loaded successfully!")

if "total price" not in df.columns:
    if "price" in df.columns and "service fee" in df.columns:
        df["total price"] = (
            pd.to_numeric(df["price"], errors="coerce") +
            pd.to_numeric(df["service fee"], errors="coerce")
        )
        
#----------------------------------------------------------------------- General Analysis ---------------------------------------------------------------------#
    
# Categorical plots
def pltshow_cat(col):
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = df[col].astype("string").value_counts(dropna=False)
    if sns is not None:
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    else:
        ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"{col} distribution")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Total price distribution
def pltshow_price(t, p, f):
    if t in df.columns:
        total_series = pd.to_numeric(df[t], errors="coerce")
    elif p in df.columns and f in df.columns:
        price_series = pd.to_numeric(df[p], errors="coerce")
        fee_series = pd.to_numeric(df[f], errors="coerce")
        total_series = price_series + fee_series
    else:
        total_series = None

    if total_series is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        if sns is not None:
            sns.histplot(total_series.dropna(), bins=30, kde=True)
        else:
            total_series.dropna().plot(kind="hist", bins=30)
        ax.set_title("Total Price distribution")
        ax.set_xlabel("total price")
        ax.set_ylabel("frequency")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        print("Total price/price/service fee columns not found, skipping total price plot.")

# Construction year
def pltshow_year(c):
    line_df = df if "df" in globals() else df

    series = pd.to_numeric(line_df[c], errors="coerce").dropna()
    counts = series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(counts.index, counts.values, marker="o", linewidth=1)
    ax.set_title("Construction year (count by year)")
    ax.set_xlabel("Construction year")
    ax.set_ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Numeric plots
def pltshow_num(col):

    fig, ax = plt.subplots(figsize=(8, 4))
    series = pd.to_numeric(df[col], errors="coerce")

    if col == "reviews per month":
        series = series[(series >= 0) & (series <= 13)]
        bins = 26
    elif col == "calculated host listings count":
        series = series[(series >= 0) & (series <= 100)]
        bins = 25
    elif col == "review rate number":
        series = series[(series >= 1) & (series <= 5)]
        bins = range(1, 7)
    else:
        bins = 30

    if col == "review rate number":
        counts = series.dropna().round(0).astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        if sns is not None:
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
        else:
            ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"{col} distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if sns is not None:
        sns.histplot(series.dropna(), bins=bins, kde=True, ax=ax)
    else:
        ax.hist(series, bins=bins)

    if col == "reviews per month":
        ax.set_xlim(0, 13)
    if col == "calculated host listings count":
        ax.set_xlim(0, 100)

    ax.set_title(f"{col} distribution")
    ax.set_xlabel(col)
    ax.set_ylabel("frequency")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Main function for general analysis
def general_analysis():
    categorical_cols = [
        "host_identity_verified",
        "neighbourhood group",
        "instant_bookable",
        "cancellation_policy",
        "room type",
    ]
    numeric_cols = [
        "minimum nights",
        "number of reviews",
        "reviews per month",
        "review rate number",
        "calculated host listings count",
        "availability 365",
    ]
    select = st.selectbox("Select Analysis:",
        ["Verified host", "Neighbourhood group", "Instant bookable", "Cancellation policy", "Room Type", 
         "Total Price", "Construction Year", "Minimum nights", "Number of reviews", "Reviews per month",
         "Review rate number", "Calculated host listing count", "Availability 365"]
    )
    
    if select == "Verified host":
        pltshow_cat(categorical_cols[0])
    elif select == "Neighbourhood group":
        pltshow_cat(categorical_cols[1])
    elif select == "Instant bookable":
        pltshow_cat(categorical_cols[2])
    elif select == "Cancellation policy":
        pltshow_cat(categorical_cols[3])
    elif select == "Room Type":
        pltshow_cat(categorical_cols[4])    
    elif select == "Total Price":
        total_col = "total price"
        price_col = "price"
        fee_col = "service fee"
        pltshow_price(total_col, price_col, fee_col)
    elif select == "Construction Year":
        col = "Construction year"
        pltshow_year(col)
    elif select == "Minimum nights":
        pltshow_num(numeric_cols[0])
    elif select == "Number of reviews":
        pltshow_num(numeric_cols[1])
    elif select == "Reviews per month":
        pltshow_num(numeric_cols[2])
    elif select == "Review rate number":
        pltshow_num(numeric_cols[3])
    elif select == "Calculated host listing count":
        pltshow_num(numeric_cols[4])
    elif select == "Availability 365":
        pltshow_num(numeric_cols[5])

#------------------------------------------------------------------------ Webapp Filter -----------------------------------------------------------------------#
def filter_sort():

    neighbourhood = st.multiselect(
        "Neighbourhood Group",
        options=df["neighbourhood group"].unique(),
        default=df["neighbourhood group"].unique()
    )

    
    min_avail, max_avail = st.slider(
        "Availability 365",
        int(df["availability 365"].min()),
        int(df["availability 365"].max()),
        (int(df["availability 365"].min()), int(df["availability 365"].max()))
    )

    min_night, max_night = st.slider(
        "Minimum nights",
        int(df["minimum nights"].min()),
        int(df["minimum nights"].max()),
        (int(df["minimum nights"].min()), int(df["minimum nights"].max()))
    )
    
    instant = st.multiselect(
        "Instant bookable",
        options=df["instant_bookable"].unique(),
        default=df["instant_bookable"].unique()
    )

    type = st.multiselect(
        "Room type",
        options=df["room type"].unique(),
        default=df["room type"].unique()
    )

    # Apply filters
    df_filter = df[
        (df["neighbourhood group"].isin(neighbourhood)) &
        (df["availability 365"] >= min_avail) &
        (df["availability 365"] <= max_avail) &
        (df["minimum nights"] >= min_night) &
        (df["minimum nights"] <= max_night) &
        (df["instant_bookable"].isin(instant)) &
        (df["room type"].isin(type))
    ]
    st.dataframe(df_filter)

#------------------------------------------------------------------------ Smart rating ------------------------------------------------------------------------#

def corr_heatmap():
    try:
        import seaborn as sns
        sns.set_theme(style="white")
    except ImportError:
        sns = None

    numeric_cols = [
        "Construction year",
        "total price",
        "minimum nights",
        "number of reviews",
        "reviews per month",
        "review rate number",
        "calculated host listings count",
        "availability 365",
    ]

    existing_numeric = [c for c in numeric_cols if c in df.columns]
    if not existing_numeric:
        print("No numeric columns found for correlation heatmap.")
    else:
        corr_matrix = df[existing_numeric].apply(pd.to_numeric, errors="coerce").corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        if sns is not None:
            sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1)
        else:
            ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_colorbar()
            ax.set_xticks(range(len(existing_numeric)), existing_numeric, rotation=45, ha="right")
            ax.set_yticks(range(len(existing_numeric)), existing_numeric)
        ax.set_title("Correlation Heatmap")
        return (fig)
        

def _pick_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None
    
    
def fpgrowth():    
    categorical_cols = [
        "host_identity_verified",
        "neighbourhood group",
        "instant_bookable",
        "cancellation_policy",
        "room type",
    ]
    existing_cat = [c for c in categorical_cols if c in df.columns]
    transactions = []
    for _, row in df[existing_cat].astype("string").iterrows():
        items = [f"{col}={row[col]}" for col in existing_cat if pd.notna(row[col])]
        transactions.append(items)

    # min_support is absolute count here
    min_support = int(0.05 * len(transactions))
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)
    rules = pyfpgrowth.generate_association_rules(patterns, 0.6)  # min confidence

    # Keep only 1->1 pairs
    pairs = []
    for antecedent, (consequent, confidence) in rules.items():
        if len(antecedent) == 1 and len(consequent) == 1:
            pairs.append((antecedent[0], consequent[0], confidence))

    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:20]
    st.dataframe(pairs)

    
def normalization():
    normalized = {}
    numeric_cols = [
        "number of reviews",
        "review rate number",
        "total price",
    ]

    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        normalized[col] = {
            "min": values.min(skipna=True),
            "max": values.max(skipna=True),
        }
    return normalized


def normalize(value, col, stats):
    min_val = stats[col]["min"]
    max_val = stats[col]["max"]
    if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return None
    norm = (value - min_val) / (max_val - min_val)
    if col == "total price":
        return 1 - norm
    return norm


def load_decision_tree_artifacts():
    model_path = Path("decision_tree_model.pkl")
    meta_path = Path("decision_tree_meta.json")
    if not model_path.exists() or not meta_path.exists():
        st.warning(
            "Decision tree artifacts not found. "
            "Run data mining copy2.ipynb to generate them."
        )
        return (None, [], None, None, None)

    with meta_path.open("r", encoding="utf-8") as meta_file:
        meta = json.load(meta_file)

    with model_path.open("rb") as model_file:
        dt_model = pickle.load(model_file)

    normalized_cols = meta.get("normalized_cols", [])
    score_min = meta.get("score_min")
    score_max = meta.get("score_max")
    default_reviews_per_month_norm = meta.get("reviews_per_month_norm_default")

    return (
        dt_model,
        normalized_cols,
        score_min,
        score_max,
        default_reviews_per_month_norm,
    )


def decision_tree_predict():
    if dt_model is None or not normalized_cols:
        st.warning(
            "Decision tree model is unavailable. "
            "Run data mining copy2.ipynb to generate artifacts."
        )
        return None

    st.subheader("Enter Information")

    total_price = st.number_input("Total price per night", min_value=0.0)
    number_of_reviews = st.number_input("Number of reviews", min_value=0.0)
    review_rate = st.number_input(
        "Review rate number",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.01,
        format="%.2f",
    )

    run_prediction = st.button("Run Prediction")
    if not run_prediction:
        return "pending"

    input_values = {
        "number of reviews": number_of_reviews,
        "review rate number": review_rate,
        "total price": total_price,
    }

    normalized_input = {}
    for col, value in input_values.items():
        normalized_input[f"{col}_norm"] = normalize(value, col, normalized)

    if "reviews per month_norm" in normalized_cols:
        normalized_input["reviews per month_norm"] = dt_reviews_per_month_norm_default

    score_features = {
        "total price_norm": 0.4,
        "number of reviews_norm": 0.3,
        "review rate number_norm": 0.3,
    }
    score_values = {
        "total price_norm": normalized_input.get("total price_norm"),
        "number of reviews_norm": normalized_input.get("number of reviews_norm"),
        "review rate number_norm": normalized_input.get("review rate number_norm"),
    }

    weighted_sum, weight_sum = 0, 0
    for feature, weight in score_features.items():
        value = score_values.get(feature)
        if value is not None and not pd.isna(value):
            weighted_sum += value * weight
            weight_sum += weight

    if weight_sum == 0:
        st.warning("Please enter valid values to generate a prediction.")
        return None

    score = weighted_sum / weight_sum
    if "score_norm" in normalized_cols:
        if pd.isna(dt_score_min) or pd.isna(dt_score_max) or dt_score_max == dt_score_min:
            score_norm = None
        else:
            score_norm = (score - dt_score_min) / (dt_score_max - dt_score_min)
        normalized_input["score_norm"] = score_norm

    input_df = pd.DataFrame([normalized_input])[normalized_cols]
    if input_df.isna().any(axis=None):
        st.warning("Please enter valid values to generate a prediction.")
        return None

    prediction = dt_model.predict(input_df)[0]
    return prediction


def smart():
    prediction = decision_tree_predict()
    if prediction == "pending":
        st.info("Click Run Prediction to see the result.")
    elif prediction is None:
        st.info("Result: Insufficient data")
    elif prediction == "Risky":
        st.markdown("🟥 Risky")
    elif prediction == "Standard":
        st.markdown("🟨 Standard")
    else:
        st.markdown("🟩 Elite")


# ------------------------------------------------------------------------- Main page -------------------------------------------------------------------------#

fig = corr_heatmap()
normalized = normalization()
if "dt_model" not in st.session_state:
    (
        st.session_state.dt_model,
        st.session_state.dt_normalized_cols,
        st.session_state.dt_score_min,
        st.session_state.dt_score_max,
        st.session_state.dt_reviews_per_month_norm_default,
    ) = load_decision_tree_artifacts()

dt_model = st.session_state.dt_model
normalized_cols = st.session_state.dt_normalized_cols
dt_score_min = st.session_state.dt_score_min
dt_score_max = st.session_state.dt_score_max
dt_reviews_per_month_norm_default = st.session_state.dt_reviews_per_month_norm_default

st.sidebar.header("Airbnb Analytics System")
tab = st.sidebar.radio("Select role",[ 
    "Smart rating",
    "Dashboard"
])

if tab == "Smart rating":
    st.session_state.role = "smart"
    st.sidebar.header("Smart Analysis")
    smart()

elif tab == "Dashboard":
    st.session_state.role = "dashboard"
    st.sidebar.header("Dashboard")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

