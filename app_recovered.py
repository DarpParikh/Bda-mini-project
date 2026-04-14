import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import pydeck as pdk
except Exception:
    pdk = None

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


st.set_page_config(page_title="Delhi Crime Analysis Dashboard", layout="wide")


# ----------------------------
# Cached data loading and compute helpers
# ----------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("processed_crime_data.csv")


@st.cache_data
def build_heatmap_pivot(frame: pd.DataFrame) -> pd.DataFrame:
    pivot_counts = frame.pivot_table(
        index="Crime Description",
        columns="Hour",
        aggfunc="size",
        fill_value=0,
    )
    if pivot_counts.empty:
        return pivot_counts

    # Sort by total crime frequency before normalization.
    sort_order = pivot_counts.sum(axis=1).sort_values(ascending=False).index
    pivot_counts = pivot_counts.loc[sort_order]

    # Normalize row-wise for clearer pattern comparison.
    row_max = pivot_counts.max(axis=1).replace(0, 1)
    pivot_normalized = pivot_counts.div(row_max, axis=0)
    return pivot_normalized


def render_heatmap(frame: pd.DataFrame) -> None:
    pivot = build_heatmap_pivot(frame)
    if pivot.empty:
        st.warning("No data available for selected filters")
        return

    try:
        if sns is not None:
            fig, ax = plt.subplots(figsize=(16, max(6, len(pivot.index) * 0.45)))
            sns.heatmap(
                pivot,
                cmap="YlOrRd",
                annot=False,
                cbar_kws={"label": "Crime Intensity"},
                ax=ax,
            )
            ax.set_title("Crime Intensity by Type and Hour")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Crime Description")
            st.pyplot(fig)
        else:
            # Safe fallback when seaborn is unavailable.
            fig, ax = plt.subplots(figsize=(16, max(6, len(pivot.index) * 0.45)))
            image = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title("Crime Intensity by Type and Hour")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Crime Description")
            fig.colorbar(image, ax=ax, label="Crime Intensity")
            st.pyplot(fig)
    except Exception as err:
        st.error(f"Heatmap rendering failed: {err}")


@st.cache_data
def compute_hourly_counts(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("Hour").size().reindex(range(24), fill_value=0)


@st.cache_data
def compute_summary_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame({"Metric": ["Total Crimes"], "Value": [0]})

    hour_counts = compute_hourly_counts(frame)
    return pd.DataFrame(
        {
            "Metric": ["Total Crimes", "Unique Crime Types", "Peak Crime Hour", "Least Crime Hour"],
            "Value": [
                int(len(frame)),
                int(frame["Crime Description"].nunique()),
                int(hour_counts.idxmax()),
                int(hour_counts.idxmin()),
            ],
        }
    )


@st.cache_data
def compute_clusters(frame: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    if (not SKLEARN_AVAILABLE) or frame.empty or n_clusters < 2:
        return pd.DataFrame()

    cluster_df = frame[["Latitude", "Longitude", "Crime Description", "Hour"]].dropna().copy()
    if len(cluster_df) < n_clusters:
        return pd.DataFrame()

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_df["Cluster"] = model.fit_predict(cluster_df[["Latitude", "Longitude"]])
    return cluster_df


@st.cache_data
def compute_ml_insights(frame: pd.DataFrame) -> tuple[float | None, pd.DataFrame]:
    if (not SKLEARN_AVAILABLE) or frame.empty:
        return None, pd.DataFrame(columns=["Feature", "Importance"])

    ml_df = frame[["Latitude", "Longitude", "Hour", "Crime Description"]].dropna().copy()
    if len(ml_df) < 20 or ml_df["Crime Description"].nunique() < 2:
        return None, pd.DataFrame(columns=["Feature", "Importance"])

    x = ml_df[["Latitude", "Longitude", "Hour"]]
    y = LabelEncoder().fit_transform(ml_df["Crime Description"].astype(str))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    importance_df = pd.DataFrame(
        {
            "Feature": ["Latitude", "Longitude", "Hour"],
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return accuracy, importance_df


# ----------------------------
# Validation and utility helpers
# ----------------------------
def validate_columns(frame: pd.DataFrame) -> bool:
    required_columns = ["Crime Description", "Latitude", "Longitude", "Hour"]
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        st.warning(f"Missing required columns: {missing}")
        return False
    return True


def apply_sort(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy()
    if mode == "Hour Ascending":
        return out.sort_values(["Hour", "Crime Description"], ascending=[True, True])
    if mode == "Hour Descending":
        return out.sort_values(["Hour", "Crime Description"], ascending=[False, True])
    if mode == "Crime A-Z":
        return out.sort_values(["Crime Description", "Hour"], ascending=[True, True])

    out["Crime Frequency"] = out.groupby("Crime Description")["Crime Description"].transform("size")
    out = out.sort_values(["Crime Frequency", "Hour"], ascending=[False, True])
    return out.drop(columns=["Crime Frequency"])


def build_crime_palette(crime_values: list[str]) -> dict[str, list[int]]:
    if not crime_values:
        return {}

    if sns is not None:
        colors = sns.color_palette("tab20", len(crime_values))
        return {
            crime: [int(r * 255), int(g * 255), int(b * 255), 190]
            for crime, (r, g, b) in zip(crime_values, colors)
        }

    # Safe deterministic fallback without seaborn.
    palette = {}
    for idx, crime in enumerate(crime_values):
        palette[crime] = [int((idx * 53) % 255), int((idx * 97) % 255), int((idx * 149) % 255), 190]
    return palette


def build_map_data(frame: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    map_df = frame[["Crime Description", "Latitude", "Longitude", "Hour"]].dropna().copy()
    if map_df.empty:
        return map_df, []

    if mode == "By Hour Intensity":
        map_df["Intensity"] = pd.cut(
            map_df["Hour"],
            bins=[-1, 8, 16, 23],
            labels=["Low", "Medium", "High"],
        ).astype(str)

        color_map = {
            "Low": [173, 216, 230, 200],    # Light blue
            "Medium": [255, 165, 0, 210],   # Orange
            "High": [220, 20, 60, 220],     # Red
        }
        map_df["Color"] = map_df["Intensity"].map(color_map)
        legend = [
            ("Low (0-8)", "rgb(173,216,230)"),
            ("Medium (9-16)", "rgb(255,165,0)"),
            ("High (17-23)", "rgb(220,20,60)"),
        ]
    elif mode == "By Crime Type":
        crime_values = sorted(map_df["Crime Description"].unique().tolist())
        palette = build_crime_palette(crime_values)
        map_df["Color"] = map_df["Crime Description"].map(palette)
        legend = [
            (name, f"rgb({color[0]},{color[1]},{color[2]})")
            for name, color in list(palette.items())[:12]
        ]
    else:
        map_df["Color"] = [[40, 120, 240, 190]] * len(map_df)
        legend = [("All Crimes", "rgb(40,120,240)")]

    return map_df, legend


# ----------------------------
# Data load and preprocessing
# ----------------------------
try:
    df = load_data()
except Exception as err:
    st.error(f"Could not load processed_crime_data.csv: {err}")
    st.stop()

if not validate_columns(df):
    st.stop()

if df.empty:
    st.warning("No data available for selected filters")
    st.stop()

# Normalize core datatypes for safe plotting and filtering.
df = df.copy()
df["Crime Description"] = df["Crime Description"].astype(str)
df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df = df.dropna(subset=["Crime Description", "Hour", "Latitude", "Longitude"])

if df.empty:
    st.warning("No data available for selected filters")
    st.stop()

df["Hour"] = np.clip(df["Hour"].astype(int), 0, 23)


# ----------------------------
# Header + sidebar controls
# ----------------------------
st.title("🚔 Delhi Crime Analysis Dashboard")
st.info("Advanced Big Data Analytics dashboard for exploring crime patterns across type, time, and location.")

st.sidebar.header("Dashboard Controls")

crime_options = sorted(df["Crime Description"].unique().tolist())
selected_crimes = st.sidebar.multiselect("Crime Description", options=crime_options, default=crime_options)
selected_hours = st.sidebar.slider("Hour range", min_value=0, max_value=23, value=(0, 23))
search_text = st.sidebar.text_input("Search crime name", placeholder="Type to filter crime names")
sort_mode = st.sidebar.selectbox(
    "Sort filtered table",
    ["Hour Ascending", "Hour Descending", "Crime A-Z", "Crime Frequency Descending"],
)
selected_columns = st.sidebar.multiselect(
    "Columns to display",
    options=df.columns.tolist(),
    default=df.columns.tolist(),
)
show_raw_data = st.sidebar.toggle("Show raw data", value=False)
row_limit = st.sidebar.slider("Rows to show", min_value=5, max_value=200, value=25, step=5)


# ----------------------------
# Filtering
# ----------------------------
filtered_df = df.copy()
if selected_crimes:
    filtered_df = filtered_df[filtered_df["Crime Description"].isin(selected_crimes)]
else:
    filtered_df = filtered_df.iloc[0:0]

filtered_df = filtered_df[filtered_df["Hour"].between(selected_hours[0], selected_hours[1])]

if search_text.strip():
    filtered_df = filtered_df[
        filtered_df["Crime Description"].str.contains(search_text.strip(), case=False, na=False)
    ]

filtered_df = apply_sort(filtered_df, sort_mode)

if filtered_df.empty:
    st.warning("No data available for selected filters")
    st.stop()

if not selected_columns:
    selected_columns = df.columns.tolist()

summary_stats = compute_summary_stats(filtered_df)


# ----------------------------
# Tabs layout
# ----------------------------
tab_overview, tab_visual, tab_advanced, tab_map = st.tabs(
    ["Overview", "Visual Analytics", "Advanced Insights", "Map"]
)


with tab_overview:
    st.header("📌 Overview")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Crimes", f"{len(filtered_df):,}")
    m2.metric("Unique Crime Types", f"{filtered_df['Crime Description'].nunique():,}")
    m3.metric("Peak Crime Hour", int(filtered_df["Hour"].value_counts().idxmax()))

    st.subheader("📄 Data Explorer")
    st.dataframe(filtered_df[selected_columns].head(row_limit), width="stretch")

    if show_raw_data:
        st.markdown("### Raw Processed Data")
        st.dataframe(df[selected_columns].head(row_limit), width="stretch")

    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Top 5 Crime Types")
        top5 = filtered_df["Crime Description"].value_counts().head(5).reset_index()
        top5.columns = ["Crime Description", "Count"]
        st.dataframe(top5, width="stretch")

    with t2:
        st.subheader("Filtered Dataset")
        st.dataframe(filtered_df[selected_columns].head(row_limit), width="stretch")

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download filtered CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_crime_data.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Download summary stats",
        data=summary_stats.to_csv(index=False).encode("utf-8"),
        file_name="crime_summary_stats.csv",
        mime="text/csv",
    )


with tab_visual:
    st.header("📊 Visual Analytics")
    st.info("Darker colors indicate higher crime frequency at a given hour")

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Top 10 Crime Types")
        try:
            top_crimes = filtered_df["Crime Description"].value_counts().head(10)
            st.bar_chart(top_crimes)
        except Exception as err:
            st.error(f"Bar chart rendering failed: {err}")

        st.subheader("Crime Distribution by Type")
        try:
            pie_counts = filtered_df["Crime Description"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(pie_counts, labels=pie_counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title("Crime Distribution by Type")
            st.pyplot(fig)
        except Exception as err:
            st.error(f"Pie chart rendering failed: {err}")

    with right_col:
        st.subheader("Crime Count by Hour")
        try:
            hour_counts = compute_hourly_counts(filtered_df)
            trend_frame = pd.DataFrame(
                {
                    "Crime Count": hour_counts.values,
                    "Rolling Avg (3h)": hour_counts.rolling(window=3, min_periods=1).mean().values,
                },
                index=hour_counts.index,
            )
            st.line_chart(trend_frame)
        except Exception as err:
            st.error(f"Line chart rendering failed: {err}")

        st.subheader("Crime Frequency Across Hours")
        try:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(filtered_df["Hour"], bins=range(0, 25), edgecolor="black", align="left")
            ax.set_xticks(range(0, 24))
            ax.set_xlabel("Hour")
            ax.set_ylabel("Crime Count")
            ax.set_title("Crime Frequency Across Hours")
            st.pyplot(fig)
        except Exception as err:
            st.error(f"Histogram rendering failed: {err}")

    st.subheader("Crime Description vs Hour Heatmap")
    render_heatmap(filtered_df)


with tab_advanced:
    st.header("🧠 Advanced Insights")

    hour_counts = compute_hourly_counts(filtered_df)
    busiest_hour = int(hour_counts.idxmax())
    least_hour = int(hour_counts.idxmin())

    p1, p2 = st.columns(2)
    p1.metric("Busiest Crime Hour", busiest_hour)
    p2.metric("Least Crime Hour", least_hour)

    st.subheader("Crime Trend Analysis")
    trend_df = pd.DataFrame(
        {
            "Crime Count": hour_counts.values,
            "Rolling Avg (3h)": hour_counts.rolling(window=3, min_periods=1).mean().values,
        },
        index=hour_counts.index,
    )
    st.line_chart(trend_df)

    st.subheader("Crime Probability Analysis")
    total = hour_counts.sum()
    prob = hour_counts / total if total > 0 else hour_counts
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(prob.index, prob.values, color="teal")
    ax.set_title("Crime Probability by Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(0, 24))
    st.pyplot(fig)

    st.subheader("Correlation Insights")
    corr_df = pd.DataFrame({"Hour": hour_counts.index, "Crime Count": hour_counts.values})
    corr_value = corr_df["Hour"].corr(corr_df["Crime Count"])
    if pd.isna(corr_value):
        st.info("Correlation is undefined due to low variation.")
    else:
        st.info(f"Correlation between hour and crime count: {corr_value:.3f}")

    st.subheader("Top Crime Locations (Simulated clustering)")
    if not SKLEARN_AVAILABLE or pdk is None:
        st.info("Install scikit-learn and pydeck to view clustering hotspots.")
    else:
        clusters = min(6, max(2, len(filtered_df) // 250 + 2))
        cluster_df = compute_clusters(filtered_df, clusters)
        if cluster_df.empty:
            st.info("Not enough data points for clustering.")
        else:
            cluster_palette = build_crime_palette([f"Cluster {c}" for c in sorted(cluster_df["Cluster"].unique())])
            cluster_df["Color"] = cluster_df["Cluster"].apply(
                lambda c: cluster_palette.get(f"Cluster {c}", [90, 140, 240, 190])
            )

            view_state = pdk.ViewState(
                latitude=28.6139,
                longitude=77.2090,
                zoom=10,
                pitch=35,
            )
            cluster_layer = pdk.Layer(
                "ScatterplotLayer",
                data=cluster_df,
                get_position="[Longitude, Latitude]",
                get_fill_color="Color",
                get_radius=170,
                pickable=True,
            )
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v10",
                    initial_view_state=view_state,
                    layers=[cluster_layer],
                    tooltip={"text": "Cluster: {Cluster}\nCrime: {Crime Description}\nHour: {Hour}"},
                )
            )

    st.subheader("ML Insights")
    if not SKLEARN_AVAILABLE:
        st.info("Install scikit-learn to enable ML insights.")
    else:
        accuracy, importance_df = compute_ml_insights(filtered_df)
        if accuracy is None or importance_df.empty:
            st.info("Not enough data diversity to train RandomForest.")
        else:
            ml_col1, ml_col2 = st.columns([1, 2])
            ml_col1.metric("RandomForest Accuracy", f"{accuracy:.3f}")
            ml_col2.bar_chart(importance_df.set_index("Feature"))


with tab_map:
    st.header("🗺️ Smart Crime Map")

    if pdk is None:
        st.warning("pydeck is required for map rendering.")
    else:
        mode = st.radio(
            "Map View Mode",
            ["All Crimes", "By Hour Intensity", "By Crime Type"],
            horizontal=True,
        )

        map_df, legend_items = build_map_data(filtered_df, mode)
        if map_df.empty:
            st.warning("No data available for selected filters")
        else:
            st.info("Interactive map with color-encoded crime patterns and hover tooltips.")

            view_state = pdk.ViewState(
                latitude=28.6139,
                longitude=77.2090,
                zoom=10,
                pitch=30,
            )
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[Longitude, Latitude]",
                get_fill_color="Color",
                get_radius=140,
                pickable=True,
            )

            tooltip_text = "Crime: {Crime Description}\nHour: {Hour}"
            if mode == "By Hour Intensity":
                tooltip_text = "Crime: {Crime Description}\nHour: {Hour}\nIntensity: {Intensity}"

            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v10",
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"text": tooltip_text},
                )
            )

            st.markdown("### Legend")
            for label, color in legend_items:
                st.markdown(
                    f"<span style='display:inline-block;width:12px;height:12px;background:{color};margin-right:8px;'></span>{label}",
                    unsafe_allow_html=True,
                )
