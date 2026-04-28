from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

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
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "processed_crime_data.csv"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIME_ORDER = ["Night", "Morning", "Afternoon", "Evening"]
TIME_TO_CODE = {name: index for index, name in enumerate(TIME_ORDER)}
DAY_TO_CODE = {name: index for index, name in enumerate(DAY_NAMES)}
DEFAULT_WARNING = "Optional dependency missing. Please install scikit-learn or pydeck."

st.set_page_config(page_title="Delhi Crime Analysis Dashboard", layout="wide")


# -----------------------------------------------------------------------------
# Optional dependency handling
# -----------------------------------------------------------------------------
def warn_optional_dependency_once() -> None:
    if not st.session_state.get("_optional_dependency_warning_shown", False):
        st.session_state["_optional_dependency_warning_shown"] = True
        st.warning(DEFAULT_WARNING)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# -----------------------------------------------------------------------------
# Data loading and feature engineering
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def find_first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {column.lower().strip(): column for column in frame.columns}
    for candidate in candidates:
        resolved = lookup.get(candidate.lower().strip())
        if resolved is not None:
            return resolved
    return None


def normalize_day_name(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    text = str(value).strip().lower()
    aliases = {
        "mon": "Monday",
        "monday": "Monday",
        "tue": "Tuesday",
        "tues": "Tuesday",
        "tuesday": "Tuesday",
        "wed": "Wednesday",
        "weds": "Wednesday",
        "wednesday": "Wednesday",
        "thu": "Thursday",
        "thur": "Thursday",
        "thurs": "Thursday",
        "thursday": "Thursday",
        "fri": "Friday",
        "friday": "Friday",
        "sat": "Saturday",
        "saturday": "Saturday",
        "sun": "Sunday",
        "sunday": "Sunday",
    }
    return aliases.get(text)


def time_category_from_hour(hour: int) -> str:
    hour = int(hour)
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


def stable_simulated_days(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")

    hashed = pd.util.hash_pandas_object(
        frame[["Crime Description", "Latitude", "Longitude", "Hour"]].astype(str),
        index=False,
    )
    seed = int(hashed.sum() % (2**32 - 1))
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice(DAY_NAMES, size=len(frame), replace=True), index=frame.index)


@st.cache_data(show_spinner=False)
def prepare_base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    data = frame.copy()
    data["Crime Description"] = data["Crime Description"].astype(str)
    data["Hour"] = pd.to_numeric(data["Hour"], errors="coerce")
    data["Latitude"] = pd.to_numeric(data["Latitude"], errors="coerce")
    data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")
    data = data.dropna(subset=["Crime Description", "Hour", "Latitude", "Longitude"])

    if data.empty:
        return data

    data["Hour"] = np.clip(data["Hour"].astype(int), 0, 23)
    data["Time Category"] = data["Hour"].apply(time_category_from_hour)
    data["Hour Sin"] = np.sin(2 * np.pi * data["Hour"] / 24)
    data["Hour Cos"] = np.cos(2 * np.pi * data["Hour"] / 24)

    day_column = find_first_existing_column(data, ["Day of Week", "DayOfWeek", "weekday", "week day", "day_name"])
    if day_column is not None:
        normalized_days = data[day_column].apply(normalize_day_name)
        fallback_days = stable_simulated_days(data)
        data["Day of Week"] = normalized_days.fillna(fallback_days)
    else:
        data["Day of Week"] = stable_simulated_days(data)

    data["Day of Week"] = pd.Categorical(data["Day of Week"], categories=DAY_NAMES, ordered=False)
    data["Time Category"] = pd.Categorical(data["Time Category"], categories=TIME_ORDER, ordered=False)
    return data


@st.cache_data(show_spinner=False)
def attach_location_clusters(frame: pd.DataFrame) -> tuple[pd.DataFrame, Any | None, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), None, pd.DataFrame(columns=["Cluster", "Latitude", "Longitude"])

    data = frame.copy()
    data["Location Cluster"] = "0"

    if not SKLEARN_AVAILABLE:
        return data, None, pd.DataFrame(columns=["Cluster", "Latitude", "Longitude"])

    coords = data[["Latitude", "Longitude"]].dropna()
    if len(coords) < 3 or coords.drop_duplicates().shape[0] < 2:
        return data, None, pd.DataFrame(columns=["Cluster", "Latitude", "Longitude"])

    cluster_count = min(6, max(2, len(coords) // 250 + 2))
    cluster_count = min(cluster_count, len(coords))
    if cluster_count < 2:
        return data, None, pd.DataFrame(columns=["Cluster", "Latitude", "Longitude"])

    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    data.loc[coords.index, "Location Cluster"] = labels.astype(str)
    data["Location Cluster"] = data["Location Cluster"].astype(str)

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=["Latitude", "Longitude"])
    centers["Cluster"] = [str(cluster_id) for cluster_id in range(cluster_count)]
    return data, kmeans, centers


@st.cache_data(show_spinner=False)
def compute_hourly_counts(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "Hour" not in frame.columns:
        return pd.Series(0, index=range(24))
    return frame.groupby("Hour").size().reindex(range(24), fill_value=0)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def build_heatmap_pivot(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    pivot_counts = frame.pivot_table(index="Crime Description", columns="Hour", aggfunc="size", fill_value=0)
    if pivot_counts.empty:
        return pivot_counts

    sort_order = pivot_counts.sum(axis=1).sort_values(ascending=False).index
    pivot_counts = pivot_counts.loc[sort_order]
    row_max = pivot_counts.max(axis=1).replace(0, 1)
    return pivot_counts.div(row_max, axis=0)


def build_correlation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    correlation_frame = pd.DataFrame(
        {
            "Hour": pd.to_numeric(frame["Hour"], errors="coerce"),
            "Latitude": pd.to_numeric(frame["Latitude"], errors="coerce"),
            "Longitude": pd.to_numeric(frame["Longitude"], errors="coerce"),
            "Hour Sin": pd.to_numeric(frame["Hour Sin"], errors="coerce"),
            "Hour Cos": pd.to_numeric(frame["Hour Cos"], errors="coerce"),
            "Day Code": frame["Day of Week"].map(DAY_TO_CODE).astype(float),
            "Time Code": frame["Time Category"].map(TIME_TO_CODE).astype(float),
            "Cluster Code": pd.to_numeric(frame["Location Cluster"], errors="coerce"),
        }
    ).dropna()
    return correlation_frame


def infer_base_feature_name(transformed_name: str, numeric_features: list[str], categorical_features: list[str]) -> str:
    base = transformed_name.split("__", 1)[-1]

    if base in numeric_features:
        return base

    for feature in categorical_features:
        if base == feature or base.startswith(f"{feature}_"):
            return feature

    return base.split("_", 1)[0]


def build_feature_importance_frame(pipeline: Pipeline) -> pd.DataFrame:
    numeric_features = ["Hour", "Latitude", "Longitude", "Hour Sin", "Hour Cos"]
    categorical_features = ["Time Category", "Day of Week", "Location Cluster"]

    try:
        transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])

    raw_importances = pipeline.named_steps["model"].feature_importances_
    feature_rows: list[dict[str, Any]] = []
    for transformed_name, importance in zip(transformed_names, raw_importances):
        base_feature = infer_base_feature_name(str(transformed_name), numeric_features, categorical_features)
        feature_rows.append({"Feature": base_feature, "Importance": float(importance)})

    importance_frame = pd.DataFrame(feature_rows)
    if importance_frame.empty:
        return importance_frame

    importance_frame = (
        importance_frame.groupby("Feature", as_index=False)["Importance"].sum().sort_values("Importance", ascending=False)
    )
    return importance_frame


@st.cache_data(show_spinner=False)
def train_crime_model(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ready": False,
        "message": "",
        "accuracy": None,
        "pipeline": None,
        "classes": [],
        "confusion_matrix": pd.DataFrame(),
        "classification_report": pd.DataFrame(),
        "feature_importance": pd.DataFrame(columns=["Feature", "Importance"]),
        "prior_distribution": pd.Series(dtype=float),
        "training_rows": 0,
    }

    if not SKLEARN_AVAILABLE or frame.empty:
        result["message"] = "ML model unavailable because scikit-learn is missing or the dataset is empty."
        return result

    if frame["Crime Description"].nunique() < 2 or len(frame) < 20:
        result["message"] = "Not enough class diversity to train the crime classifier."
        result["prior_distribution"] = frame["Crime Description"].value_counts(normalize=True)
        return result

    feature_columns = ["Hour", "Latitude", "Longitude", "Hour Sin", "Hour Cos", "Time Category", "Day of Week", "Location Cluster"]
    training_frame = frame[feature_columns + ["Crime Description"]].dropna().copy()
    if training_frame.empty:
        result["message"] = "No usable rows were available for model training."
        return result

    target_counts = training_frame["Crime Description"].value_counts()
    stratify_target = training_frame["Crime Description"] if target_counts.min() >= 2 else None

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            training_frame[feature_columns],
            training_frame["Crime Description"].astype(str),
            test_size=0.25,
            random_state=42,
            stratify=stratify_target,
        )
    except Exception as exc:
        result["message"] = f"Train-test split failed: {exc}"
        result["prior_distribution"] = training_frame["Crime Description"].value_counts(normalize=True)
        return result

    numeric_features = ["Hour", "Latitude", "Longitude", "Hour Sin", "Hour Cos"]
    categorical_features = ["Time Category", "Day of Week", "Location Cluster"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    try:
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = float(accuracy_score(y_test, predictions))

        classes = pipeline.named_steps["model"].classes_.tolist()
        confusion = confusion_matrix(y_test, predictions, labels=classes)
        confusion_frame = pd.DataFrame(confusion, index=classes, columns=classes)

        report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        report_frame = pd.DataFrame(report_dict).T

        result.update(
            {
                "ready": True,
                "accuracy": accuracy,
                "pipeline": pipeline,
                "classes": classes,
                "confusion_matrix": confusion_frame,
                "classification_report": report_frame,
                "feature_importance": build_feature_importance_frame(pipeline),
                "prior_distribution": training_frame["Crime Description"].value_counts(normalize=True),
                "training_rows": int(len(training_frame)),
            }
        )
        return result
    except Exception as exc:
        result["message"] = f"Model training failed: {exc}"
        result["prior_distribution"] = training_frame["Crime Description"].value_counts(normalize=True)
        return result


@st.cache_data(show_spinner=False)
def build_analysis_context(frame: pd.DataFrame) -> dict[str, Any]:
    base_frame = prepare_base_frame(frame)
    clustered_frame, kmeans, cluster_centers = attach_location_clusters(base_frame)
    model_result = train_crime_model(clustered_frame)

    return {
        "base_frame": base_frame,
        "clustered_frame": clustered_frame,
        "kmeans": kmeans,
        "cluster_centers": cluster_centers,
        "model": model_result,
        "correlation_frame": build_correlation_frame(clustered_frame),
        "hourly_counts": compute_hourly_counts(clustered_frame),
        "summary_stats": compute_summary_stats(clustered_frame),
        "heatmap_pivot": build_heatmap_pivot(clustered_frame),
    }


# -----------------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------------
def build_prediction_sample(hour: int, latitude: float, longitude: float) -> pd.DataFrame:
    sample = pd.DataFrame(
        [
            {
                "Hour": int(hour),
                "Latitude": float(latitude),
                "Longitude": float(longitude),
            }
        ]
    )
    sample["Hour Sin"] = np.sin(2 * np.pi * sample["Hour"] / 24)
    sample["Hour Cos"] = np.cos(2 * np.pi * sample["Hour"] / 24)
    sample["Time Category"] = sample["Hour"].apply(time_category_from_hour)
    sample["Time Category"] = pd.Categorical(sample["Time Category"], categories=TIME_ORDER, ordered=False)
    return sample


def predict_crime_type(model_result: dict[str, Any], kmeans: Any | None, hour: int, latitude: float, longitude: float) -> tuple[str, float, pd.DataFrame]:
    if not model_result.get("ready") or model_result.get("pipeline") is None:
        raise RuntimeError("Model is not available")

    day_distribution = model_result.get("prior_distribution", pd.Series(dtype=float))
    if day_distribution.empty:
        day_distribution = pd.Series(1.0 / len(DAY_NAMES), index=DAY_NAMES)
    else:
        day_distribution = day_distribution.reindex(DAY_NAMES).fillna(0)
        if float(day_distribution.sum()) <= 0:
            day_distribution = pd.Series(1.0 / len(DAY_NAMES), index=DAY_NAMES)
        else:
            day_distribution = day_distribution / day_distribution.sum()

    sample = build_prediction_sample(hour, latitude, longitude)
    if kmeans is not None:
        cluster_label = int(kmeans.predict(sample[["Latitude", "Longitude"]])[0])
    else:
        cluster_label = 0
    sample["Location Cluster"] = str(cluster_label)

    pipeline: Pipeline = model_result["pipeline"]
    classes: list[str] = list(model_result["classes"])
    probability_vectors: list[np.ndarray] = []
    total_weight = 0.0

    for day_name, weight in day_distribution.items():
        day_sample = sample.copy()
        day_sample["Day of Week"] = pd.Categorical([day_name], categories=DAY_NAMES, ordered=False)

        try:
            probabilities = pipeline.predict_proba(day_sample)[0]
            probability_vectors.append(probabilities * float(weight))
            total_weight += float(weight)
        except Exception:
            continue

    if not probability_vectors or total_weight <= 0:
        raise RuntimeError("Prediction failed")

    averaged_probabilities = np.sum(probability_vectors, axis=0) / total_weight
    best_index = int(np.argmax(averaged_probabilities))
    prediction = classes[best_index]
    confidence = float(averaged_probabilities[best_index])

    probability_frame = pd.DataFrame({"Crime Type": classes, "Probability": averaged_probabilities}).sort_values(
        "Probability", ascending=False
    )
    return prediction, confidence, probability_frame


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def render_plotly_or_matplotlib_bar(series: pd.Series, title: str, x_title: str, y_title: str) -> None:
    if px is not None:
        frame = series.reset_index()
        frame.columns = [x_title, y_title]
        fig = px.bar(frame, x=x_title, y=y_title, title=title)
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title=x_title, yaxis_title=y_title)
        st.plotly_chart(fig, use_container_width=True)
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(series.index.astype(str), series.values, color="teal")
    ax.set_title(title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


def render_confusion_matrix(confusion_frame: pd.DataFrame) -> None:
    if confusion_frame.empty:
        st.info("Confusion matrix is unavailable.")
        return

    if go is not None:
        fig = go.Figure(
            data=go.Heatmap(
                z=confusion_frame.values,
                x=confusion_frame.columns.tolist(),
                y=confusion_frame.index.tolist(),
                colorscale="Blues",
                colorbar=dict(title="Count"),
            )
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            margin=dict(l=10, r=10, t=50, b=10),
            height=max(450, 22 * len(confusion_frame)),
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_frame, annot=False, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    st.pyplot(fig)


def render_feature_importance(importance_frame: pd.DataFrame) -> None:
    if importance_frame.empty:
        st.info("Not enough data diversity to compute feature importance.")
        return

    sorted_frame = importance_frame.sort_values("Importance", ascending=True)
    if px is not None:
        fig = px.bar(
            sorted_frame,
            x="Importance",
            y="Feature",
            orientation="h",
            title="RandomForest Feature Importance",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)
        return

    st.bar_chart(sorted_frame.set_index("Feature")["Importance"])


def render_time_type_heatmap(frame: pd.DataFrame) -> None:
    pivot = build_heatmap_pivot(frame)
    if pivot.empty:
        st.warning("No data available for selected filters")
        return

    if go is not None:
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(column) for column in pivot.columns],
                y=pivot.index.tolist(),
                colorscale="YlOrRd",
                colorbar=dict(title="Intensity"),
            )
        )
        fig.update_layout(
            title="Crime Intensity by Type and Hour",
            xaxis_title="Hour",
            yaxis_title="Crime Description",
            margin=dict(l=10, r=10, t=50, b=10),
            height=max(500, 22 * len(pivot.index)),
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    try:
        if sns is not None:
            fig, ax = plt.subplots(figsize=(16, max(6, len(pivot.index) * 0.45)))
            sns.heatmap(pivot, cmap="YlOrRd", annot=False, cbar_kws={"label": "Crime Intensity"}, ax=ax)
            ax.set_title("Crime Intensity by Type and Hour")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Crime Description")
            st.pyplot(fig)
        else:
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


def build_cluster_color_map(cluster_ids: list[str]) -> dict[str, list[int]]:
    palette = {}
    for index, cluster_id in enumerate(sorted(cluster_ids, key=lambda value: int(value))):
        palette[cluster_id] = [
            int((index * 67 + 40) % 255),
            int((index * 131 + 80) % 255),
            int((index * 193 + 160) % 255),
            210,
        ]
    return palette


def render_cluster_visualization(cluster_frame: pd.DataFrame, kmeans: Any | None, centers: pd.DataFrame) -> None:
    if cluster_frame.empty:
        st.info("Not enough data points for clustering.")
        return

    if not SKLEARN_AVAILABLE:
        warn_optional_dependency_once()
        st.info("Clustering is unavailable without scikit-learn. Showing a raw location scatter instead.")

    summary = (
        cluster_frame.groupby("Location Cluster")
        .agg(Records=("Location Cluster", "size"), AvgLatitude=("Latitude", "mean"), AvgLongitude=("Longitude", "mean"))
        .reset_index()
        .sort_values("Location Cluster")
    )

    st.subheader("Cluster Summary")
    st.dataframe(summary, use_container_width=True)
    st.caption("These clusters represent high-density crime zones.")

    if cluster_frame["Location Cluster"].nunique() <= 1 or kmeans is None:
        if px is not None:
            fig = px.scatter(
                cluster_frame,
                x="Longitude",
                y="Latitude",
                hover_data={"Crime Description": True, "Hour": True, "Longitude": False, "Latitude": False},
                title="Crime Locations",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Longitude", yaxis_title="Latitude")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.scatter_chart(cluster_frame[["Longitude", "Latitude"]])
        return

    color_map = build_cluster_color_map(cluster_frame["Location Cluster"].unique().tolist())
    visual_frame = cluster_frame.copy()
    visual_frame["Cluster Label"] = visual_frame["Location Cluster"].astype(str)
    visual_frame["Color"] = visual_frame["Cluster Label"].map(color_map)

    if pdk is not None:
        try:
            view_state = pdk.ViewState(
                latitude=float(visual_frame["Latitude"].mean()),
                longitude=float(visual_frame["Longitude"].mean()),
                zoom=10,
                pitch=35,
            )
            point_layer = pdk.Layer(
                "ScatterplotLayer",
                data=visual_frame,
                get_position="[Longitude, Latitude]",
                get_fill_color="Color",
                get_radius=150,
                pickable=True,
            )
            layers = [point_layer]

            if not centers.empty:
                center_frame = centers.copy()
                center_frame["Color"] = [[15, 15, 15, 240]] * len(center_frame)
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=center_frame,
                        get_position="[Longitude, Latitude]",
                        get_fill_color="Color",
                        get_radius=250,
                        get_line_color=[255, 255, 255, 255],
                        stroked=True,
                        pickable=True,
                    )
                )
                layers.append(
                    pdk.Layer(
                        "TextLayer",
                        data=center_frame,
                        get_position="[Longitude, Latitude]",
                        get_text="Cluster",
                        get_size=14,
                        get_color=[0, 0, 0, 220],
                        get_alignment_baseline="bottom",
                    )
                )

            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip={"text": "Cluster: {Cluster Label}\nCrime: {Crime Description}\nHour: {Hour}"},
                )
            )
            return
        except Exception:
            pass

    if px is not None:
        fig = px.scatter(
            visual_frame,
            x="Longitude",
            y="Latitude",
            color="Cluster Label",
            hover_data={"Crime Description": True, "Hour": True, "Latitude": False, "Longitude": False},
            title="Crime Clusters by Latitude and Longitude",
            labels={"Cluster Label": "Cluster"},
        )
        if not centers.empty:
            fig.add_scatter(
                x=centers["Longitude"],
                y=centers["Latitude"],
                mode="markers+text",
                marker=dict(symbol="x", size=16, color="black"),
                text=centers["Cluster"],
                textposition="top center",
                name="Cluster Centers",
                hovertemplate="Cluster Center %{text}<extra></extra>",
            )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Longitude", yaxis_title="Latitude")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.scatter_chart(visual_frame[["Longitude", "Latitude"]])


def render_density_heatmap(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No data available for heatmap generation.")
        return

    density_frame = frame[["Latitude", "Longitude"]].copy()
    density_frame["Crime Weight"] = 1

    if pdk is not None:
        try:
            view_state = pdk.ViewState(
                latitude=float(density_frame["Latitude"].mean()),
                longitude=float(density_frame["Longitude"].mean()),
                zoom=10,
                pitch=25,
            )
            heat_layer = pdk.Layer(
                "HeatmapLayer",
                data=density_frame,
                get_position="[Longitude, Latitude]",
                get_weight="Crime Weight",
                radius_pixels=60,
            )
            st.pydeck_chart(pdk.Deck(layers=[heat_layer], initial_view_state=view_state))
            return
        except Exception:
            pass

    if px is not None:
        fig = px.density_heatmap(
            density_frame,
            x="Longitude",
            y="Latitude",
            nbinsx=30,
            nbinsy=30,
            color_continuous_scale="YlOrRd",
            title="Crime Density Heatmap",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Longitude", yaxis_title="Latitude")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist2d(density_frame["Longitude"], density_frame["Latitude"], bins=30, cmap="YlOrRd")
        ax.set_title("Crime Density Heatmap")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        st.pyplot(fig)


def build_map_data(frame: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    map_frame = frame[["Crime Description", "Latitude", "Longitude", "Hour"]].dropna().copy()
    if map_frame.empty:
        return map_frame, []

    if mode == "By Hour Intensity":
        map_frame["Intensity"] = pd.cut(
            map_frame["Hour"],
            bins=[-1, 8, 16, 23],
            labels=["Low", "Medium", "High"],
        ).astype(str)
        color_map = {
            "Low": [173, 216, 230, 200],
            "Medium": [255, 165, 0, 210],
            "High": [220, 20, 60, 220],
        }
        map_frame["Color"] = map_frame["Intensity"].map(color_map)
        legend = [("Low (0-8)", "rgb(173,216,230)"), ("Medium (9-16)", "rgb(255,165,0)"), ("High (17-23)", "rgb(220,20,60)")]
    elif mode == "By Crime Type":
        unique_crimes = sorted(map_frame["Crime Description"].unique().tolist())
        palette = {}
        base_palette = [
            [31, 119, 180, 200],
            [255, 127, 14, 200],
            [44, 160, 44, 200],
            [214, 39, 40, 200],
            [148, 103, 189, 200],
            [140, 86, 75, 200],
            [227, 119, 194, 200],
            [127, 127, 127, 200],
            [188, 189, 34, 200],
            [23, 190, 207, 200],
        ]
        for index, crime in enumerate(unique_crimes):
            palette[crime] = base_palette[index % len(base_palette)]
        map_frame["Color"] = map_frame["Crime Description"].map(palette)
        legend = [(crime, f"rgb({color[0]},{color[1]},{color[2]})") for crime, color in list(palette.items())[:12]]
    else:
        map_frame["Color"] = [[40, 120, 240, 190]] * len(map_frame)
        legend = [("All Crimes", "rgb(40,120,240)")]

    return map_frame, legend


def render_map_section(frame: pd.DataFrame) -> None:
    st.subheader("Crime Map")

    mode = st.radio("Map View Mode", ["All Crimes", "By Hour Intensity", "By Crime Type"], horizontal=True)
    map_frame, legend_items = build_map_data(frame, mode)
    if map_frame.empty:
        st.warning("No data available for selected filters")
        return

    if pdk is not None:
        try:
            view_state = pdk.ViewState(
                latitude=float(map_frame["Latitude"].mean()),
                longitude=float(map_frame["Longitude"].mean()),
                zoom=10,
                pitch=30,
            )
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_frame,
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
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"text": tooltip_text},
                )
            )
        except Exception:
            if px is not None:
                color_series = map_frame["Crime Description"] if mode == "By Crime Type" else map_frame["Hour"].astype(str)
                fig = px.scatter(
                    map_frame,
                    x="Longitude",
                    y="Latitude",
                    color=color_series,
                    hover_data={"Crime Description": True, "Hour": True, "Latitude": False, "Longitude": False},
                    title="Crime Map",
                )
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Longitude", yaxis_title="Latitude")
                st.plotly_chart(fig, use_container_width=True)
    elif px is not None:
        color_series = map_frame["Crime Description"] if mode == "By Crime Type" else map_frame["Hour"].astype(str)
        fig = px.scatter(
            map_frame,
            x="Longitude",
            y="Latitude",
            color=color_series,
            hover_data={"Crime Description": True, "Hour": True, "Latitude": False, "Longitude": False},
            title="Crime Map",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Longitude", yaxis_title="Latitude")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.scatter_chart(map_frame[["Longitude", "Latitude"]])

    st.markdown("### Legend")
    for label, color in legend_items:
        st.markdown(
            f"<span style='display:inline-block;width:12px;height:12px;background:{color};margin-right:8px;'></span>{label}",
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def render_fallback_summary(frame: pd.DataFrame) -> None:
    st.info("Model training is unavailable. Showing descriptive summaries instead.")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.metric("Top Crime Type", frame["Crime Description"].value_counts().idxmax())
    with summary_col2:
        st.metric("Top Crime Share", f"{frame['Crime Description'].value_counts(normalize=True).max():.1%}")


def render_correlation_section(frame: pd.DataFrame, correlation_frame: pd.DataFrame) -> None:
    st.subheader("Correlation Insights")
    st.caption("The heatmap highlights how engineered time and location features move together across the filtered dataset.")

    if correlation_frame.empty:
        st.info("Correlation analysis is unavailable for the current filter selection.")
        return

    corr_matrix = correlation_frame.corr(numeric_only=True)
    if corr_matrix.empty:
        st.info("Correlation matrix could not be computed.")
        return

    upper_mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    upper_pairs = corr_matrix.where(upper_mask).stack()
    strongest_pairs = upper_pairs.abs().sort_values(ascending=False).head(5)

    pair_summary = pd.DataFrame(
        {
            "Feature Pair": [f"{left} vs {right}" for left, right in strongest_pairs.index],
            "Correlation": [float(upper_pairs.loc[pair]) for pair in strongest_pairs.index],
        }
    )

    if go is not None:
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                zmid=0,
                colorscale="RdBu",
                colorbar=dict(title="Correlation"),
            )
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), title="Correlation Matrix", height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, cmap="RdBu", center=0, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    strongest_label = pair_summary.iloc[0]["Feature Pair"] if not pair_summary.empty else "N/A"
    strongest_value = pair_summary.iloc[0]["Correlation"] if not pair_summary.empty else 0.0

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Strongest Relationship", strongest_label)
    metric_col2.metric("Correlation Value", f"{strongest_value:.3f}")

    st.dataframe(pair_summary, use_container_width=True)
    st.caption("Use these relationships as directional signals, not causal evidence.")


def render_ml_section(frame: pd.DataFrame, model_result: dict[str, Any]) -> None:
    st.subheader("ML Insights")
    st.caption("The classifier uses engineered time context, simulated day-of-week, and location clusters instead of raw coordinates alone.")

    if not SKLEARN_AVAILABLE:
        warn_optional_dependency_once()
        render_fallback_summary(frame)
        return

    if not model_result.get("ready"):
        if model_result.get("message"):
            st.info(model_result["message"])
        render_fallback_summary(frame)
        return

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Accuracy", f"{model_result['accuracy']:.3f}")
    metric_col2.metric("Training Rows", f"{model_result['training_rows']:,}")
    metric_col3.metric("Classes", f"{len(model_result['classes'])}")

    st.info("The model learns when and where crime patterns repeat. Location clusters and time categories usually provide more signal than raw latitude and longitude alone.")

    render_feature_importance(model_result["feature_importance"])

    st.subheader("Confusion Matrix")
    render_confusion_matrix(model_result["confusion_matrix"])

    with st.expander("Classification Report"):
        st.dataframe(model_result["classification_report"].round(3), use_container_width=True)


def render_prediction_section(model_result: dict[str, Any], kmeans: Any | None) -> None:
    st.subheader("Crime Prediction")
    st.caption("Enter the time and coordinates to estimate the most likely crime type and confidence score.")

    with st.form("crime_prediction_form"):
        input_col1, input_col2, input_col3 = st.columns(3)
        hour = input_col1.slider("Hour", 0, 23, 12)
        latitude = input_col2.number_input("Latitude", value=28.6139, format="%.6f")
        longitude = input_col3.number_input("Longitude", value=77.2090, format="%.6f")
        submitted = st.form_submit_button("Predict Crime Type")

    if not submitted:
        return

    if not SKLEARN_AVAILABLE or not model_result.get("ready"):
        prior = model_result.get("prior_distribution", pd.Series(dtype=float))
        if prior.empty:
            st.warning("Prediction is unavailable because the ML model could not be trained.")
            return

        fallback_class = prior.idxmax()
        fallback_confidence = float(prior.max())
        st.warning("Prediction is using baseline class frequency because the model is unavailable.")
        st.metric("Predicted Crime Type", str(fallback_class))
        st.metric("Confidence Score", f"{fallback_confidence:.1%}")
        return

    try:
        prediction, confidence, probability_frame = predict_crime_type(model_result, kmeans, hour, latitude, longitude)
        display_col1, display_col2 = st.columns(2)
        display_col1.metric("Predicted Crime Type", prediction)
        display_col2.metric("Confidence Score", f"{confidence:.1%}")

        st.subheader("Top Prediction Probabilities")
        top_probabilities = probability_frame.head(10)
        if px is not None:
            fig = px.bar(
                top_probabilities.sort_values("Probability", ascending=True),
                x="Probability",
                y="Crime Type",
                orientation="h",
                title="Top Predicted Crime Types",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Probability", yaxis_title="Crime Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(top_probabilities, use_container_width=True)
    except Exception as exc:
        st.warning(f"Prediction failed: {exc}")


# -----------------------------------------------------------------------------
# App bootstrap
# -----------------------------------------------------------------------------
try:
    raw_df = load_data()
except Exception as exc:
    st.error(f"Could not load processed_crime_data.csv: {exc}")
    st.stop()

required_columns = ["Crime Description", "Latitude", "Longitude", "Hour"]
missing_columns = [column for column in required_columns if column not in raw_df.columns]
if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

if raw_df.empty:
    st.warning("No data available for selected filters")
    st.stop()

st.title("🚔 Delhi Crime Analysis Dashboard")
st.info("Explore crime patterns across time, space, clustering, and model-driven predictions.")

st.sidebar.header("Dashboard Controls")
crime_options = sorted(raw_df["Crime Description"].astype(str).unique().tolist())
selected_crimes = st.sidebar.multiselect("Crime Description", options=crime_options, default=crime_options)
selected_hours = st.sidebar.slider("Hour range", min_value=0, max_value=23, value=(0, 23))
search_text = st.sidebar.text_input("Search crime name", placeholder="Type to filter crime names")
sort_mode = st.sidebar.selectbox(
    "Sort filtered table",
    ["Hour Ascending", "Hour Descending", "Crime A-Z", "Crime Frequency Descending"],
)
selected_columns = st.sidebar.multiselect("Columns to display", options=raw_df.columns.tolist(), default=raw_df.columns.tolist())
show_raw_data = st.sidebar.toggle("Show raw data", value=False)
row_limit = st.sidebar.slider("Rows to show", min_value=5, max_value=200, value=25, step=5)


def apply_sort(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    data = frame.copy()
    if mode == "Hour Ascending":
        return data.sort_values(["Hour", "Crime Description"], ascending=[True, True])
    if mode == "Hour Descending":
        return data.sort_values(["Hour", "Crime Description"], ascending=[False, True])
    if mode == "Crime A-Z":
        return data.sort_values(["Crime Description", "Hour"], ascending=[True, True])

    data["Crime Frequency"] = data.groupby("Crime Description")["Crime Description"].transform("size")
    data = data.sort_values(["Crime Frequency", "Hour"], ascending=[False, True])
    return data.drop(columns=["Crime Frequency"])


filtered_df = raw_df.copy()
filtered_df["Crime Description"] = filtered_df["Crime Description"].astype(str)
filtered_df["Hour"] = pd.to_numeric(filtered_df["Hour"], errors="coerce")
filtered_df["Latitude"] = pd.to_numeric(filtered_df["Latitude"], errors="coerce")
filtered_df["Longitude"] = pd.to_numeric(filtered_df["Longitude"], errors="coerce")
filtered_df = filtered_df.dropna(subset=["Crime Description", "Hour", "Latitude", "Longitude"])
filtered_df["Hour"] = np.clip(filtered_df["Hour"].astype(int), 0, 23)

if selected_crimes:
    filtered_df = filtered_df[filtered_df["Crime Description"].isin(selected_crimes)]
else:
    filtered_df = filtered_df.iloc[0:0]

filtered_df = filtered_df[filtered_df["Hour"].between(selected_hours[0], selected_hours[1])]
if search_text.strip():
    filtered_df = filtered_df[filtered_df["Crime Description"].str.contains(search_text.strip(), case=False, na=False)]

filtered_df = apply_sort(filtered_df, sort_mode)

if filtered_df.empty:
    st.warning("No data available for selected filters")
    st.stop()

if not selected_columns:
    selected_columns = raw_df.columns.tolist()

analysis_context = build_analysis_context(filtered_df)
base_frame = analysis_context["base_frame"]
clustered_frame = analysis_context["clustered_frame"]
kmeans = analysis_context["kmeans"]
cluster_centers = analysis_context["cluster_centers"]
model_result = analysis_context["model"]
correlation_frame = analysis_context["correlation_frame"]
hourly_counts = analysis_context["hourly_counts"]
summary_stats = analysis_context["summary_stats"]

tab_overview, tab_visual, tab_advanced, tab_prediction, tab_map = st.tabs(
    ["Overview", "Visual Analytics", "Advanced Insights", "Crime Prediction", "Map & Heatmap"]
)


with tab_overview:
    st.subheader("Overview")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Crimes", f"{len(filtered_df):,}")
    metric_col2.metric("Unique Crime Types", f"{filtered_df['Crime Description'].nunique():,}")
    metric_col3.metric("Peak Crime Hour", int(hourly_counts.idxmax()))

    st.divider()
    st.subheader("Data Explorer")
    st.dataframe(filtered_df[selected_columns].head(row_limit), use_container_width=True)

    if show_raw_data:
        st.subheader("Raw Processed Data")
        st.dataframe(raw_df[selected_columns].head(row_limit), use_container_width=True)

    overview_col1, overview_col2 = st.columns(2)
    with overview_col1:
        st.subheader("Top 5 Crime Types")
        top5 = filtered_df["Crime Description"].value_counts().head(5)
        if px is not None:
            top5_df = top5.rename_axis("Crime Description").reset_index(name="Count")
            fig = px.bar(top5_df, x="Crime Description", y="Count", title="Top 5 Crime Types")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Crime Description", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(top5)

    with overview_col2:
        st.subheader("Filtered Dataset Snapshot")
        st.dataframe(filtered_df[selected_columns].head(row_limit), use_container_width=True)

    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "Download filtered CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_crime_data.csv",
        mime="text/csv",
    )
    download_col2.download_button(
        "Download summary stats",
        data=summary_stats.to_csv(index=False).encode("utf-8"),
        file_name="crime_summary_stats.csv",
        mime="text/csv",
    )


with tab_visual:
    st.subheader("Visual Analytics")

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Top 10 Crime Types")
        top_crimes = filtered_df["Crime Description"].value_counts().head(10)
        render_plotly_or_matplotlib_bar(top_crimes, "Top 10 Crime Types", "Crime Description", "Count")

        st.subheader("Crime Distribution by Type")
        try:
            pie_counts = filtered_df["Crime Description"].value_counts().head(10)
            if px is not None:
                fig = px.pie(values=pie_counts.values, names=pie_counts.index, title="Crime Distribution by Type", hole=0.35)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.pie(pie_counts, labels=pie_counts.index, autopct="%1.1f%%", startangle=90)
                ax.set_title("Crime Distribution by Type")
                st.pyplot(fig)
        except Exception as exc:
            st.error(f"Pie chart rendering failed: {exc}")

    with right_col:
        st.subheader("Crime Count by Hour")
        trend_frame = pd.DataFrame(
            {
                "Crime Count": hourly_counts.values,
                "Rolling Avg (3h)": hourly_counts.rolling(window=3, min_periods=1).mean().values,
            },
            index=hourly_counts.index,
        )
        if px is not None:
            trend_plot = trend_frame.reset_index().rename(columns={"index": "Hour"})
            fig = px.line(trend_plot, x="Hour", y=["Crime Count", "Rolling Avg (3h)"], markers=True, title="Crime Count by Hour")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Hour", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(trend_frame)

        st.subheader("Crime Frequency Across Hours")
        if px is not None:
            hist_frame = filtered_df[["Hour"]].copy()
            fig = px.histogram(hist_frame, x="Hour", nbins=24, title="Crime Frequency Across Hours")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Hour", yaxis_title="Crime Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(filtered_df["Hour"], bins=range(0, 25), edgecolor="black", align="left")
            ax.set_xticks(range(0, 24))
            ax.set_xlabel("Hour")
            ax.set_ylabel("Crime Count")
            ax.set_title("Crime Frequency Across Hours")
            st.pyplot(fig)

    st.subheader("Crime Description vs Hour Heatmap")
    render_time_type_heatmap(filtered_df)


with tab_advanced:
    st.subheader("Advanced Insights")

    advanced_col1, advanced_col2, advanced_col3 = st.columns(3)
    advanced_col1.metric("Busiest Crime Hour", int(hourly_counts.idxmax()))
    advanced_col2.metric("Least Crime Hour", int(hourly_counts.idxmin()))
    advanced_col3.metric("Observation Count", f"{len(filtered_df):,}")

    st.divider()
    render_correlation_section(clustered_frame, correlation_frame)

    st.divider()
    st.subheader("Crime Clusters")
    render_cluster_visualization(clustered_frame, kmeans, cluster_centers)

    st.divider()
    render_ml_section(clustered_frame, model_result)


with tab_prediction:
    render_prediction_section(model_result, kmeans)


with tab_map:
    st.subheader("Map & Heatmap")
    render_map_section(filtered_df)

    st.divider()
    st.subheader("Crime Density Heatmap")
    st.caption("Darker regions indicate higher concentration of incident locations.")
    render_density_heatmap(filtered_df)
