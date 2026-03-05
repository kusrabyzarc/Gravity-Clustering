import streamlit as st
import numpy as np
import time

# sklearn
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# core modules
from core import grid, density, modes, merge, clustering
# viz module
from viz import plot
# data generator
from data import generators


st.set_page_config(page_title="Density Clustering Interactive", layout="wide")
st.title("Интерактивная плотностная кластеризация")


# --------------------------
# Sidebar: параметры Gravity
# --------------------------
st.sidebar.markdown("## Gravity параметры")

SIGMA = st.sidebar.slider("SIGMA (ширина ядра)", 0.1, 3.0, 0.8, 0.05)
GRID_SIZE = st.sidebar.slider("GRID_SIZE", 100, 400, 200, 20)
MERGE_RADIUS = st.sidebar.slider("MERGE_RADIUS", 0.1, 2.0, 0.6, 0.05)
CONTOUR_LEVEL = st.sidebar.slider("CONTOUR_LEVEL", 0.1, 1.0, 0.4, 0.05)


# --------------------------
# Sidebar: параметры DBSCAN
# --------------------------
st.sidebar.markdown("## DBSCAN параметры")

EPS = st.sidebar.slider("EPS", 0.1, 2.0, 0.5, 0.05)
MIN_SAMPLES = st.sidebar.slider("MIN_SAMPLES", 2, 20, 5)


# --------------------------
# Параметры генерации
# --------------------------
st.sidebar.markdown("## Генерация данных")

N_CLUSTERS = st.sidebar.slider("Количество кластеров", 2, 10, 5)
POINTS_PER_CLUSTER = st.sidebar.slider("Точек на кластер", 10, 200, 50)
CLUSTER_SPREAD = st.sidebar.slider("Радиус разброса кластера", 0.1, 2.0, 0.6, 0.05)
SEED = st.sidebar.text_input("Сид рандома", value="42")


# --------------------------
# Выбор генератора
# --------------------------
generator_name = st.sidebar.selectbox(
    "Тип генератора",
    list(generators.GENERATOR_REGISTRY.keys())
)


@st.cache_data
def generate_points(generator_name, n_clusters, points_per_cluster, spread, seed=42):
    fn = generators.GENERATOR_REGISTRY[generator_name]
    return fn(
        n_clusters=n_clusters,
        points_per_cluster=points_per_cluster,
        cluster_spread=spread,
        seed=seed
    )


points, true_centers = generate_points(
    generator_name,
    N_CLUSTERS,
    POINTS_PER_CLUSTER,
    CLUSTER_SPREAD,
    seed=int(SEED, 0)
)


# ==========================
# -------- GRAVITY ---------
# ==========================

# Сетка
start = time.perf_counter()
X, Y = grid.make_grid(bounds=(-10, 10), grid_size=GRID_SIZE)

# Плотность
@st.cache_data
def compute_field(points, X, Y, sigma):
    return density.compute_density(points, X, Y, sigma)

field = compute_field(points, X, Y, SIGMA)

# Моды
mode_coords = modes.find_modes(
    field,
    X,
    Y,
    min_distance_px=int(GRID_SIZE * 0.05)
)

# Слияние мод
merged_modes = merge.merge_close_modes(
    mode_coords,
    MERGE_RADIUS
)

gravity_labels = clustering.assign_points(points, merged_modes)
gravity_time = time.perf_counter() - start


# ==========================
# -------- DBSCAN ----------
# ==========================

dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)

start = time.perf_counter()
db_labels = dbscan.fit_predict(points)
db_time = time.perf_counter() - start


# ==========================
# Метрики
# ==========================

def compute_metrics(X, labels):
    if len(np.unique(labels)) <= 1:
        return None, None
    return (
        silhouette_score(X, labels),
        davies_bouldin_score(X, labels)
    )


gravity_sil, gravity_db = compute_metrics(points, gravity_labels)
db_sil, db_db = compute_metrics(points, db_labels)


# ==========================
# Визуализация
# ==========================

fig = plot.draw(
    field=field,
    X=X,
    Y=Y,
    points=points,
    centers=merged_modes,
    labels=gravity_labels,
    db_labels=db_labels,
    contour_level=CONTOUR_LEVEL
)

st.pyplot(fig, width="stretch")


# ==========================
# Результаты сравнения
# ==========================

st.markdown("## 📊 Сравнение алгоритмов")

gravity_clusters = len(np.unique(gravity_labels))
db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
db_noise = int(np.sum(db_labels == -1))

comparison_table = {
    "Gravity": {
        "time_sec": gravity_time,
        "clusters": gravity_clusters,
        "silhouette": gravity_sil,
        "davies_bouldin": gravity_db
    },
    "DBSCAN": {
        "time_sec": db_time,
        "clusters": db_clusters,
        "noise_points": db_noise,
        "silhouette": db_sil,
        "davies_bouldin": db_db
    }
}

st.json(comparison_table)


# ==========================
# Информация о Gravity
# ==========================

st.markdown("## ℹ️ Gravity информация")

st.write(f"Найдено кластеров: {gravity_clusters}")

unique, counts = np.unique(gravity_labels, return_counts=True)

cluster_sizes = {
    int(k): int(v)
    for k, v in zip(unique, counts)
}

st.write("Размеры кластеров (Gravity):")
st.json(cluster_sizes)