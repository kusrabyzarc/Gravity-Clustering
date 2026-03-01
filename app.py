import streamlit as st
import numpy as np

# core modules
from core import grid, density, modes, merge, clustering
# viz module
from viz import plot
# data generator
from data import generators

st.set_page_config(page_title="Density Clustering Interactive", layout="wide")
st.title("Интерактивная плотностная кластеризация")

# --------------------------
# Sidebar: параметры
# --------------------------
SIGMA = st.sidebar.slider("SIGMA (ширина ядра)", 0.1, 3.0, 0.8, 0.05)
GRID_SIZE = st.sidebar.slider("GRID_SIZE", 100, 400, 200, 20)
MERGE_RADIUS = st.sidebar.slider("MERGE_RADIUS", 0.1, 2.0, 0.6, 0.05)
CONTOUR_LEVEL = st.sidebar.slider("CONTOUR_LEVEL", 0.1, 1.0, 0.4, 0.05)

N_CLUSTERS = st.sidebar.slider("Количество кластеров", 2, 10, 5)
POINTS_PER_CLUSTER = st.sidebar.slider("Точек на кластер", 10, 100, 50)
CLUSTER_SPREAD = st.sidebar.slider("Радиус разброса кластера", 0.1, 2.0, 0.6, 0.05)
SEED = st.sidebar.text_input("Сид рандома", value="42")

# --------------------------
# Выбор генератора
# --------------------------
generator_name = st.sidebar.selectbox(
    "Тип генератора",
    list(generators.GENERATOR_REGISTRY.keys())
)

generator_fn = generators.GENERATOR_REGISTRY[generator_name]


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

# --------------------------
# Сетка
# --------------------------
X, Y = grid.make_grid(bounds=(-10, 10), grid_size=GRID_SIZE)

# --------------------------
# Плотность
# --------------------------
@st.cache_data
def compute_field(points, X, Y, sigma):
    return density.compute_density(points, X, Y, sigma)

field = compute_field(points, X, Y, SIGMA)

# --------------------------
# Моды (реальные координаты!)
# --------------------------
mode_coords = modes.find_modes(
    field,
    X,
    Y,
    min_distance_px=int(GRID_SIZE * 0.05)
)

# --------------------------
# Слияние мод (в реальном пространстве)
# --------------------------
merged_modes = merge.merge_close_modes(
    mode_coords,
    MERGE_RADIUS
)

# --------------------------
# Классификация точек
# --------------------------
labels = clustering.assign_points(points, merged_modes)

# --------------------------
# Визуализация
# --------------------------
fig = plot.draw(
    field=field,
    X=X,
    Y=Y,
    points=points,
    centers=merged_modes,
    labels=labels,
    contour_level=CONTOUR_LEVEL
)

st.pyplot(fig, width="stretch")

# --------------------------
# Информация о кластерах
# --------------------------
st.write(f"Найдено кластеров: {len(merged_modes)}")
st.write("Размеры кластеров:")
unique, counts = np.unique(labels, return_counts=True)

cluster_sizes = {
    int(k): int(v)
    for k, v in zip(unique, counts)
}

st.json(cluster_sizes)