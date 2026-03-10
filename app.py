import csv
import io
from pathlib import Path

import streamlit as st
import numpy as np
import time

# sklearn
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# core modules
from core import grid, density, gravity
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
MERGE_RADIUS = st.sidebar.slider("MERGE_RADIUS", 0.1, 2.0, 0.6, 0.05)

# Внутренние параметры итерационного шага (не выносим в UI)
GRAVITY_MAX_ITER = 60
GRAVITY_TOL = 1e-3

# 2D-only control for heatmap quality
GRID_SIZE = st.sidebar.slider("GRID_SIZE (для 2D heatmap)", 100, 400, 200, 20)


# --------------------------
# Sidebar: параметры DBSCAN
# --------------------------
st.sidebar.markdown("## DBSCAN параметры")

EPS = st.sidebar.slider("EPS", 0.1, 4.0, 0.5, 0.05)
MIN_SAMPLES = st.sidebar.slider("MIN_SAMPLES", 2, 20, 5)
SCALE_FOR_DBSCAN = st.sidebar.checkbox("Scale features for DBSCAN", value=True)
USE_RECOMMENDED_EPS = st.sidebar.checkbox("Use recommended EPS", value=False)
EPS_QUANTILE = st.sidebar.slider("EPS quantile (k-distance)", 0.5, 0.99, 0.9, 0.01)


# --------------------------
# Выбор генератора
# --------------------------
st.sidebar.markdown("## Источник данных")
generator_name = st.sidebar.selectbox(
    "Тип генератора",
    list(generators.GENERATOR_REGISTRY.keys()),
)

# --------------------------
# Параметры генерации (для синтетики)
# --------------------------
if generator_name != generators.FROM_FILE_GENERATOR_NAME:
    st.sidebar.markdown("## Генерация данных")

    N_CLUSTERS = st.sidebar.slider("Количество кластеров", 2, 10, 5)
    POINTS_PER_CLUSTER = st.sidebar.slider("Точек на кластер", 10, 200, 50)
    CLUSTER_SPREAD = st.sidebar.slider("Радиус разброса кластера", 0.1, 2.0, 0.6, 0.05)
    SEED = st.sidebar.text_input("Сид рандома", value="42")


@st.cache_data
def generate_points(generator_name, n_clusters, points_per_cluster, spread, seed=42):
    fn = generators.GENERATOR_REGISTRY[generator_name]
    return fn(
        n_clusters=n_clusters,
        points_per_cluster=points_per_cluster,
        cluster_spread=spread,
        seed=seed,
    )


@st.cache_data
def recommend_eps(points, min_samples, quantile):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        return None

    k = max(2, min_samples)
    if points.shape[0] < k:
        k = points.shape[0]

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    kth_distances = distances[:, -1]

    return float(np.quantile(kth_distances, quantile))


@st.cache_data
def parse_csv_text(csv_text, delimiter=",", quotechar='"'):
    reader = csv.reader(io.StringIO(csv_text), delimiter=delimiter, quotechar=quotechar)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return [], [], np.empty((0, 0), dtype=float)

    header = rows[0]
    body = rows[1:] if header else rows

    numeric_col_names = []
    numeric_col_indices = []

    n_cols = len(header)
    for idx in range(n_cols):
        values = []
        ok = True
        for row in body:
            if idx >= len(row):
                ok = False
                break
            cell = row[idx].strip()
            if cell == "":
                ok = False
                break
            try:
                values.append(float(cell))
            except ValueError:
                ok = False
                break

        if ok and values:
            numeric_col_indices.append(idx)
            numeric_col_names.append(header[idx])

    if not numeric_col_indices:
        return header, [], np.empty((0, 0), dtype=float)

    matrix = np.array(
        [[float(row[idx].strip()) for idx in numeric_col_indices] for row in body],
        dtype=float,
    )

    return header, numeric_col_names, matrix


def build_results_csv(points, feature_names, gravity_labels, db_labels):
    out = io.StringIO()
    writer = csv.writer(out)

    header = ["row_id", *feature_names, "gravity_cluster", "dbscan_cluster"]
    writer.writerow(header)

    for i, row in enumerate(points):
        writer.writerow([
            i,
            *[float(x) for x in row],
            int(gravity_labels[i]),
            int(db_labels[i]),
        ])

    return out.getvalue().encode("utf-8-sig")


def safe_metrics(X_data, labels):
    if len(np.unique(labels)) <= 1:
        return None, None
    try:
        return (
            silhouette_score(X_data, labels),
            davies_bouldin_score(X_data, labels),
        )
    except Exception:
        return None, None


def make_sweep_values(base_value, min_value, max_value, multipliers=(0.7, 1.0, 1.3), ndigits=4):
    vals = []
    for m in multipliers:
        v = max(min_value, min(max_value, float(base_value) * float(m)))
        vals.append(round(v, ndigits))
    return sorted(set(vals))


def run_parameter_sweep(
    points,
    dbscan_input,
    sigma,
    merge_radius,
    eps,
    min_samples,
    gravity_max_iter,
    gravity_tol,
):
    gravity_sigmas = make_sweep_values(sigma, 0.1, 3.0)
    gravity_merges = make_sweep_values(merge_radius, 0.1, 2.0)

    db_eps_values = make_sweep_values(eps, 0.1, 4.0)
    db_min_samples = sorted(set([
        int(max(2, min(20, min_samples - 2))),
        int(max(2, min(20, min_samples))),
        int(max(2, min(20, min_samples + 2))),
    ]))

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([
        "algorithm",
        "sigma",
        "merge_radius",
        "eps",
        "min_samples",
        "time_sec",
        "clusters",
        "noise_points",
        "silhouette",
        "davies_bouldin",
        "converged",
        "iterations",
    ])

    row_count = 0

    for s in gravity_sigmas:
        for mr in gravity_merges:
            start = time.perf_counter()
            result = gravity.cluster_nd(
                points=points,
                sigma=s,
                merge_radius=mr,
                max_iter=gravity_max_iter,
                tol=gravity_tol,
            )
            elapsed = time.perf_counter() - start

            labels = result["labels"]
            n_clusters = len(np.unique(labels))
            sil, dbi = safe_metrics(points, labels)

            writer.writerow([
                "Gravity",
                s,
                mr,
                "",
                "",
                elapsed,
                n_clusters,
                "",
                sil,
                dbi,
                result["converged"],
                result["iterations"],
            ])
            row_count += 1

    for e in db_eps_values:
        for ms in db_min_samples:
            start = time.perf_counter()
            labels = DBSCAN(eps=e, min_samples=ms).fit_predict(dbscan_input)
            elapsed = time.perf_counter() - start

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = int(np.sum(labels == -1))
            sil, dbi = safe_metrics(points, labels)

            writer.writerow([
                "DBSCAN",
                "",
                "",
                e,
                ms,
                elapsed,
                n_clusters,
                noise_points,
                sil,
                dbi,
                "",
                "",
            ])
            row_count += 1

    return out.getvalue().encode("utf-8-sig"), row_count


# ==========================
# Подготовка данных
# ==========================

true_centers = None
feature_names = None

if generator_name == generators.FROM_FILE_GENERATOR_NAME:
    st.sidebar.markdown("## From file")
    source_kind = st.sidebar.radio("Источник CSV", ["Drag'n'drop", "Путь к файлу"])

    csv_text = None
    file_label = None

    if source_kind == "Drag'n'drop":
        uploaded = st.sidebar.file_uploader(
            "CSV файл",
            type=["csv", "txt"],
            help="Перетащите CSV сюда или выберите файл",
        )
        if uploaded is not None:
            file_label = uploaded.name
            csv_text = uploaded.getvalue().decode("utf-8-sig", errors="replace")
    else:
        file_path = st.sidebar.text_input("Путь к CSV", value="")
        if file_path.strip():
            p = Path(file_path.strip())
            if p.exists() and p.is_file():
                file_label = str(p)
                csv_text = p.read_text(encoding="utf-8-sig", errors="replace")
            else:
                st.sidebar.error("Файл не найден")

    delimiter = st.sidebar.selectbox("Разделитель", [",", ";", "\t"], index=0)

    if not csv_text:
        st.info("Выберите CSV файл для анализа.")
        st.stop()

    header, numeric_columns, numeric_matrix = parse_csv_text(csv_text, delimiter=delimiter)

    if numeric_matrix.shape[0] == 0 or numeric_matrix.shape[1] == 0:
        st.error("Не удалось найти числовые столбцы в CSV.")
        st.stop()

    default_cols = numeric_columns
    selected_cols = st.sidebar.multiselect(
        "Числовые столбцы для кластеризации",
        options=numeric_columns,
        default=default_cols,
    )

    if len(selected_cols) == 0:
        st.error("Нужно выбрать хотя бы один столбец.")
        st.stop()

    selected_idx = [numeric_columns.index(col) for col in selected_cols]
    points = numeric_matrix[:, selected_idx]
    feature_names = selected_cols

    st.caption(f"Источник: {file_label}")
    st.caption(f"Используемые признаки: {', '.join(selected_cols)}")
    st.caption(f"Форма данных: {points.shape[0]} x {points.shape[1]}")

else:
    points, true_centers = generate_points(
        generator_name,
        N_CLUSTERS,
        POINTS_PER_CLUSTER,
        CLUSTER_SPREAD,
        seed=int(SEED, 0),
    )
    feature_names = [f"f{i}" for i in range(points.shape[1])]


# ==========================
# -------- GRAVITY ---------
# ==========================

start = time.perf_counter()
gravity_result = gravity.cluster_nd(
    points=points,
    sigma=SIGMA,
    merge_radius=MERGE_RADIUS,
    max_iter=GRAVITY_MAX_ITER,
    tol=GRAVITY_TOL,
)
gravity_time = time.perf_counter() - start

gravity_labels = gravity_result["labels"]
merged_modes = gravity_result["centers"]


# 2D density field is available only for 2D inputs
field = None
X = None
Y = None
if points.shape[1] == 2:
    X, Y = grid.make_grid(bounds=(-10, 10), grid_size=GRID_SIZE)

    @st.cache_data
    def compute_field(points2d, X_grid, Y_grid, sigma):
        return density.compute_density(points2d, X_grid, Y_grid, sigma)

    field = compute_field(points, X, Y, SIGMA)


# ==========================
# -------- DBSCAN ----------
# ==========================

dbscan_input = points
if SCALE_FOR_DBSCAN:
    dbscan_input = StandardScaler().fit_transform(points)

recommended_eps = recommend_eps(dbscan_input, MIN_SAMPLES, EPS_QUANTILE)
actual_eps = recommended_eps if (USE_RECOMMENDED_EPS and recommended_eps is not None) else EPS

if recommended_eps is not None:
    st.sidebar.caption(
        f"Recommended eps (q={EPS_QUANTILE:.2f}, k={max(2, MIN_SAMPLES)}): {recommended_eps:.3f}"
    )

if points.shape[1] > 2 and not SCALE_FOR_DBSCAN:
    st.sidebar.warning("D>2: enable 'Scale features for DBSCAN'.")

dbscan = DBSCAN(eps=actual_eps, min_samples=MIN_SAMPLES)

start = time.perf_counter()
db_labels = dbscan.fit_predict(dbscan_input)
db_time = time.perf_counter() - start


# ==========================
# Метрики
# ==========================

gravity_sil, gravity_db = safe_metrics(points, gravity_labels)
db_sil, db_db = safe_metrics(points, db_labels)


# ==========================
# Экспорт CSV
# ==========================

results_csv = build_results_csv(points, feature_names, gravity_labels, db_labels)
st.download_button(
    label="Скачать результаты кластеризации (CSV)",
    data=results_csv,
    file_name="cluster_assignments.csv",
    mime="text/csv",
)


# ==========================
# Перебор параметров
# ==========================

if st.button("Поиграться с параметрами"):
    with st.spinner("Выполняю перебор параметров..."):
        sweep_csv, sweep_rows = run_parameter_sweep(
            points=points,
            dbscan_input=dbscan_input,
            sigma=SIGMA,
            merge_radius=MERGE_RADIUS,
            eps=actual_eps,
            min_samples=MIN_SAMPLES,
            gravity_max_iter=GRAVITY_MAX_ITER,
            gravity_tol=GRAVITY_TOL,
        )
        st.session_state["sweep_csv"] = sweep_csv
        st.session_state["sweep_rows"] = sweep_rows

if "sweep_csv" in st.session_state:
    st.success(f"Перебор готов: {st.session_state['sweep_rows']} запусков.")
    st.download_button(
        label="Скачать перебор параметров (CSV)",
        data=st.session_state["sweep_csv"],
        file_name="parameter_sweep_results.csv",
        mime="text/csv",
        key="download_parameter_sweep",
    )


# ==========================
# Подготовка проекции для plot
# ==========================

projection_note = None
n_features = points.shape[1]

if n_features == 1:
    plot_points = np.column_stack([points[:, 0], np.zeros(points.shape[0])])
    plot_centers = np.column_stack([merged_modes[:, 0], np.zeros(merged_modes.shape[0])])
    projection_note = "1D -> (x0, 0)"
elif n_features == 2:
    plot_points = points
    plot_centers = merged_modes
else:
    plot_points = points[:, :2]
    plot_centers = merged_modes[:, :2]
    projection_note = f"{n_features}D -> x0,x1 | clustering in full space"


# ==========================
# Визуализация
# ==========================

fig = plot.draw(
    field=field,
    X=X,
    Y=Y,
    points=plot_points,
    centers=plot_centers,
    labels=gravity_labels,
    db_labels=db_labels,
    projection_note=projection_note,
)

st.pyplot(fig, width="stretch")


# ==========================
# Результаты сравнения
# ==========================

st.markdown("## Сравнение алгоритмов")

gravity_clusters = len(np.unique(gravity_labels))
db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
db_noise = int(np.sum(db_labels == -1))

comparison_table = {
    "Gravity": {
        "time_sec": gravity_time,
        "clusters": gravity_clusters,
        "silhouette": gravity_sil,
        "davies_bouldin": gravity_db,
        "converged": gravity_result["converged"],
        "iterations": gravity_result["iterations"],
    },
    "DBSCAN": {
        "time_sec": db_time,
        "clusters": db_clusters,
        "noise_points": db_noise,
        "silhouette": db_sil,
        "davies_bouldin": db_db,
        "eps_used": actual_eps,
        "scaled_features": SCALE_FOR_DBSCAN,
    },
}

st.json(comparison_table)


# ==========================
# Информация о Gravity
# ==========================

st.markdown("## Gravity информация")

st.write(f"Найдено кластеров: {gravity_clusters}")

unique, counts = np.unique(gravity_labels, return_counts=True)
cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}

st.write("Размеры кластеров (Gravity):")
st.json(cluster_sizes)

if projection_note:
    st.info(projection_note)
