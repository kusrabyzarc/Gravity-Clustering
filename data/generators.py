import numpy as np


FROM_FILE_GENERATOR_NAME = "From file"


def clearly_clusterized(n_clusters=4, points_per_cluster=50, cluster_spread=0.5, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-7, 7, size=(n_clusters, 2))
    points = []

    for c in centers:
        pts = rng.normal(loc=c, scale=cluster_spread, size=(points_per_cluster, 2))
        points.append(pts)

    return np.vstack(points), centers


def multidimensional_clustered_gaussian(
    n_clusters=4,
    points_per_cluster=50,
    cluster_spread=0.5,
    seed=42,
    n_features=5,
):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-7, 7, size=(n_clusters, n_features))
    points = []

    for c in centers:
        pts = rng.normal(loc=c, scale=cluster_spread, size=(points_per_cluster, n_features))
        points.append(pts)

    return np.vstack(points), centers


def concentric_circles(n_clusters=3, points_per_cluster=100, cluster_spread=0.05, seed=42):
    rng = np.random.default_rng(seed)
    radii = np.linspace(2, 6, n_clusters)
    points = []

    for r in radii:
        angles = rng.uniform(0, 2 * np.pi, points_per_cluster)
        x = r * np.cos(angles) + rng.normal(0, cluster_spread, points_per_cluster)
        y = r * np.sin(angles) + rng.normal(0, cluster_spread, points_per_cluster)
        points.append(np.column_stack((x, y)))

    return np.vstack(points), None


def spiral(n_clusters=3, points_per_cluster=200, cluster_spread=0.2, seed=42):
    rng = np.random.default_rng(seed)
    points = []

    for i in range(n_clusters):
        t = np.linspace(0, 4 * np.pi, points_per_cluster)
        r = 0.5 * t
        x = r * np.cos(t + i * 2 * np.pi / n_clusters)
        y = r * np.sin(t + i * 2 * np.pi / n_clusters)

        x += rng.normal(0, cluster_spread, points_per_cluster)
        y += rng.normal(0, cluster_spread, points_per_cluster)

        points.append(np.column_stack((x, y)))

    return np.vstack(points), None


def single_circle(n_clusters=3, points_per_cluster=200, cluster_spread=0.2, seed=42):
    rng = np.random.default_rng(seed)

    r = cluster_spread ** 2

    total_points = n_clusters * points_per_cluster
    angles = rng.uniform(0, 2 * np.pi, total_points)

    x = r * np.cos(angles)
    y = r * np.sin(angles)

    points = np.column_stack((x, y))

    return points, np.array([[0.0, 0.0]])



def uniform_noise(n_clusters=1, points_per_cluster=300, cluster_spread=None, seed=42):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-8, 8, size=(points_per_cluster, 2))
    return points, None



def from_file_generator(n_clusters=1, points_per_cluster=1, cluster_spread=1.0, seed=42):
    raise RuntimeError("'From file' data source should be handled in app.py")


GENERATOR_REGISTRY = {
    "Clustered Gaussian": clearly_clusterized,
    "Multidimensional Gaussian": multidimensional_clustered_gaussian,
    "Concentric Circles": concentric_circles,
    "Spiral": spiral,
    "Uniform Noise": uniform_noise,
    "Single circle": single_circle,
    FROM_FILE_GENERATOR_NAME: from_file_generator,
}
