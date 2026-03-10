import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


def draw(
    field,
    X,
    Y,
    points,
    centers,
    labels=None,
    db_labels=None,
    contour_level=0.4,
    draw_lines=True,
    projection_note=None,
):
    """
    Visualization:
    - 2D density field (if available)
    - gravity clusters
    - DBSCAN clusters
    """

    fig, ax = plt.subplots(figsize=(8, 7))
    im = None

    if field is not None and X is not None and Y is not None:
        field_norm = (field - field.min()) / (field.max() - field.min() + 1e-12)
        im = ax.imshow(
            field_norm,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            cmap="hot",
            interpolation="bilinear",
            alpha=0.6,
        )

    unique_labels = np.unique(labels) if labels is not None else np.array([])

    for lab in unique_labels:
        if lab == -1:
            continue

        cluster_points = points[labels == lab]

        if cluster_points.shape[0] < 3:
            continue

        try:
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])

            ax.plot(
                hull_points[:, 0],
                hull_points[:, 1],
                color="cyan",
                linewidth=2,
                zorder=15,
            )
        except Exception:
            # Degenerate geometry (e.g., collinear points) is safe to ignore.
            pass

    if points.size > 0 and labels is not None:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=labels,
            cmap="Blues",
            s=25,
            edgecolors="navy",
            linewidths=0.6,
            label="Gravity clusters",
            zorder=10,
        )

        if draw_lines and centers is not None and centers.size > 0:
            for i in range(points.shape[0]):
                c = centers[labels[i]]
                p = points[i]

                ax.plot(
                    [p[0], c[0]],
                    [p[1], c[1]],
                    color="white",
                    lw=0.5,
                    alpha=0.6,
                    zorder=5,
                )

    if centers is not None and centers.size > 0:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c="white",
            marker="*",
            s=200,
            edgecolors="black",
            linewidths=1.5,
            label="Gravity centers",
            zorder=20,
        )

    if points.size > 0 and db_labels is not None:
        db_labels = np.array(db_labels)

        noise_mask = db_labels == -1
        cluster_mask = db_labels != -1

        if np.any(cluster_mask):
            ax.scatter(
                points[cluster_mask, 0],
                points[cluster_mask, 1],
                c=db_labels[cluster_mask],
                cmap="Greens",
                s=30,
                marker="o",
                edgecolors="black",
                linewidths=0.4,
                label="DBSCAN clusters",
                zorder=12,
            )

        if np.any(noise_mask):
            ax.scatter(
                points[noise_mask, 0],
                points[noise_mask, 1],
                color="red",
                marker="x",
                s=35,
                label="DBSCAN noise",
                zorder=13,
            )

    ax.set_aspect("equal")

    if X is not None and Y is not None:
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Сравнение методов кластеризации", fontsize=14)

    if projection_note:
        ax.text(
            0.02,
            0.98,
            projection_note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            wrap=True,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.legend()

    if im is not None:
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Нормализованная плотность", fontsize=11)

    plt.tight_layout()

    return fig
