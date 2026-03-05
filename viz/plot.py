import matplotlib.pyplot as plt
import numpy as np


def draw(
    field,
    X,
    Y,
    points,
    centers,
    labels=None,
    db_labels=None,
    contour_level=0.4,
    draw_lines=True
):
    """
    Визуализация:
    - плотностного поля
    - кластеров гравитационного метода
    - кластеров DBSCAN
    """

    field_norm = (field - field.min()) / (field.max() - field.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 7))

    # --- тепловая карта плотности ---
    im = ax.imshow(
        field_norm,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        cmap="hot",
        interpolation="bilinear",
        alpha=0.6,
    )

    # --- контур плотности ---
    if contour_level is not None:
        ax.contour(
            X,
            Y,
            field_norm,
            levels=[contour_level],
            colors="cyan",
            linewidths=2,
        )

    # --------------------------------------------------
    # ГРАВИТАЦИОННЫЙ МЕТОД
    # --------------------------------------------------

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

        # линии точка -> центр
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

    # центры
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

    # --------------------------------------------------
    # DBSCAN
    # --------------------------------------------------

    if points.size > 0 and db_labels is not None:

        db_labels = np.array(db_labels)

        noise_mask = db_labels == -1
        cluster_mask = db_labels != -1

        # кластеры
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

        # шум
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

    # --------------------------------------------------

    ax.set_aspect("equal")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Сравнение методов кластеризации", fontsize=14)

    ax.legend()

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Нормализованная плотность", fontsize=11)

    plt.tight_layout()

    return fig