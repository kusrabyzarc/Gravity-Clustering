import matplotlib.pyplot as plt
import numpy as np

def draw(
    field,
    X,
    Y,
    points,
    centers,
    labels,
    contour_level=0.4,
    draw_lines=True
):
    """
    Визуализация плотностного поля и кластеров
    Все координаты — в реальном пространстве.
    """

    field_norm = (field - field.min()) / (field.max() - field.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Тепловая карта
    im = ax.imshow(
        field_norm,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        cmap='hot',
        interpolation='bilinear',
        alpha=0.6
    )

    # Контуры (ВАЖНО: через X,Y)
    if contour_level is not None:
        ax.contour(
            X,
            Y,
            field_norm,
            levels=[contour_level],
            colors='cyan',
            linewidths=2
        )

    # Линии точка → центр
    if draw_lines and points.size > 0 and centers.size > 0:
        for i in range(points.shape[0]):
            c = centers[labels[i]]
            p = points[i]
            ax.plot(
                [p[0], c[0]],
                [p[1], c[1]],
                color='white',
                lw=0.5,
                alpha=0.6,
                zorder=5
            )

    # Точки
    if points.size > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=labels,
            cmap='Blues',
            s=25,
            edgecolors='navy',
            linewidths=0.6,
            zorder=10
        )

    # Центры
    if centers.size > 0:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c='white',
            marker='*',
            s=200,
            edgecolors='black',
            linewidths=1.5,
            zorder=20
        )

    ax.set_aspect('equal')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Плотностная кластеризация', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Нормализованная плотность', fontsize=11)

    plt.tight_layout()
    return fig