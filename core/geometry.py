def point_to_center_lines(points, centers, labels):
    lines = []
    for p, c in zip(points, labels):
        lines.append((points[p], centers[c]))
    return lines
