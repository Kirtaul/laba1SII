import math
def read_cities(filename):
    cities = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            name = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            cities.append((name, x, y))
    return cities
def build_matrix(cities):
    n = len(cities)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = cities[i][1], cities[i][2]
        for j in range(i + 1, n):
            xj, yj = cities[j][1], cities[j][2]
            dist = math.hypot(xi - xj, yi - yj)
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix
def print_matrix(cities, matrix):
    n = len(cities)
    header = "     " + " ".join(f"{cities[i][0]:>6}" for i in range(n))
    print(header)
    for i in range(n):
        row = f"{cities[i][0]:>3} " + " ".join(f"{matrix[i][j]:6.2f}" for j in range(n))
        print(row)
def min_distances(cities, dist):
    print("\nМинимальное расстояние до ближайшего города для каждого города:")
    for i in range(len(cities)):
        min_d = min(dist[i][j] for j in range(len(cities)) if j != i)
        print(f"{cities[i][0]}: {min_d:.3f}")
