import math
import random
import time
import tracemalloc
from instr import build_matrix, print_matrix, min_distances, read_cities
def kmeans_clustering(cities, k, max_iter=100, tol=1e-4):
    n = len(cities)
    indices = list(range(n))
    random.shuffle(indices)
    centroids = [(cities[idx][1], cities[idx][2]) for idx in indices[:k]]
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for i, (_, x, y) in enumerate(cities):
            distances = [math.hypot(x - cx, y - cy) for (cx, cy) in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(i)
        new_centroids = []
        for idx in range(k):
            if not clusters[idx]:
                new_center_idx = random.choice(range(n))
                new_centroids.append((cities[new_center_idx][1], cities[new_center_idx][2]))
            else:
                avg_x = sum(cities[i][1] for i in clusters[idx]) / len(clusters[idx])
                avg_y = sum(cities[i][2] for i in clusters[idx]) / len(clusters[idx])
                new_centroids.append((avg_x, avg_y))
        shift = sum(math.hypot(new_centroids[i][0] - centroids[i][0],
                               new_centroids[i][1] - centroids[i][1])
                    for i in range(k))
        centroids = new_centroids
        if shift < tol:
            break
    else:
        print("Достигнуто максимальное число итераций")
    result = []
    inertia = 0.0
    for cluster_idx, cluster in enumerate(clusters):
        if cluster:
            names = [cities[i][0] for i in cluster]
            result.append(names)
            cx, cy = centroids[cluster_idx]
            for i in cluster:
                x, y = cities[i][1], cities[i][2]
                inertia += (x - cx) ** 2 + (y - cy) ** 2
    return result, inertia
def find_optimal_k(cities, max_k=None):
    n = len(cities)
    if max_k is None:
        max_k = min(50, n)
    inertias = []
    for k in range(1, max_k + 1):
        best_inertia = float('inf')
        for _ in range(5):
            _, inertia = kmeans_clustering(cities, k)
            if inertia < best_inertia:
                best_inertia = inertia
        inertias.append(best_inertia)
        print(f"k={k}, inertia={best_inertia:.2f}")
    if max_k < 3:
        return 1, inertias
    x1, y1 = 1, inertias[0]
    x2, y2 = max_k, inertias[-1]
    max_dist = -1
    optimal_k = 1
    for k in range(2, max_k):
        x0 = k
        y0 = inertias[k - 1]
        numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
        denominator = math.hypot(x2 - x1, y2 - y1)
        if denominator == 0:
            continue
        dist = numerator / denominator
        if dist > max_dist:
            max_dist = dist
            optimal_k = k
    return optimal_k, inertias
def main():
    filename = "cities"
    cities = read_cities(filename)
    if not cities:
        print("Файл пуст или не содержит данных.")
    print("Прочитаны города:")
    for name, x, y in cities:
        print(f"{name}: ({x}, {y})")
    dist = build_matrix(cities)
    print("\nМатрица смежности (расстояния):")
    print_matrix(cities, dist)
    min_distances(cities, dist)
    start_opt = time.perf_counter()
    optimal_k, inertias = find_optimal_k(cities)
    end_opt = time.perf_counter()
    print(f"\nОптимальное число кластеров по правилу локтя: {optimal_k}")
    print(f"Время поиска оптимального k: {end_opt - start_opt:.4f} сек")
    # Замер памяти и времени для финальной кластеризации
    tracemalloc.start()
    start_final = time.perf_counter()
    best_clusters = None
    best_inertia = float('inf')
    for _ in range(5):
        clusters, inertia = kmeans_clustering(cities, optimal_k)
        if inertia < best_inertia:
            best_inertia = inertia
            best_clusters = clusters
    end_final = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nВремя финальной кластеризации: {end_final - start_final:.4f} сек")
    print(f"Пиковое использование памяти (финальная кластеризация): {peak / 1024:.2f} КБ")
    print(f"\nРезультат кластеризации (метод K-средних, K={optimal_k}):")
    for idx, cluster in enumerate(best_clusters):
        print(f"Кластер {idx + 1}: {cluster}")
    print(f"Инерция (лучшая из 5 запусков): {best_inertia:.2f}")
if __name__ == "__main__":
    main()
