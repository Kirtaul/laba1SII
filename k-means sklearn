import time
import math
import tracemalloc
from sklearn.cluster import KMeans
from instr import build_matrix, print_matrix, min_distances, read_cities
def find_optimal_k(cities, max_k=5):
    n = len(cities)
    if max_k > n:
        max_k = n
    X = [[city[1], city[2]] for city in cities]
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        print(f"k={k}, inertia={kmeans.inertia_:.2f}")
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
def sk_kmeans_clustering(cities, k):
    X = [[city[1], city[2]] for city in cities]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    clusters_dict = {}
    for idx, label in enumerate(labels):
        name = cities[idx][0]
        clusters_dict.setdefault(label, []).append(name)
    clusters = [clusters_dict[i] for i in sorted(clusters_dict.keys())]
    return clusters, kmeans.inertia_
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
    tracemalloc.start()
    start_final = time.perf_counter()
    clusters, inertia = sk_kmeans_clustering(cities, optimal_k)
    end_final = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nВремя финальной кластеризации: {end_final - start_final:.4f} сек")
    print(f"Пиковое использование памяти (финальная кластеризация): {peak / 1024:.2f} КБ")
    print(f"\nРезультат кластеризации (метод K-средних через sklearn, K={optimal_k}):")
    for idx, cluster in enumerate(clusters):
        print(f"Кластер {idx + 1}: {cluster}")
    print(f"Инерция: {inertia:.2f}")
if __name__ == "__main__":
    main()
