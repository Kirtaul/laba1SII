import time
import tracemalloc
from instr import build_matrix, print_matrix, min_distances, read_cities
def ward_clustering(cities, k):
    n = len(cities)
    clusters = []
    for i in range(n):
        clusters.append({
            'indices': [i],
            'centroid': (cities[i][1], cities[i][2]),
            'size': 1
        })
    while len(clusters) > k:
        min_delta = float('inf')
        to_merge = (0, 0)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1 = clusters[i]['centroid']
                c2 = clusters[j]['centroid']
                n1 = clusters[i]['size']
                n2 = clusters[j]['size']
                dx = c1[0] - c2[0]
                dy = c1[1] - c2[1]
                dist2 = dx * dx + dy * dy
                delta = (n1 * n2) / (n1 + n2) * dist2
                if delta < min_delta:
                    min_delta = delta
                    to_merge = (i, j)
        i, j = to_merge
        new_indices = clusters[i]['indices'] + clusters[j]['indices']
        new_size = clusters[i]['size'] + clusters[j]['size']
        sum_x = sum(cities[idx][1] for idx in new_indices)
        sum_y = sum(cities[idx][2] for idx in new_indices)
        new_centroid = (sum_x / new_size, sum_y / new_size)
        new_cluster = {
            'indices': new_indices,
            'centroid': new_centroid,
            'size': new_size
        }
        clusters[i] = new_cluster
        del clusters[j]
    result = []
    for cl in clusters:
        names = [cities[idx][0] for idx in cl['indices']]
        result.append(names)
    return result
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
    try:
        k = int(input("\nВведите количество кластеров K: "))
    except:
        print("Ошибка ввода, будет использовано K = 2")
        k = 2
    k = max(1, min(k, len(cities)))
    # Замер времени и памяти
    tracemalloc.start()
    start_cluster = time.perf_counter()
    clusters = ward_clustering(cities, k)
    end_cluster = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nВремя выполнения кластеризации: {end_cluster - start_cluster:.4f} сек")
    print(f"Пиковое использование памяти: {peak / 1024:.2f} КБ")
    print(f"\nРезультат кластеризации (метод Уорда, K={k}):")
    for idx, cluster in enumerate(clusters):
        print(f"Кластер {idx + 1}: {cluster}")
    # Оценка качества: инерция
    # Преобразуем в метки
    labels = [-1] * len(cities)
    for label, cluster in enumerate(clusters):
        for city_name in cluster:
            idx = next(i for i, c in enumerate(cities) if c[0] == city_name)
            labels[idx] = label
    # Вычисляем центроиды и инерцию
    centroids = []
    for cluster in clusters:
        if cluster:
            sum_x = sum(cities[i][1] for i in [j for j, (name,_,_) in enumerate(cities) if name in cluster])
            sum_y = sum(cities[i][2] for i in [j for j, (name,_,_) in enumerate(cities) if name in cluster])
            cnt = len(cluster)
            centroids.append((sum_x / cnt, sum_y / cnt))
        else:
            centroids.append((0.0, 0.0))
    inertia = 0.0
    for i, (_, x, y) in enumerate(cities):
        cluster_id = labels[i]
        cx, cy = centroids[cluster_id]
        inertia += (x - cx) ** 2 + (y - cy) ** 2
    print(f"\nИнерция (сумма квадратов расстояний до центроидов): {inertia:.2f}")
if __name__ == "__main__":
    main()
