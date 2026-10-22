import numpy as np
import sader_math
import time

# 1. Имитируем реальную задачу: поиск похожих новостей
# (как в проекте fin_news_ai)
print("=== Демонстрация интеграции C++ в Python ===")

# 5000 новостей с эмбеддингами по 768 чисел (как BERT)
news_embeddings = np.random.randn(5000, 768).astype(np.float64)
query_embedding = np.random.randn(768).astype(np.float64)

# Замеряем скорость через чистый NumPy
t0 = time.perf_counter()
numpy_distances = 1.0 - (news_embeddings @ query_embedding) / (
    np.linalg.norm(news_embeddings, axis=1) * np.linalg.norm(query_embedding)
)
numpy_time = (time.perf_counter() - t0) * 1000

# Замеряем через наш C++ модуль
t0 = time.perf_counter()
cpp_distances = sader_math.batch_distance(news_embeddings, query_embedding)
cpp_time = (time.perf_counter() - t0) * 1000

# Проверяем, что результаты одинаковые
np.testing.assert_allclose(numpy_distances, cpp_distances, rtol=1e-10)

print(f"NumPy:     {numpy_time:8.2f} ms")
print(f"C++ (pybind11): {cpp_time:8.2f} ms")
print(f"Ускорение: в {numpy_time/cpp_time:.2f} раз")

best_match = np.argmin(cpp_distances)
print(f"\nСамый похожий текст: индекс {best_match}, расстояние {cpp_distances[best_match]:.4f}")
