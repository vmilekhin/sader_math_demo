import time, statistics
import numpy as np
import sader_math

N = 20000
DIMENSIONS = [32, 64, 128, 256, 384, 512, 768, 1024]

def measure(fn, repeats=7):
    fn()  # прогрев
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), result

rng = np.random.default_rng(42)
print(f"{'D':>6} {'NumPy, ms':>12} {'C++, ms':>12} {'Speedup':>10}")

for d in DIMENSIONS:
    x = rng.normal(size=(N, d)).astype(np.float64)
    q = rng.normal(size=d).astype(np.float64)

    def numpy_version():
        dot = x @ q
        norms = np.linalg.norm(x, axis=1) * np.linalg.norm(q)
        return 1.0 - dot / norms

    def cpp_version():
        return sader_math.batch_distance(x, q)

    numpy_ms, numpy_result = measure(numpy_version)
    cpp_ms, cpp_result = measure(cpp_version)

    np.testing.assert_allclose(numpy_result, cpp_result, rtol=1e-10, atol=1e-10)
    print(f"{d:6d} {numpy_ms:12.2f} {cpp_ms:12.2f} {numpy_ms/cpp_ms:10.2f}x")
