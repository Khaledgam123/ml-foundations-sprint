import numpy as np, time

def flatten_np(a: np.ndarray) -> np.ndarray:
    """Vectorized flatten using NumPy ravel (no Python loops)."""
    return a.ravel()

def argmax_np(a: np.ndarray) -> int:
    """Vectorized arg-max index."""
    return int(np.argmax(a))

if __name__ == "__main__":
    # Build a 1000×1000 list of ints
    lst = [list(range(1000))] * 1000
    arr = np.array(lst)

    t = time.time()
    _ = [x for sub in lst for x in sub]
    print("python flatten:", round(time.time() - t, 4), "s")

    t = time.time()
    _ = flatten_np(arr)
    print("numpy  flatten:", round(time.time() - t, 4), "s")

    t = time.time()
    _ = max(range(len(lst[0])), key=lambda i: lst[0][i])
    print("python argmax :", round(time.time() - t, 4), "s")

    t = time.time()
    _ = argmax_np(arr)
    print("numpy  argmax :", round(time.time() - t, 4), "s")