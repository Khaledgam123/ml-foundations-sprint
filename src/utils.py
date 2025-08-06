def flatten(list_of_lists):
    """[[1,2],[3]] → [1,2,3]"""
    return [x for lst in list_of_lists for x in lst]

def argmax(nums):
    """index of max value"""
    best_i, best_v = 0, float("-inf")
    for i, v in enumerate(nums):
        if v > best_v:
            best_i, best_v = i, v
    return best_i
def train_val_split(X, y, val_ratio=0.2, seed=42):
    """shuffle + split"""
    import random
    assert 0.0 < val_ratio < 1.0
    assert len(X) == len(y)
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - val_ratio))
    tr, va = idx[:cut], idx[cut:]
    Xtr = [X[i] for i in tr]; ytr = [y[i] for i in tr]
    Xva = [X[i] for i in va]; yva = [y[i] for i in va]
    return Xtr, Xva, ytr, yva
