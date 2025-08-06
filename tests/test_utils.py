from src.utils import flatten, argmax, train_val_split
def test_flatten():
    assert flatten([[1, 2], [3]]) == [1, 2, 3]
def test_argmax():
    assert argmax([0, 5, 2, 4]) == 1
def test_train_val_split_shapes():
    X = list(range(100)); y = list(range(100))
    Xtr, Xva, ytr, yva = train_val_split(X, y, val_ratio=0.2, seed=0)
    assert len(Xtr) == 80 and len(Xva) == 20
    assert len(ytr) == 80 and len(yva) == 20
