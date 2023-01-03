import numpy as np


def shuffle(array_1, array_2, random_state=42):
    if random_state:
        np.random.seed(random_state)
    print(len(array_1), len(array_2))
    assert len(array_1) == len(array_2)
    permutation = np.random.permutation(len(array_1))
    return array_1[permutation], array_2[permutation]


