import numpy as np

from evaluation.modern_ai.statistics import stratified_group_split


def test_group_split_never_leaks_source():
    groups = np.array(["a","a","b","b","c","c","d","d"], dtype=object)
    y = np.array([0,0,0,0,1,1,1,1])
    val, test = stratified_group_split(y, groups, validation_fraction=.5, seed=1)
    assert set(groups[val]).isdisjoint(set(groups[test]))
    assert set(y[val]) == {0,1}
    assert set(y[test]) == {0,1}
