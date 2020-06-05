from clsframework import BatchIterator


def test_base_batch_iteration():
    b = BatchIterator(list(range(100)), batchsize=32)
    res = []
    for batch in b:
        res.append(batch)

    # Check if tuple is returned
    assert len(res[0]) == 2
    # Check if labels are None
    assert res[0][1] is None
    # Check size of first batch is ok
    assert len(res[0][0]) == 32
    # Check if size of last batch is ok
    assert len(res[-1][0]) == 4
