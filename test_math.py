import numpy as np
import pytest
import sader_math

def test_orthogonal():
    a = np.array([1., 0.])
    b = np.array([0., 1.])
    assert sader_math.cosine_distance(a, b) == pytest.approx(1.0)

def test_identical():
    a = np.array([1., 2., 3.])
    assert sader_math.cosine_distance(a, a) == pytest.approx(0.0)

def test_batch():
    x = np.array([[1., 0.], [0., 1.]])
    q = np.array([1., 0.])
    np.testing.assert_allclose(sader_math.batch_distance(x, q), [0., 1.])

def test_wrong_dimensions():
    with pytest.raises(ValueError):
        sader_math.cosine_distance(np.ones(2), np.ones(3))

def test_zero_vector():
    with pytest.raises(ValueError):
        sader_math.cosine_distance(np.zeros(2), np.ones(2))

def test_reject_wrong_dtype():
    with pytest.raises(TypeError):
        sader_math.cosine_distance(
            np.array([1, 2], dtype=np.int32),
            np.array([1, 2], dtype=np.int32)
        )
