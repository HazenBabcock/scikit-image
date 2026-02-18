import numpy as np
from skimage.measure import geometric_hashing
from skimage.transform import AffineTransform

xy1 = np.array(
    [
        [0.0, 0.0],
        [10.0, 10.0],
        [4.0, 6.0],
        [7.0, 3.0],
    ]
)

xy2 = np.array(
    [
        [0.0, 0.0],
        [10.0, 10.0],
        [4.0, 6.5],
        [7.0, 3.0],
    ]
)


def test_match():
    score, trans = geometric_hashing(xy1, xy1,
                                     min_size_1 = 1.0,
                                     max_size_1 = 100.0,
                                     min_size_2 = 1.0,
                                     max_size_2 = 100.0,
                                     verbose = False)
    assert score > 0.0
    assert np.allclose(np.eye(3), trans.params)

def test_no_match():
    score, trans = geometric_hashing(xy1, xy2,
                                     min_size_1 = 1.0,
                                     max_size_1 = 100.0,
                                     min_size_2 = 1.0,
                                     max_size_2 = 100.0,
                                     verbose = False)
    assert score == 0.0
    assert trans == None

def test_affine():
    atrans = AffineTransform(rotation = 0.25*np.pi, scale = 0.8)
    xy3 = atrans(xy1)
    score, trans = geometric_hashing(xy1, xy3,
                                     min_size_1 = 0.1,
                                     max_size_1 = 100.0,
                                     min_size_2 = 0.1,
                                     max_size_2 = 100.0,
                                     verbose = False)
    assert score > 0.0
    assert np.allclose(atrans.params, trans.params)

    
