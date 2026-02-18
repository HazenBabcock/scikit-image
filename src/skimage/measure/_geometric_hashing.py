import math
import numpy as np
import scipy as sp
from ..transform import AffineTransform


def _fg_probability(kd1, kd2, transform, bg_p):
    """
    Returns an estimate of how likely the transform is correct.

    Parameters
    ----------
    kd1 : scipy.spatial.KDTree object.
        A KDTree with the first set of points.
    kd2 : scipy.spatial.KDTree object.
        A KDTree with the second set of points.
    transform : An AffineTransform object.
        A transform form that maps the first set of points
        to the second set of points.
    bg_p : float
        The density of points in the first set. This is used
        as an estimate of how likely the two sets of points
        are to overlap by chance.

    Returns
    -------
    foreground_probability : float
        An estimate of how likely the transform is correct.
    """
    # Transform 'other' coordinates into the 'reference' frame.
    xy2 = transform.inverse(kd2.data)

    # Calculate distance to nearest point in 'reference'.
    [dist, index] = kd1.query(xy2)

    # Score assuming a localization accuracy of 1 pixel.
    fg_p = bg_p + (1.0 - bg_p) * np.sum(np.exp(-dist*dist*0.5))/float(len(xy2))
    return fg_p


def geometric_hashing(
    xy1,
    xy2,
    min_size_1 = None,
    max_size_1 = None,
    max_neighbors_1 = None,
    min_size_2 = None,
    max_size_2 = None,
    max_neighbors_2 = None,
    tolerance = 1.0e-2,
    verbose = True
):
    """
    Given two ndarrays of the locations of reference points in 2D
    this returns a 'first guess' at the transform between them.
    This is an AffineTransform. As such this could fail for images
    with large differences in field curvature.

    This uses the ideas in [1], applied to fiducial references
    such as flourescent beads.

    Unlike astrometry.net we are just comparing all the quads looking
    for the one that has the best score. This should be approximately
    7 (for 200 pts on 512 x 512 image) as, based on testing, you can
    sometimes get scores as high as 4-5 for random sets of points.

    This can be quite slow depending on the settings. The idea is to
    tune min_size and max_size such that a few hundred quads are
    created for each set of points. You need enough a quads to have
    a good chance of seeing a match, but not so many that the search
    takes forever (it scales as the number of quads squared). The
    ideal number of quads depends on how many points are 'noise',
    i.e. present in one set but not present in the other.

    Parameters
    ----------
    xy1 : (N, 2) ndarray
        The points in the first image.
    xy2 : (N, 2) ndarray
        The points in the second image.
    min_size_1 : float, optional
        The minimum quad size in the first coordinate space.
    max_size_1 : float, optional
        The maximum quad size in the first coordinate space.
    max_neighbors_1 : int, optional
        The maximum number of neighboring points to consider when
        making quads.
    min_size_2 : float, optional
        The minimum quad size in the second coordinate space.
    max_size_2 : float, optional
        The maximum quad size in the second coordinate space.
    max_neighbors_2 : int, optional
        The maximum number of neighboring points to consider when
        making quads.
    tolerance : float, optional
        The tolerance factor for matching quads.
    verbose : boolean, optional
        Print progress messages.
    
    Returns
    -------
    best_score : float
        The score of the best transform.
    
    best_transform : ndarry
        The best AffineTransform to map from xy1 coordinate space
        to xy2 coordinate space.
    
    References
    ----------
    .. [1] Lang, D., Hogg, D. W., Mierle, K., Blanton, M., & Roweis, S.,
           Astrometry.net: Blind astrometric calibration of arbitrary
           astronomical images, The Astronomical Journal 139, pp1782-1800,  
           (2010). :DOI: `10.1088/0004-6256/139/5/1782`
    
    Examples
    --------

    Create two sets of points for best transform identification.
    This is 100 points in a 512x512 image.
    
    >>> rng = np.random.default_rng(42)
    >>> npts = 100
    >>> xy1 = np.stack((rng.uniform(0, 512, npts), rng.uniform(0, 512, npts)), axis = 1)
    >>> xy2 = np.copy(xy1)
    >>> xy1 += rng.normal(scale = 0.2, size = xy1.shape)
    >>> xy2 += rng.normal(scale = 0.2, size = xy1.shape)

    Add noise points.
    
    >>> nns = 30
    >>> xy1n = np.concatenate((xy1, np.stack((rng.uniform(0, 512, nns), rng.uniform(0, 512, nns)), axis = 1)), axis = 0)
    >>> xy2n = np.concatenate((xy2, np.stack((rng.uniform(0, 512, nns), rng.uniform(0, 512, nns)), axis = 1)), axis = 0)

    Shuffle points and apply an affine tranform to the second set of points.
    
    >>> atrans = ski.transform.AffineTransform(scale = 1.5, rotation = 0.2*np.pi, translation = (100, 200))
    >>> rng.shuffle(xy1n)
    >>> rng.shuffle(xy2n)
    >>> xy3n = atrans(xy2n)

    Find the best transform between the two sets of points.

    >>> score, trans = ski.measure.geometric_hashing(xy1n, xy3n)
    Created 6259 quads from xy1
    Created 4933 quads from xy2
    
    Comparing quads.
    Match 0 score 6.10
    Match 1 score 6.93
    Match 4 score 7.09
    Match 42 score 7.11
    Match 49 score 7.11
    Match 276 score 7.18
    Match 361 score 7.19
    Found 1111 matching quads

    Compare the original transform to the found transform.
    
    >>> print(trans)
    <AffineTransform(matrix=
    [[  1.21420286,  -0.88214283,  99.97610601],
     [  0.88080492,   1.21170114, 200.72283236],
     [  0.        ,   0.        ,   1.        ]])>
    >>> print(atrans)
    <AffineTransform(matrix=
    [[  1.21352549,  -0.88167788, 100.        ],
     [  0.88167788,   1.21352549, 200.        ],
     [  0.        ,   0.        ,   1.        ]])>
    """

    # Estimate some sensible defaults..
    max_neighbors_1 = max_neighbors_1 if not max_neighbors_1 is None else 20
    max_neighbors_2 = max_neighbors_2 if not max_neighbors_2 is None else 20

    if min_size_1 is None:
        min_size_1 = 0.05 * np.max(np.max(xy1, axis = 0) - np.min(xy1, axis = 0))
        
    if min_size_2 is None:
        min_size_2 = 0.05 * np.max(np.max(xy2, axis = 0) - np.min(xy2, axis = 0))

    if max_size_1 is None:
        max_size_1 = 0.5 * np.max(np.max(xy1, axis = 0) - np.min(xy1, axis = 0))

    if max_size_2 is None:
        max_size_2 = 0.5 * np.max(np.max(xy2, axis = 0) - np.min(xy2, axis = 0))

    # Estimate the density of points in xy1.
    sxy = np.max(xy1, axis = 0) - np.min(xy1, axis = 0)
    density = len(xy1)/(sxy[0]*sxy[1])

    # Make KDTrees and Quads for matching.
    kd1, quads1 = _make_tree_and_quads(xy1, min_size_1, max_size_1, max_neighbors_1)
    kd2, quads2 = _make_tree_and_quads(xy2, min_size_2, max_size_2, max_neighbors_2)

    if verbose:
        print("Created", len(quads1), "quads from xy1")
        print("Created", len(quads2), "quads from xy2")
        print("")
        print("Comparing quads.")

    best_score = 0.0
    best_transform = None
    matches = 0
    for q1 in quads1:
        for q2 in quads2:
            if q1.is_match(q2, tolerance):
                fg_p = _fg_probability(kd1, kd2, q1.get_transform(q2), density)
                score = math.log(fg_p/density)
                if (score > best_score):
                    if verbose:
                        print("Match {0:d} score {1:.2f}".format(matches, score))
                    best_score = score
                    best_transform = q1.get_transform(q2)
                matches += 1

    if verbose:
        print("Found", matches, "matching quads")

    return [best_score, best_transform]


def _make_quad(A, B, C, D):
    """
    Returns a _MicroQuad if points A,B,C,D form a proper 
    quad, otherwise returns None.
    
    A, B define the coordinate system of the quad.
    C, D are the internal points.
    
    Parameters
    ----------
    A : (1, 2) ndarray
        A point in 2D.
    B : (1, 2) ndarray
        A point in 2D.
    C : (1, 2) ndarray
        A point in 2D.
    D : (1, 2) ndarray
        A point in 2D.
    
    Returns
    -------
    quad : _MicroQuad or None
        A _MicroQuad if the 4 points for a proper quad, otherwise
        None.
    """

    # Calculate scale.
    dab_x = B[0] - A[0]
    dab_y = B[1] - A[1]
    dab_l = math.sqrt(dab_x*dab_x + dab_y*dab_y)

    dab_l = 1.0/dab_l
    
    # Calculate circle center.
    cx = 0.5*(A[0]+B[0])
    cy = 0.5*(A[1]+B[1])

    # Calculate radius (squared).
    dx = A[0] - cx
    dy = A[1] - cy
    max_rr = dx*dx + dy*dy

    # Verify that C,B are within radius of the center point.
    for P in [C,D]:
        dx = P[0] - cx
        dy = P[1] - cy
        rr = dx*dx + dy*dy
        if (rr > max_rr):
            return

    # Calculate basis vectors.
    dab_x = dab_x * dab_l
    dab_y = dab_y * dab_l

    c45 = math.cos(0.25 * math.pi)
    s45 = math.sin(0.25 * math.pi)    
    x_vec = [c45 * dab_x + s45 * dab_y, -s45 * dab_x + c45*dab_y]
    y_vec = [c45 * dab_x - s45 * dab_y, s45 * dab_x + c45*dab_y]
    
    dab_l = math.sqrt(2.0) * dab_l
        
    # Calculate xc, yc.
    dac_x = dab_l * (C[0] - A[0])
    dac_y = dab_l * (C[1] - A[1])

    xc = x_vec[0] * dac_x + x_vec[1] * dac_y
    yc = y_vec[0] * dac_x + y_vec[1] * dac_y

    # Calcule xd, yd.
    dad_x = dab_l * (D[0] - A[0])
    dad_y = dab_l * (D[1] - A[1])

    xd = x_vec[0] * dad_x + x_vec[1] * dad_y
    yd = y_vec[0] * dad_x + y_vec[1] * dad_y
    
    if (xc > xd):
        return

    if ((xc + xd) > 1.0):
        return
    
    return _MicroQuad(A, B, C, D, xc, yc, xd, yd)


def _make_quads(kd, min_size, max_size, max_neighbors):
    """
    Construct MicroQuads.

    Note: In theory the run time of this algorithm is going to be
          proportional to the number of points times max_neighbors 
          to the 3rd power.

    Parameters
    ----------
    kd : scipy.spatial.KDTree object.
    min_size : float
        A, B points must be at least this distance from each
        other.
    max_size : float
        A, B points must be at most this distance from each 
        other.
    max_neighbors : int
        Only consider at most this many neighbors when
        constructing quads.

    Returns
    -------
    quads : A list of _MicroQuad
        A list of _MicroQuad to use for matching.
    """

    quads = []
    kd_data = kd.data

    #
    # Iterate over points in the tree to identify groups of
    # points and possibly make quads from them.
    #
    for i in range(kd_data.shape[0]):
        A = kd_data[i,:]

        # Add to max_neighbors as A will always have itself as a neighbor.
        if max_size is None:
            [dist, index] = kd.query(A, k = max_neighbors + 1)
        else:
            [dist, index] = kd.query(A, k = max_neighbors + 1, distance_upper_bound = max_size)

        #
        # Filter out points closer than the minimum distance.
        # Filter out points at infinite distance. I think these
        # are returned by KDTree when you specify both
        # 'max_neighbors' and 'distance_upper_bound'.
        #
        if min_size is None:
            mask = (dist > 1.0e-6) & (dist != np.inf)
        else:
            mask = (dist > min_size) & (dist != np.inf)

        dist = dist[mask]
        index = index[mask]

        # If we don't have at least 3 points proceed to the next A.
        if (index.size < 3):
            continue
        
        #
        # Iterate over all variations of the points in index
        # trying to create quads. Note that this is going to
        # be slow if there are lots of points to consider.
        #
        for j in index:
            B = kd_data[j,:]
            for k in index:
                if (k == j):
                    continue
                C = kd_data[k,:]
                for l in index:
                    if (l == j) or (l == k):
                        continue
                    D = kd_data[l,:]
                    quad = _make_quad(A, B, C, D)
                    if quad is not None:
                        quads.append(quad)
        
    return quads


def _make_tree_and_quads(xy, min_size, max_size, max_neighbors):
    """
    Make a KD tree and a list of quads from xy (2, N) points.

    Parameters
    ----------
    xy : (N, 2) ndarray.
        A ndarray of points to use to make a scipy.spatial.KDTree
        and a list of quads.
    min_size : float
        A, B points must be at least this distance from each
        other.
    max_size : float
        A, B points must be at most this distance from each 
        other.
    max_neighbors : int
        Only consider at most this many neighbors when
        constructing quads.

    Returns
    -------
    kd : scipy.spatial.KDTree object.
        A KDTree containing the xy points.    
    quads : List of _MicroQuad
        A list of _MicroQuad to use for matching.    
    """
    kd = sp.spatial.KDTree(xy)
    m_quads = _make_quads(kd,
                          min_size = min_size,
                          max_size = max_size,
                          max_neighbors = max_neighbors)
    return [kd, m_quads]


class _MicroQuad(object):
    """
    A geometric quad object, used to identify the mapping
    between the two sets of points.
    """
    def __init__(self, A, B, C, D, xc, yc, xd, yd):
        """
        Parameters
        ----------
        A : (1, 2) ndarray
            A point in 2D.
        B : (1, 2) ndarray
            A point in 2D.
        C : (1, 2) ndarray
            A point in 2D.
        D : (1, 2) ndarray
            A point in 2D.
        xc : float
            C point x position in A/B space.
        yc : float
            C point y position in A/B space.
        xd : float
            D point x position in A/B space.
        yd : float
            D point y position in A/B space.
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.xc = xc
        self.xd = xd
        self.yc = yc
        self.yd = yd

    def __str__(self):
        return "Quad {0:.3f} {1:.3f} {2:.3f} {3:.3f}".format(self.xc, self.yc, self.xd, self.yd)

    def get_transform(self, other):
        """
        Returns the transform to go from self space to other space.

        Parameters
        ----------
        other : _MicroQuad
            A _MicroQuad object.

        Returns
        -------
        trans : AffineTransform.
            An AffineTransform from self points to other points.
        """
        assert isinstance(other, _MicroQuad)

        src = np.array([self.A, self.B, self.C, self.D])
        dst = np.array([other.A, other.B, other.C, other.D])
        return AffineTransform.from_estimate(src, dst)
        
    def is_match(self, other, tolerance = 1.0e-2):
        """
        Returns True if two quads match each other.

        Parameters
        ----------
        other : _MicroQuad
            A _MicroQuad object.
        tolerance : float, optional
            The tolerance for matching the two quads to each other,
            default is 0.01.

        Returns
        -------
        match : bool
            True if the two quads match each other, otherwise False.
        """
        #
        # There are only two ways to match:
        #
        # 1. xc1 = xc2, yc1 = yc2, xd1 = xd1, yd1 = yd2
        #
        if (abs(self.xc - other.xc) < tolerance):
            if (abs(self.yc - other.yc) < tolerance):
                if (abs(self.xd - other.xd) < tolerance):
                    if (abs(self.yd - other.yd) < tolerance):
                        return True

        #
        # 2. xc1 = yc2, yc1 = xc2, xd1 = yd1, yd1 = xd2
        #
        if (abs(self.xc - other.yc) < tolerance):
            if (abs(self.yc - other.xc) < tolerance):
                if (abs(self.xd - other.yd) < tolerance):
                    if (abs(self.yd - other.xd) < tolerance):
                        return True

        return False
