import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import selem
import scipy.signal

def mask2ctr(mask):
    ctr = np.zeros(mask.shape, dtype=np.bool)
    for j in range(1, np.max(mask) + 1):
        dt = distance_transform_edt(mask == j)
        tmp_ctr = (dt == 1)
        tmp_ctr = binary_dilation(tmp_ctr, selem.disk(1))
        ctr[tmp_ctr] = True

    return ctr

def mask2ctr_1(mask):
    ctr = scipy.signal.convolve2d(mask, [[1, 1, 1], [1, -8, 1], [1, 1, 1]], mode='same')
    ctr = (ctr != 0)
    return ctr

def delineate(img, ctr):
    if img.shape[2] == 3:
        img_1 = img.copy()
    elif img.shape[2] == 1:
        img_1 = np.stack([img, img, img], 2)
    img_1[ctr, :] = [0, 255, 0]
    return img_1

from functools import reduce
def decorrstretch(A, tol=None):
    """
    Apply decorrelation stretch to image

    Arguments:
    A   -- image in cv2/numpy.array format
    tol -- upper and lower limit of contrast stretching
    """

    # save the original shape
    orig_shape = A.shape
    # reshape the image
    #         B G R
    # pixel 1 .
    # pixel 2   .
    #  . . .      .
    A = A.reshape((-1,3)).astype(np.float)
    # covariance matrix of A
    cov = np.cov(A.T)
    # source and target sigma
    sigma = np.diag(np.sqrt(cov.diagonal()))
    # eigen decomposition of covariance matrix
    eigval, V = np.linalg.eig(cov)
    # stretch matrix
    S = np.diag(1/np.sqrt(eigval))
    # compute mean of each color
    mean = np.mean(A, axis=0)
    # substract the mean from image
    A -= mean
    # compute the transformation matrix
    T = reduce(np.dot, [sigma, V, S, V.T])
    # compute offset
    offset = mean - np.dot(mean, T)
    # transform the image
    A = np.dot(A, T)
    # add the mean and offset
    A += mean + offset
    # restore original shape
    B = A.reshape(orig_shape)
    # for each color...
    for b in range(3):
        # apply contrast stretching if requested
        if tol:
            # find lower and upper limit for contrast stretching
            low, high = np.percentile(B[:,:,b], 100*tol), np.percentile(B[:,:,b], 100-100*tol)
            B[B<low] = low
            B[B>high] = high
        # ...rescale the color values to 0..255
        B[:,:,b] = 255 * (B[:,:,b] - B[:,:,b].min())/(B[:,:,b].max() - B[:,:,b].min())
    # return it as uint8 (byte) image
    return B.astype(np.uint8)

import xml.etree.ElementTree as ET

def get_annotation(filelist):
    annotations = []
    for j in filelist:
        anno = ET.parse(j)
        point = []
        for k in anno.getroot().find('image').find('overlays').findall('graphic'):
            if k.attrib['description'] in ['TIL-E', 'TIL-S']:
                tmp_type = 2
            elif k.attrib['description'] in ['normal', 'UDH', 'ADH']:
                tmp_type = 1
            elif k.attrib['description'] in ['IDC', 'ILC', 'Muc C', 'DCIS 1', 'DCIS 2', 'DCIS 3', 'MC- E', 'MC - C', 'MC - M']:
                tmp_type = 0
            else:
                continue
            for l in k.find('point-list'):
                y = int(l.text.split(',')[0])
                x = int(l.text.split(',')[1])
                point.append([x, y, tmp_type])
        annotations.append(np.array(point))
    annotations = np.array(annotations)
    return annotations

from sklearn.metrics import confusion_matrix

def sen_and_spe(y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)
    sen = conf[1, 1] / (conf[1, 1] + conf[1, 0])
    spe = conf[0, 0] / (conf[0, 0] + conf[0, 1])

    return sen, spe

def predprob(x, y, initial_lexsort=True):
    """
    Calculates the prediction probability. Adapted from scipy's implementation of Kendall's Tau

    Note: x should be the truth labels.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they will
        be flattened to 1-D.
    initial_lexsort : bool, optional
        Whether to use lexsort or quicksort as the sorting method for the
        initial sort of the inputs. Default is lexsort (True), for which
        `predprob` is of complexity O(n log(n)). If False, the complexity is
        O(n^2), but with a smaller pre-factor (so quicksort may be faster for
        small arrays).
    Returns
    -------
    Prediction probability : float

    Notes
    -----
    The definition of prediction probability that is used is::
      p_k = (((P - Q) / (P + Q + T)) + 1)/2
    where P is the number of concordant pairs, Q the number of discordant
    pairs, and T the number of ties only in `x`.
    References
    ----------
    Smith W.D, Dutton R.C, Smith N.T. (1996) A measure of association for assessing prediction accuracy
    that is a generalization of non-parametric ROC area. Stat Med. Jun 15;15(11):1199-215
    """

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if not x.size or not y.size:
        return (np.nan, np.nan)  # Return NaN if arrays are empty

    n = np.int64(len(x))
    temp = list(range(n))  # support structure used by mergesort
    # this closure recursively sorts sections of perm[] by comparing
    # elements of y[perm[]] using temp[] as support
    # returns the number of swaps required by an equivalent bubble sort

    def mergesort(offs, length):
        exchcnt = 0
        if length == 1:
            return 0
        if length == 2:
            if y[perm[offs]] <= y[perm[offs+1]]:
                return 0
            t = perm[offs]
            perm[offs] = perm[offs+1]
            perm[offs+1] = t
            return 1
        length0 = length // 2
        length1 = length - length0
        middle = offs + length0
        exchcnt += mergesort(offs, length0)
        exchcnt += mergesort(middle, length1)
        if y[perm[middle - 1]] < y[perm[middle]]:
            return exchcnt
        # merging
        i = j = k = 0
        while j < length0 or k < length1:
            if k >= length1 or (j < length0 and y[perm[offs + j]] <=
                                                y[perm[middle + k]]):
                temp[i] = perm[offs + j]
                d = i - j
                j += 1
            else:
                temp[i] = perm[middle + k]
                d = (offs + i) - (middle + k)
                k += 1
            if d > 0:
                exchcnt += d
            i += 1
        perm[offs:offs+length] = temp[0:length]
        return exchcnt

    # initial sort on values of x and, if tied, on values of y
    if initial_lexsort:
        # sort implemented as mergesort, worst case: O(n log(n))
        perm = np.lexsort((y, x))
    else:
        # sort implemented as quicksort, 30% faster but with worst case: O(n^2)
        perm = list(range(n))
        perm.sort(key=lambda a: (x[a], y[a]))

    # compute joint ties
    first = 0
    t = 0
    for i in range(1, n):
        if x[perm[first]] != x[perm[i]] or y[perm[first]] != y[perm[i]]:
            t += ((i - first) * (i - first - 1)) // 2
            first = i
    t += ((n - first) * (n - first - 1)) // 2

    # compute ties in x
    first = 0
    u = 0
    for i in range(1,n):
        if x[perm[first]] != x[perm[i]]:
            u += ((i - first) * (i - first - 1)) // 2
            first = i
    u += ((n - first) * (n - first - 1)) // 2

    # count exchanges
    exchanges = mergesort(0, n)
    # compute ties in y after mergesort with counting
    first = 0
    v = 0
    for i in range(1,n):
        if y[perm[first]] != y[perm[i]]:
            v += ((i - first) * (i - first - 1)) // 2
            first = i
    v += ((n - first) * (n - first - 1)) // 2

    tot = (n * (n - 1)) // 2
    if tot == u or tot == v:
        return 0.0 # (np.nan, np.nan)    # Special case for all ties in both ranks

    p_k = (((tot - (v + u - t)) - 2.0 * exchanges) / (tot - u) + 1)/2

    return p_k

from sklearn.utils import resample
from scipy.stats import kendalltau as tau

def predprob_ci(x, y, alpha=0.95, times=1000, seed=0):
    np.random.seed(seed)
    measure = []
    for _ in range(times):
        tmp_1, tmp_2 = resample(x, y)
        measure.append(predprob(tmp_1, tmp_2))
    measure = np.array(measure)
    lower = np.percentile(measure, ((1.0-alpha)/2.0) * 100)
    upper = np.percentile(measure, (alpha / 2.0 + 0.5) * 100)
    return predprob(x, y), lower, upper

def tau_ci(x, y, alpha=0.95, times=1000, seed=0):
    np.random.seed(seed)
    measure = []
    for _ in range(times):
        tmp_1, tmp_2 = resample(x, y)
        measure.append(tau(tmp_1, tmp_2)[0])
    measure = np.array(measure)
    lower = np.percentile(measure, ((1.0-alpha)/2.0) * 100)
    upper = np.percentile(measure, (alpha / 2.0 + 0.5) * 100)
    return tau(x, y)[0], lower, upper
