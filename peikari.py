import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import histomicstk as htk
import cv2
import matlab
from matlab import engine
from utils import mask2ctr, mask2ctr_1, delineate, decorrstretch
import scipy.ndimage as ndi
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_fill_holes, distance_transform_edt, binary_dilation
from skimage.measure import label
from skimage.morphology import local_maxima, watershed, reconstruction
from skimage.feature import peak_local_max
from skimage.morphology import selem
from skimage.measure import regionprops
from skimage.exposure import histogram
from sklearn.utils import resample, shuffle
import argparse, glob, os


eng = engine.start_matlab()

mean_ref = np.array([8.87064396, -0.05504628,  0.04430442])
std_ref = np.array([0.64901406, 0.10696055, 0.02916382])


def nuclei_seg(img, threshold=None):    # refine large blobs with regional thresholding
    '''
    implementation of the H&E stained histology image nuclei segmentation method described in
    [Peikari et al., 2016] and [Peikari et al., 2017]

    :param img: np.array, [MxNx3] uint8, input RGB image
    :param threshold: array-like, [2] float, pre-computed thresholds
    :return: final_mask: np.array, [MxNx3] int16, segmentation mask
    th_1, th_2: float, thresholds

    references:
    [1] Peikari, M., Salama, S., Nofech‐Mozes, S., & Martel, A. L. (2017).
    Automatic cellularity assessment from post‐treated breast surgical specimens.
    Cytometry Part A, 91(11), 1078-1087.

    [2] Peikari, M., & Martel, A. L. (2016, March). Automatic cell detection and segmentation from H and E
    stained pathology slides using colorspace decorrelation stretching. In Medical Imaging 2016: Digital Pathology
    (Vol. 9791, p. 979114). International Society for Optics and Photonics.
    '''
    im_nmzd = htk.preprocessing.color_normalization.reinhard(img, mean_ref, std_ref)
    img_ds = decorrstretch(im_nmzd)
    img_lab = htk.preprocessing.color_conversion.rgb_to_lab(img_ds)
    img_lab = 0.25 * img_lab[:,:,0] + 0.5 * img_lab[:,:,1] + 0.25 * img_lab[:, :, 2]
    if threshold is None:
        th = eng.multithresh(matlab.double(img_lab.tolist()), 2)
        th_1 = np.sort(th)[0][0]
        th_2 = np.sort(th)[0][1]
    else:
        th_1 = threshold[0]
        th_2 = threshold[1]

    mask = (img_lab < th_1)
    mask = binary_fill_holes(mask)
    mask = binary_opening(mask, selem.disk(3))

    # correct for mis-filled holes
    mask_holes = (mask & (img_lab > th_2))
    mask_holes = (htk.segmentation.label.area_open(label(mask_holes, connectivity=1), 40) > 0)
    mask_holes = (binary_dilation(mask_holes, selem.disk(2)))
    mask[mask_holes] = False
    mask_conn = label(mask, connectivity=1)

    # refine large blobs
    area = np.array([[_.label, _.area] for _ in regionprops(mask_conn)])
    overlap_label = area[area[:, 1] > 300, 0]
    overlap_mask = np.isin(mask_conn, overlap_label)
    dt = distance_transform_edt(overlap_mask)
    local_maxi = peak_local_max(dt, labels=mask_conn, min_distance=4, indices=False)
    markers = label(local_maxi)
    overlap_seg = watershed(-dt, markers, mask=overlap_mask)
    mask_conn[overlap_seg > 0] = overlap_seg[overlap_seg > 0] + mask_conn.max()
    final_mask = htk.segmentation.label.condense(mask_conn)

    # refine large blobs by regional threshold
    area = np.array([[_.label, _.area,] for _ in regionprops(final_mask)])
    overlap_label = area[area[:, 1] > 300, 0]
    overlap_mask = np.isin(final_mask, overlap_label)
    final_mask[overlap_mask] = 0
    tmp_th = eng.multithresh(matlab.double([img_lab[overlap_mask].tolist()]), 2)
    th_3 = np.sort(tmp_th)[0][1]
    overlap_mask = overlap_mask & (img_lab < th_3)
    overlap_mask = binary_opening(overlap_mask, selem.disk(3))

    dt = distance_transform_edt(overlap_mask)
    local_maxi = peak_local_max(dt, labels=overlap_mask, min_distance=4, indices=False)
    markers = label(local_maxi)
    overlap_seg = watershed(-dt, markers, mask=overlap_mask)
    final_mask[overlap_seg > 0] = overlap_seg[overlap_seg > 0] + final_mask.max()

    # remove boundary regions
    for _1 in regionprops(final_mask):
        min_row, min_col, max_row, max_col = _1.bbox
        if min_row <= 0 or min_col <= 0 or max_row >= img.shape[0] or max_col >= img.shape[1]:
            final_mask[final_mask == _1.label] = 0
    final_mask = htk.segmentation.label.condense(final_mask)
    final_mask = _label_fill_holes(final_mask)

    final_mask = htk.segmentation.label.area_open(final_mask, 40).astype('int16')

    return final_mask, th_1, th_2

def _label_fill_holes(im_label):
    im_label_out = np.copy(im_label)
    for j in regionprops(im_label):
        mask = (im_label == j.label)
        mask = binary_fill_holes(mask)
        im_label_out[mask] = j.label
    return im_label_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, nargs='+', default=['cells', 'val', 'corr'],
                        help="the phase of which the images will be segmented, should be among 'cells', 'val', 'corr', "
                             "multiple phases are supported")
    parser.add_argument('--precomputed_threshold', action='store_true',
                        help='whether to use the precomputed thresholds stored in the directory of each phase to '
                             'speedup segmentation')
    args = parser.parse_args()

    assert np.all([_ in ['cells', 'val', 'corr'] for _ in args.phase])

    for PHASE in args.phase:    # nuclei seg for cells and val
        if PHASE == 'cells':
            filelist = glob.glob('data/cells/*.tif')
            filelist.sort()
            x = np.array([cv2.cvtColor(cv2.imread(_), cv2.COLOR_BGR2RGB) for _ in filelist])
        else:
            csv = pd.read_csv('data/{}_labels.csv'.format(PHASE))
            x = np.array([cv2.cvtColor(cv2.imread('data/{}/{:.0f}_{:.0f}.tif'.format(PHASE, j['slide'], j['rid'])),
                                       cv2.COLOR_BGR2RGB) for _, j in csv.iterrows()])
        if args.precomputed_threshold:
            thresh = np.load('data/{}/threshold.npy'.format(PHASE))
        else:
            thresh = [None] * len(x)
        plt.rcParams['image.cmap'] = 'gray'

        final_seg = []
        threshold = []
        for i, (img, th) in enumerate(zip(x, thresh)):
            final_mask, th_1, th_2 = nuclei_seg(img, th)
            final_seg.append(final_mask)
            threshold.append([th_1, th_2])
            print('Segmenting image {:05d}/{:d} of phase {}...'.format(i, len(x), PHASE))
        final_seg = np.array(final_seg)
        threshold = np.array(threshold)
        if not os.path.exists('segmentation&classification/{}'.format(PHASE)):
            os.mkdir('segmentation&classification/{}'.format(PHASE))
        np.save('segmentation&classification/{}/seg.npy'.format(PHASE), final_seg)
        print('Saving segmentation of phase {} to file \'segmentation&classification/{}/seg.npy\''.format(PHASE, PHASE))
    eng.exit()