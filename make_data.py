import sys
sys.path.append('.')

import numpy as np
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import warnings
import skimage.io
import cv2
import histomicstk as htk
import matlab, matlab.engine
import argparse, glob, pandas as pd, os
from utils import get_annotation


parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str,
                    help="the action, should be among 'stack', 'get_annotation', 'get_centroid', 'match', "
                         "'mark_countour', 'extract_nuclei&intensity', 'extract_shape&lbp', 'extract_haralick', "
                         "only one action is supported each time")
parser.add_argument('--phase', type=str, default='None',
                    help="the phase of which the features will be extracted, should be among 'cells', 'val', 'corr', "
                         "only one phase is supported each time")
args = parser.parse_args()
assert args.opt in ['stack', 'get_annotation', 'get_centroid', 'match', 'mark_countour', 'extract_nuclei&intensity',
                    'extract_shape&lbp', 'extract_haralick']
assert args.phase in ['cells', 'val', 'corr', 'None']

if args.opt == 'stack':    # stack images into one file to speedup loading
    if args.phase == 'cells':
        filelist = glob.glob('data/cells/*_crop.tif')
        filelist.sort()
        x = np.array([cv2.cvtColor(cv2.imread(_), cv2.COLOR_BGR2RGB) for _ in filelist])
    else:
        csv = pd.read_csv('data/{}_labels.csv'.format(args.phase))
        x = np.array([cv2.cvtColor(cv2.imread('data/{}/{:.0f}_{:.0f}.tif'.format(args.phase, j['slide'], j['rid'])),
                                   cv2.COLOR_BGR2RGB) for _, j in csv.iterrows()])
        np.save('data/{}/y.npy'.format(args.phase), np.array(csv['y'].tolist()))
    np.save('data/{}/x.npy'.format(args.phase), x)

elif args.opt == 'get_annotation':
    filelist = glob.glob('data/cells/Sedeen/*.session.xml')
    filelist.sort()
    annotations = get_annotation(filelist)
    if not os.path.exists('segmentation&classification/cells'):
        os.mkdir('segmentation&classification/cells')
    np.save('segmentation&classification/cells/annotation.npy', annotations)

elif args.opt == 'get_centroid':
    seg = np.load('segmentation&classification/{}/seg.npy'.format(args.phase))

    def _add_data(region_prop):
        tmp = []
        for j in region_prop:
            if j.major_axis_length / j.minor_axis_length < 3 and j.area < 1200:
                tmp.append(j.centroid + (j.label,))
        tmp = np.array(tmp)
        return tmp

    centroid = np.array([_add_data(regionprops(j)) for j in seg])
    if not os.path.exists('segmentation&classification/{}'.format(args.phase)):
        os.mkdir('segmentation&classification/{}'.format(args.phase))
    np.save('segmentation&classification/{}/centroid.npy'.format(args.phase), centroid)

elif args.opt == 'match':
    annotations = np.load('segmentation&classification/cells/annotation.npy')
    centroids = np.load('segmentation&classification/cells/centroid.npy')

    anno_match = []
    for k in range(len(centroids)):
        annotation = annotations[k]
        centroid = centroids[k]

        x0, x1 = np.meshgrid(centroid[:, 0], annotation[:, 0])
        y0, y1 = np.meshgrid(centroid[:, 1], annotation[:, 1])
        dist = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        is_row_min = np.array([j == np.min(dist, 0) for j in dist])
        is_col_min = np.array([j == np.min(dist, 1) for j in dist.T]).T
        is_lesser_10 = (dist < 10)
        match_mask = is_row_min & is_col_min & is_lesser_10
        # check duplicate
        if np.sum(np.sum(match_mask, 0)>1)>0 or np.sum(np.sum(match_mask, 1)>1)>0:
            warnings.warn('dup found! k=%d'%k)
            match_mask[:, np.where(np.sum(match_mask, 0) > 1)[0]] = False
            match_mask[np.where(np.sum(match_mask, 1) > 1)[0]] = False
        coor = np.array(np.where(match_mask))
        anno_match_tmp = np.concatenate([annotation[coor[0], :2], centroid[coor[1], :2],
                                         annotation[coor[0], 2:3], centroid[coor[1], 2:3]], 1)
        anno_match.append(anno_match_tmp)
    anno_match = np.array(anno_match)
    if not os.path.exists('segmentation&classification/cells'):
        os.mkdir('segmentation&classification/cells')
    np.save('segmentation&classification/cells/anno_match.npy', anno_match)

elif args.opt == 'mark_countour':
    from utils import *
    if args.phase == 'cells':
        x = np.load('data/cells/x.npy')
        seg = np.load('segmentation&classification/cells/seg.npy')
        anno_match = np.load('segmentation&classification/cells/anno_match.npy')
        x_contour = []
        for j, k, m in zip(x, anno_match, seg):
            m[np.logical_not(np.isin(m, k[:, 5]))] = 0
            tmp = delineate(j, mask2ctr_1(m))
            for l in k:
                if l[4] == 0:
                    cv2.circle(tmp, tuple(l[1::-1].astype('int16')), 4, (255, 0, 0), -1)
                elif l[4] == 1:
                    cv2.circle(tmp, tuple(l[1::-1].astype('int16')), 4, (0, 0, 255), -1)
                elif l[4] == 2:
                    cv2.circle(tmp, tuple(l[1::-1].astype('int16')), 4, (255, 255, 0), -1)
            x_contour.append(tmp)
        x_contour = np.array(x_contour)
    else:
        x = np.load('data/{}/x.npy'.format(args.phase))
        seg = np.load('segmentation&classification/{}/seg.npy'.format(args.phase))
        centroid = np.load('segmentation&classification/{}/centroid.npy'.format(args.phase))
        x_contour = []
        for j, k, m in zip(x, centroid, seg):
            m[np.logical_not(np.isin(m, k[:, 2]))] = 0
            tmp = delineate(j, mask2ctr_1(m))
            x_contour.append(tmp)
        x_contour = np.array(x_contour)

    if not os.path.exists('segmentation&classification/{}/seg_contour'.format(args.phase)):
        os.mkdir('segmentation&classification/{}/seg_contour'.format(args.phase))
    for i, j in enumerate(x_contour):
        skimage.io.imsave('segmentation&classification/{}/seg_contour/{:03d}.tif'.format(args.phase, i), j)

elif args.opt == 'extract_nuclei&intensity':
    from histomicstk.features import *

    mean_ref = np.array([8.87064396, -0.05504628, 0.04430442])
    std_ref = np.array([0.64901406, 0.10696055, 0.02916382])

    if args.phase == 'cells':
        x = np.load('data/cells/x.npy')
        x = np.array([htk.preprocessing.color_normalization.reinhard(j, mean_ref, std_ref) for j in x])
        final_seg = np.load('segmentation&classification/cells/seg.npy')
        anno_match = np.load('segmentation&classification/cells/anno_match.npy')
        nuclei37 = []
        inten6 = []
        for j, k, l in zip(x, anno_match, final_seg):
            img = cv2.cvtColor(j, cv2.COLOR_RGB2GRAY)
            img = img.astype('float64')
            img -= img.mean()
            img /= img.std()
            # compute nuclei37
            feat = compute_nuclei_features(l, img, haralick_features_flag=False)
            tmp_nuclei37 = feat.loc[k[:, 5].astype('int16') - 1]
            tmp_nuclei37 = tmp_nuclei37.values
            tmp_nuclei37 = np.concatenate([k[:, 4:5], tmp_nuclei37], 1)
            # compute inten6
            tmp_inten6 = feat.loc[k[:, 5].astype('int16') - 1, ['Nucleus.Intensity.Mean', 'Nucleus.Intensity.Mode',
                                                                'Nucleus.Intensity.Median', 'Nucleus.Intensity.Std',
                                                                'Nucleus.Intensity.Skewness',
                                                                'Nucleus.Intensity.Kurtosis']]
            tmp_inten6 = tmp_inten6.values
            tmp_inten6 = np.concatenate([k[:, 4:5], tmp_inten6], 1)

            nuclei37.append(tmp_nuclei37)
            inten6.append(tmp_inten6)
        nuclei37 = np.array(nuclei37)
        inten6 = np.array(inten6)

    else:
        x = np.load('data/{}/x.npy'.format(args.phase))
        x = np.array([htk.preprocessing.color_normalization.reinhard(j, mean_ref, std_ref) for j in x])
        final_seg = np.load('segmentation&classification/{}/seg.npy'.format(args.phase))
        centroid = np.load('segmentation&classification/{}/centroid.npy'.format(args.phase))
        nuclei37 = []
        inten6 = []
        for j, k, l in zip(x, centroid, final_seg):
            img = cv2.cvtColor(j, cv2.COLOR_RGB2GRAY)
            img = img.astype('float64')
            img -= img.mean()
            img /= img.std()
            # compute nuclei37
            feat = compute_nuclei_features(l, img, haralick_features_flag=False)
            tmp_nuclei37 = feat.loc[k[:, 2].astype('int16') - 1]
            tmp_nuclei37 = tmp_nuclei37.values
            # compute inten6
            tmp_inten6 = feat.loc[k[:, 2].astype('int16') - 1, ['Nucleus.Intensity.Mean', 'Nucleus.Intensity.Mode',
                                                                'Nucleus.Intensity.Median', 'Nucleus.Intensity.Std',
                                                                'Nucleus.Intensity.Skewness',
                                                                'Nucleus.Intensity.Kurtosis']]
            tmp_inten6 = tmp_inten6.values

            nuclei37.append(tmp_nuclei37)
            inten6.append(tmp_inten6)
        nuclei37 = np.array(nuclei37)
        inten6 = np.array(inten6)

    if not os.path.exists('segmentation&classification/{}/features'.format(args.phase)):
        os.mkdir('segmentation&classification/{}/features'.format(args.phase))
    np.save('segmentation&classification/{}/features/nuclei37.npy'.format(args.phase), nuclei37)
    np.save('segmentation&classification/{}/features/inten6.npy'.format(args.phase), inten6)

elif args.opt == 'extract_shape&lbp':
    from histomicstk.features import *
    from skimage.feature import local_binary_pattern

    mean_ref = np.array([8.87064396, -0.05504628, 0.04430442])
    std_ref = np.array([0.64901406, 0.10696055, 0.02916382])

    if args.phase == 'cells':
        x = np.load('data/cells/x.npy')
        x = np.array([htk.preprocessing.color_normalization.reinhard(j, mean_ref, std_ref) for j in x])
        final_seg = np.load('segmentation&classification/cells/seg.npy')
        anno_match = np.load('segmentation&classification/cells/anno_match.npy')
        centroid = np.load('segmentation&classification/cells/centroid.npy')
        shape7 = []
        lbp10 = []
        for j, k, l, m in zip(x, anno_match, final_seg, centroid):
            img = cv2.cvtColor(j, cv2.COLOR_RGB2GRAY)
            img = img.astype('float64')
            img -= img.mean()
            img /= img.std()
            feat = compute_nuclei_features(l, img, fsd_features_flag=False,
                                           intensity_features_flag=False, gradient_features_flag=False,
                                           haralick_features_flag=False)
            lbp_map = local_binary_pattern(img, 8, 1, method='uniform')

            tmp_shape7 = []
            tmp_lbp10 = []
            for anno in k:
                idx = (m[(np.abs(m[:, 0] - anno[0]) <= 125) & (np.abs(m[:, 1] - anno[1]) <= 125), 2] - 1).astype(
                    'int16')
                f = np.array([feat.loc[idx, 'Size.Area'].values, feat.loc[idx, 'Size.Perimeter'].values,
                              feat.loc[idx, 'Shape.Eccentricity'].values]).T
                f_mean = np.mean(f, 0)
                f_std = np.std(f, 0)
                shape_factor = feat.loc[anno[5].astype('int16') - 1, 'Size.Perimeter'] / \
                               (2 * np.sqrt(feat.loc[anno[5].astype('int16') - 1, 'Size.Area']))

                lbp_feat = np.array([np.sum(lbp_map[l == anno[5]] == n) for n in range(10)]).astype('float64')
                lbp_feat /= lbp_feat.sum()

                tmp_shape7.append(np.concatenate([anno[4:5], f_mean, f_std, [shape_factor]], 0))
                tmp_lbp10.append(np.concatenate([anno[4:5], lbp_feat], 0))
            shape7.append(np.array(tmp_shape7))
            lbp10.append(np.array(tmp_lbp10))
        shape7 = np.array(shape7)
        lbp10 = np.array(lbp10)

    else:
        x = np.load('data/{}/x.npy'.format(args.phase))
        x = np.array([htk.preprocessing.color_normalization.reinhard(j, mean_ref, std_ref) for j in x])
        final_seg = np.load('segmentation&classification/{}/seg.npy'.format(args.phase))
        centroid = np.load('segmentation&classification/{}/centroid.npy'.format(args.phase))
        shape7 = []
        lbp10 = []
        for j, l, m in zip(x, final_seg, centroid):
            img = cv2.cvtColor(j, cv2.COLOR_RGB2GRAY)
            img = img.astype('float64')
            img -= img.mean()
            img /= img.std()
            feat = compute_nuclei_features(l, img, fsd_features_flag=False,
                                           intensity_features_flag=False, gradient_features_flag=False,
                                           haralick_features_flag=False)
            lbp_map = local_binary_pattern(img, 8, 1, method='uniform')

            tmp_shape7 = []
            tmp_lbp10 = []
            for anno in m:
                idx = (m[(np.abs(m[:, 0] - anno[0]) <= 125) & (np.abs(m[:, 1] - anno[1]) <= 125), 2] - 1).astype(
                    'int16')
                f = np.array([feat.loc[idx, 'Size.Area'].values, feat.loc[idx, 'Size.Perimeter'].values,
                              feat.loc[idx, 'Shape.Eccentricity'].values]).T
                f_mean = np.mean(f, 0)
                f_std = np.std(f, 0)
                shape_factor = feat.loc[anno[2].astype('int16') - 1, 'Size.Perimeter'] / \
                               (2 * np.sqrt(feat.loc[anno[2].astype('int16') - 1, 'Size.Area']))

                lbp_feat = np.array([np.sum(lbp_map[l == anno[2]] == n) for n in range(10)]).astype('float64')
                lbp_feat /= lbp_feat.sum()

                tmp_shape7.append(np.concatenate([f_mean, f_std, [shape_factor]], 0))
                tmp_lbp10.append(lbp_feat)
            shape7.append(np.array(tmp_shape7))
            lbp10.append(np.array(tmp_lbp10))
        shape7 = np.array(shape7)
        lbp10 = np.array(lbp10)

    if not os.path.exists('segmentation&classification/{}/features'.format(args.phase)):
        os.mkdir('segmentation&classification/{}/features'.format(args.phase))
    np.save('segmentation&classification/{}/features/nuclei37.npy'.format(args.phase), shape7)
    np.save('segmentation&classification/{}/features/inten6.npy'.format(args.phase), lbp10)

elif args.opt == 'extract_haralick':
    mean_ref = np.array([8.87064396, -0.05504628, 0.04430442])
    std_ref = np.array([0.64901406, 0.10696055, 0.02916382])

    if args.phase == 'cells':
        x = np.load('data/cells/x.npy')
        x = np.array([htk.preprocessing.color_normalization.reinhard(j, mean_ref, std_ref) for j in x])
        final_seg = np.load('segmentation&classification/cells/seg.npy')
        anno_match = np.load('segmentation&classification/cells/anno_match.npy')
        hara22 = []
        hara88 = []
        eng = matlab.engine.start_matlab()
        for j, k, l in zip(x, anno_match, final_seg):
            img = cv2.cvtColor(j, cv2.COLOR_RGB2GRAY)
            img = img.astype('float64')
            img -= img.mean()
            img /= img.std()
            region_prop = regionprops(l)
            tmp_hara22 = []
            tmp_hara88 = []
            for anno in k:
                minr, minc, maxr, maxc = region_prop[anno[5].astype('int16') - 1].bbox
                subImage = img[minr:maxr + 1, minc:maxc + 1]
                glcm = eng.graycomatrix(matlab.double(subImage.tolist()),
                                        'Offset', matlab.int8([[0, 1], [-1, 1], [-1, 0], [-1, -1]]),
                                        'Symmetric', True, 'NumLevels', 32, 'GrayLimits', matlab.double([]))
                # compute isotropical haralick
                glcm_iso = np.asarray(glcm)
                glcm_iso = np.sum(glcm_iso, 2, keepdims=True)
                feat_iso = eng.GLCM_Features4(matlab.double(glcm_iso.tolist()), 0)
                tmp_hara22.append([anno[4],
                                   np.mean(feat_iso['autoc']), np.mean(feat_iso['contr']), np.mean(feat_iso['corrm']),
                                   np.mean(feat_iso['corrp']),
                                   np.mean(feat_iso['cprom']), np.mean(feat_iso['cshad']), np.mean(feat_iso['denth']),
                                   np.mean(feat_iso['dissi']),
                                   np.mean(feat_iso['dvarh']), np.mean(feat_iso['energ']), np.mean(feat_iso['entro']),
                                   np.mean(feat_iso['homom']),
                                   np.mean(feat_iso['homop']), np.mean(feat_iso['idmnc']), np.mean(feat_iso['indnc']),
                                   np.mean(feat_iso['inf1h']),
                                   np.mean(feat_iso['inf2h']), np.mean(feat_iso['maxpr']), np.mean(feat_iso['savgh']),
                                   np.mean(feat_iso['senth']),
                                   np.mean(feat_iso['sosvh']), np.mean(feat_iso['svarh'])])
                # compute anisotropical haralick
                feat_aniso = eng.GLCM_Features4(glcm, 0)
                tmp_hara88.append(np.squeeze(np.concatenate([[anno[4:5]],
                                                             feat_aniso['autoc'], feat_aniso['contr'],
                                                             feat_aniso['corrm'], feat_aniso['corrp'],
                                                             feat_aniso['cprom'], feat_aniso['cshad'],
                                                             feat_aniso['denth'], feat_aniso['dissi'],
                                                             feat_aniso['dvarh'], feat_aniso['energ'],
                                                             feat_aniso['entro'], feat_aniso['homom'],
                                                             feat_aniso['homop'], feat_aniso['idmnc'],
                                                             feat_aniso['indnc'], feat_aniso['inf1h'],
                                                             feat_aniso['inf2h'], feat_aniso['maxpr'],
                                                             feat_aniso['savgh'], feat_aniso['senth'],
                                                             feat_aniso['sosvh'], feat_aniso['svarh']], 1)))
            hara22.append(np.array(tmp_hara22))
            hara88.append(np.array(tmp_hara88))
        hara22 = np.array(hara22)
        hara88 = np.array(hara88)
        eng.exit()

    else:
        x = np.load('data/{}/x.npy'.format(args.phase))
        x = np.array([htk.preprocessing.color_normalization.reinhard(j, mean_ref, std_ref) for j in x])
        final_seg = np.load('segmentation&classification/{}/seg.npy'.format(args.phase))
        centroid = np.load('segmentation&classification/{}/centroid.npy'.format(args.phase))
        hara22 = []
        hara88 = []
        eng = matlab.engine.start_matlab()
        for j, k, l in zip(x, centroid, final_seg):
            img = cv2.cvtColor(j, cv2.COLOR_RGB2GRAY)
            img = img.astype('float64')
            img -= img.mean()
            img /= img.std()
            region_prop = regionprops(l)
            tmp_hara22 = []
            tmp_hara88 = []
            for anno in k:
                minr, minc, maxr, maxc = region_prop[anno[2].astype('int16') - 1].bbox
                subImage = img[minr:maxr + 1, minc:maxc + 1]
                glcm = eng.graycomatrix(matlab.double(subImage.tolist()),
                                        'Offset', matlab.int8([[0, 1], [-1, 1], [-1, 0], [-1, -1]]),
                                        'Symmetric', True, 'NumLevels', 32, 'GrayLimits', matlab.double([]))
                # compute isotropical haralick
                glcm_iso = np.asarray(glcm)
                glcm_iso = np.sum(glcm_iso, 2, keepdims=True)
                feat_iso = eng.GLCM_Features4(matlab.double(glcm_iso.tolist()), 0)
                tmp_hara22.append([
                    np.mean(feat_iso['autoc']), np.mean(feat_iso['contr']), np.mean(feat_iso['corrm']),
                    np.mean(feat_iso['corrp']),
                    np.mean(feat_iso['cprom']), np.mean(feat_iso['cshad']), np.mean(feat_iso['denth']),
                    np.mean(feat_iso['dissi']),
                    np.mean(feat_iso['dvarh']), np.mean(feat_iso['energ']), np.mean(feat_iso['entro']),
                    np.mean(feat_iso['homom']),
                    np.mean(feat_iso['homop']), np.mean(feat_iso['idmnc']), np.mean(feat_iso['indnc']),
                    np.mean(feat_iso['inf1h']),
                    np.mean(feat_iso['inf2h']), np.mean(feat_iso['maxpr']), np.mean(feat_iso['savgh']),
                    np.mean(feat_iso['senth']),
                    np.mean(feat_iso['sosvh']), np.mean(feat_iso['svarh'])])
                # compute anisotropical haralick
                feat_aniso = eng.GLCM_Features4(glcm, 0)
                tmp_hara88.append(np.squeeze(np.concatenate([
                    feat_aniso['autoc'], feat_aniso['contr'], feat_aniso['corrm'], feat_aniso['corrp'],
                    feat_aniso['cprom'], feat_aniso['cshad'], feat_aniso['denth'], feat_aniso['dissi'],
                    feat_aniso['dvarh'], feat_aniso['energ'], feat_aniso['entro'], feat_aniso['homom'],
                    feat_aniso['homop'], feat_aniso['idmnc'], feat_aniso['indnc'], feat_aniso['inf1h'],
                    feat_aniso['inf2h'], feat_aniso['maxpr'], feat_aniso['savgh'], feat_aniso['senth'],
                    feat_aniso['sosvh'], feat_aniso['svarh']], 1)))
            hara22.append(np.array(tmp_hara22))
            hara88.append(np.array(tmp_hara88))
        hara22 = np.array(hara22)
        hara88 = np.array(hara88)
        eng.exit()

    if not os.path.exists('segmentation&classification/{}/features'.format(args.phase)):
        os.mkdir('segmentation&classification/{}/features'.format(args.phase))
    np.save('segmentation&classification/{}/features/nuclei37.npy'.format(args.phase), hara22)
    np.save('segmentation&classification/{}/features/inten6.npy'.format(args.phase), hara88)
