import sys
sys.path.append('.')

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt, seaborn as sns
import pickle as pkl
from utils import sen_and_spe
from skimage.morphology import selem, binary_dilation
from sklearn.linear_model import LinearRegression
import pandas as pd, argparse, os, skimage.io
from sklearn.neighbors import KNeighborsClassifier
from utils import *


sns.set_style('whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str,
                    help="the action of nuclei classification, should be among 'lvsbm', 'bvsm', 'validate', 'predict', "
                         "only one action is supported each time")
args = parser.parse_args()
assert args.opt in ['lvsbm', 'bvsm', 'validate', 'predict']

if args.opt == 'lvsbm':    # train SVM L vs. BM image-wise split
    split_point = 110

    # generate training data for cells
    x_cells_train = np.concatenate([
        np.concatenate(np.load('segmentation&classification/cells/features/nuclei37.npy')[:split_point], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/shape7.npy')[:split_point], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/lbp10.npy')[:split_point], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/hara88.npy')[:split_point], 0)[:, 1:]
    ], 1)
    x_cells_train[np.isnan(x_cells_train)] = 0
    y_cells_train_ = np.concatenate(
        np.load('segmentation&classification/cells/features/shape7.npy')[:split_point],
        0)[:, 0].astype('uint8')
    y_cells_train = y_cells_train_.copy()
    y_cells_train[np.isin(y_cells_train_, [0, 1])] = 0
    y_cells_train[np.isin(y_cells_train_, [2])] = 1

    # generate test data for cells
    x_cells_test = np.concatenate([
        np.concatenate(np.load('segmentation&classification/cells/features/nuclei37.npy')[split_point:], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/shape7.npy')[split_point:], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/lbp10.npy')[split_point:], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/hara88.npy')[split_point:], 0)[:, 1:]
    ], 1)
    x_cells_test[np.isnan(x_cells_test)] = 0
    y_cells_test_ = np.concatenate(
        np.load('segmentation&classification/cells/features/shape7.npy')[split_point:],
                       0)[:, 0].astype('uint8')
    y_cells_test = y_cells_test_.copy()
    y_cells_test[np.isin(y_cells_test_, [0, 1])] = 0
    y_cells_test[np.isin(y_cells_test_, [2])] = 1

    # scale data
    scaler =  MinMaxScaler((-1, 1))
    scaler.fit(x_cells_train)
    x_cells_train = scaler.transform(x_cells_train)
    x_cells_test = scaler.transform(x_cells_test)

    # fit and save model
    model = SVC(C=100, gamma=0.01,)
    model.fit(x_cells_train, y_cells_train, )
    if not os.path.exists('segmentation&classification/model'):
        os.mkdir('segmentation&classification/model')
    with open('segmentation&classification/model/svm-lvsbm.pkl', 'wb') as f:
        pkl.dump(model, f)
    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_cells_cv = []
    y_pred_cells_cv = []
    for train_index, test_index in skf.split(x_cells_train, y_cells_train):
        X_train, X_test = x_cells_train[train_index], x_cells_train[test_index]
        y_train, y_test = y_cells_train[train_index], y_cells_train[test_index]
        y_cells_cv.append(y_test)
        model_cv = SVC(C=100, gamma=0.01, )
        model_cv.fit(X_train, y_train)
        y_pred_cells_cv.append(model_cv.decision_function(X_test))
    y_cells_cv = np.concatenate(y_cells_cv, 0)
    y_pred_cells_cv = np.concatenate(y_pred_cells_cv, 0)
    # plot ROC based on 5fold CV
    fpr, tpr, th = roc_curve(y_cells_cv, y_pred_cells_cv, drop_intermediate=False)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='b', label='Our ROC curve (AUC={:.2f})'.format(roc_auc_score(y_cells_cv, y_pred_cells_cv)), lw=3)
    plt.scatter(0.01, 0.79, color='r', s=100, label='Results reported by Peikari et al.')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=3)
    plt.xticks(np.arange(0, 1.01, 0.2), fontsize=20)
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=20)
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.xlabel('1-Specificity', fontsize=20)
    plt.ylabel('Sensitivity', fontsize=20)
    plt.legend(loc='lower right', fontsize=15)
    plt.title('L vs. BM', fontsize=20)
    plt.savefig('segmentation&classification/model/roc-lvsbm.png')
    pass

elif args.opt == 'bvsm':    # train SVM B vs. M image-wise split
    split_point = 110

    # generate training data for cells
    x_cells_train = np.concatenate([
        np.concatenate(np.load('segmentation&classification/cells/features/nuclei37.npy')[:split_point], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/shape7.npy')[:split_point], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/lbp10.npy')[:split_point], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/hara88.npy')[:split_point], 0)[:, 1:]
    ], 1)
    x_cells_train[np.isnan(x_cells_train)] = 0
    y_cells_train_ = np.concatenate(
        np.load('segmentation&classification/cells/features/shape7.npy')[:split_point],
        0)[:, 0].astype('uint8')
    x_cells_train = x_cells_train[np.isin(y_cells_train_, [0, 1])]
    y_cells_train = y_cells_train_[np.isin(y_cells_train_, [0, 1])]

    # generate test data for cells
    x_cells_test = np.concatenate([
        np.concatenate(np.load('segmentation&classification/cells/features/nuclei37.npy')[split_point:], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/shape7.npy')[split_point:], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/lbp10.npy')[split_point:], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/hara88.npy')[split_point:], 0)[:, 1:]
    ], 1)
    x_cells_test[np.isnan(x_cells_test)] = 0
    y_cells_test_ = np.concatenate(
        np.load('segmentation&classification/cells/features/shape7.npy')[split_point:],
                       0)[:, 0].astype('uint8')
    x_cells_test = x_cells_test[np.isin(y_cells_test_, [0, 1])]
    y_cells_test = y_cells_test_[np.isin(y_cells_test_, [0, 1])]

    # scale data
    scaler =  MinMaxScaler((-1, 1))
    scaler.fit(x_cells_train)
    x_cells_train = scaler.transform(x_cells_train)
    x_cells_test = scaler.transform(x_cells_test)

    # fit model
    model = SVC(C=100, gamma=0.01,)
    model.fit(x_cells_train, y_cells_train, )
    if not os.path.exists('segmentation&classification/model'):
        os.mkdir('segmentation&classification/model')
    with open('segmentation&classification/model/svm-bvsm.pkl', 'wb') as f:
        pkl.dump(model, f)
    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_cells_cv = []
    y_pred_cells_cv = []
    for train_index, test_index in skf.split(x_cells_train, y_cells_train):
        X_train, X_test = x_cells_train[train_index], x_cells_train[test_index]
        y_train, y_test = y_cells_train[train_index], y_cells_train[test_index]
        y_cells_cv.append(y_test)
        model_cv = SVC(C=100, gamma=0.01, )
        model_cv.fit(X_train, y_train)
        y_pred_cells_cv.append(model_cv.decision_function(X_test))
    y_cells_cv = np.concatenate(y_cells_cv, 0)
    y_pred_cells_cv = np.concatenate(y_pred_cells_cv, 0)
    # plot ROC based on 5fold CV
    fpr, tpr, th = roc_curve(y_cells_cv, y_pred_cells_cv, drop_intermediate=False)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='b',
             label='Our ROC curve (AUC={:.2f})'.format(roc_auc_score(y_cells_cv, y_pred_cells_cv)), lw=3)
    plt.scatter(0.08, 0.57, color='r', s=100, label='Results reported by Peikari et al.')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=3)
    plt.xticks(np.arange(0, 1.01, 0.2), fontsize=20)
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=20)
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.xlabel('1-Specificity', fontsize=20)
    plt.ylabel('Sensitivity', fontsize=20)
    plt.legend(loc='lower right', fontsize=15)
    plt.title('B vs. M', fontsize=20)
    plt.savefig('segmentation&classification/model/roc-bvsm.png')
    pass

elif args.opt == 'validate':    # validate clf models
    split_point = 110

    # generate training data for cells
    x_cells_train = np.concatenate([
        np.concatenate(np.load('segmentation&classification/cells/features/nuclei37.npy')[:split_point], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/shape7.npy')[:split_point], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/lbp10.npy')[:split_point], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/hara88.npy')[:split_point], 0)[:, 1:]
    ], 1)
    x_cells_train[np.isnan(x_cells_train)] = 0
    y_cells_train = np.concatenate(
        np.load('segmentation&classification/cells/features/shape7.npy')[:split_point],
        0)[:, 0].astype('uint8')

    # generate test data for cells
    x_cells_test = np.concatenate([
        np.concatenate(np.load('segmentation&classification/cells/features/nuclei37.npy')[split_point:], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/shape7.npy')[split_point:], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/lbp10.npy')[split_point:], 0)[:, 1:],
        np.concatenate(np.load('segmentation&classification/cells/features/hara88.npy')[split_point:], 0)[:, 1:]
    ], 1)
    x_cells_test[np.isnan(x_cells_test)] = 0
    y_cells_test = np.concatenate(
        np.load('segmentation&classification/cells/features/shape7.npy')[split_point:],
                       0)[:, 0].astype('uint8')
    anno_match = np.load('segmentation&classification/cells/anno_match.npy')

    # generate val data
    x_val = np.concatenate([
        np.concatenate(np.load('segmentation&classification/val/features/nuclei37.npy'), 0),
        np.concatenate(np.load('segmentation&classification/val/features/shape7.npy'), 0),
        np.concatenate(np.load('segmentation&classification/val/features/lbp10.npy'), 0),
        np.concatenate(np.load('segmentation&classification/val/features/hara88.npy'), 0)
    ], 1)
    x_val[np.isnan(x_val)] = 0
    centroid_val = np.load('segmentation&classification/val/centroid.npy')

    # generate corr data (for linear correction)
    x_corr = np.concatenate([
        np.concatenate(np.load('segmentation&classification/corr/features/nuclei37.npy'), 0),
        np.concatenate(np.load('segmentation&classification/corr/features/shape7.npy'), 0),
        np.concatenate(np.load('segmentation&classification/corr/features/lbp10.npy'), 0),
        np.concatenate(np.load('segmentation&classification/corr/features/hara88.npy'), 0)
    ], 1)
    x_corr[np.isnan(x_corr)] = 0
    centroid_corr = np.load('segmentation&classification/corr/centroid.npy')

    # scale data
    scaler =  MinMaxScaler((-1, 1))
    scaler.fit(x_cells_train)
    x_cells_train = scaler.transform(x_cells_train)
    x_cells_test = scaler.transform(x_cells_test)
    x_val = scaler.transform(x_val)
    x_corr = scaler.transform(x_corr)

    # load models
    with open('segmentation&classification/model/svm-lvsbm.pkl', 'rb') as f:
        model_1 = pkl.load(f)
    with open('segmentation&classification/model/svm-bvsm.pkl', 'rb') as f:
        model_2 = pkl.load(f)

    # make predictions
    y_pred_cells_test_1 = model_1.decision_function(x_cells_test)
    y_pred_cells_test_2 = model_2.decision_function(x_cells_test)
    y_pred_val_1 = model_1.decision_function(x_val)
    y_pred_val_2 = model_2.decision_function(x_val)
    y_pred_corr_1 = model_1.decision_function(x_corr)
    y_pred_corr_2 = model_2.decision_function(x_corr)

    y_pred_cells_test = np.array([2 if i >= -0.2 else (1 if j >= 0.7 else 0)
                                  for i, j in zip(y_pred_cells_test_1, y_pred_cells_test_2)])
    y_pred_cells_test_slidewise = []
    l = 0
    for j in anno_match[split_point:]:
        tmp = []
        for k in j:
            tmp.append(np.concatenate([k[2:4], k[5:6], [y_pred_cells_test[l]]], 0))
            l += 1
        y_pred_cells_test_slidewise.append(np.array(tmp))
    y_pred_cells_test_slidewise = np.array(y_pred_cells_test_slidewise)

    y_pred_val = np.array([2 if i >= -0.2 else (1 if j >= 0.7 else 0) for i, j in zip(y_pred_val_1, y_pred_val_2)])
    y_pred_val_with_label = []
    l = 0
    for j in centroid_val:
        tmp = []
        for k in j:
            tmp.append(np.append(k, y_pred_val[l]))
            l += 1
        y_pred_val_with_label.append(np.array(tmp))
    y_pred_val_with_label = np.array(y_pred_val_with_label)
    np.save('segmentation&classification/val/y_pred_val_with_label.npy', y_pred_val_with_label)

    y_pred_corr = np.array([2 if i >= -0.2 else (1 if j >= 0.7 else 0) for i, j in zip(y_pred_corr_1, y_pred_corr_2)])
    y_pred_corr_with_label = []
    l = 0
    for j in centroid_corr:
        tmp = []
        for k in j:
            tmp.append(np.append(k, y_pred_val[l]))
            l += 1
        y_pred_corr_with_label.append(np.array(tmp))
    y_pred_corr_with_label = np.array(y_pred_corr_with_label)
    np.save('segmentation&classification/corr/y_pred_corr_with_label.npy', y_pred_corr_with_label)

    # print clf performance
    print('\nPerformance of nuclei classification on test set')
    print('Class\tACC\t\tSEN\t\tSPE')
    print('Lym\t\t{:.3f}\t{:.3f}\t{:.3f}'.format(
        accuracy_score(y_cells_test == 2, y_pred_cells_test == 2),
        sen_and_spe(y_cells_test == 2, y_pred_cells_test == 2)[0],
        sen_and_spe(y_cells_test == 2, y_pred_cells_test == 2)[1]))
    print('Ben\t\t{:.3f}\t{:.3f}\t{:.3f}'.format(
        accuracy_score(y_cells_test == 1, y_pred_cells_test == 1),
        sen_and_spe(y_cells_test == 1, y_pred_cells_test == 1)[0],
        sen_and_spe(y_cells_test == 1, y_pred_cells_test == 1)[1]))
    print('Mal\t\t{:.3f}\t{:.3f}\t{:.3f}'.format(
        accuracy_score(y_cells_test == 0, y_pred_cells_test == 0),
        sen_and_spe(y_cells_test == 0, y_pred_cells_test == 0)[0],
        sen_and_spe(y_cells_test == 0, y_pred_cells_test == 0)[1]))

elif args.opt == 'predict':    # visual check & cellularity estimation & linear correction of svm clf for val
    # load data
    val_seg = np.load('segmentation&classification/val/seg.npy')
    corr_seg = np.load('segmentation&classification/corr/seg.npy')
    y_pred_val_with_label = np.load('segmentation&classification/val/y_pred_val_with_label.npy')
    y_pred_corr_with_label = np.load('segmentation&classification/corr/y_pred_corr_with_label.npy')

    # estimation for val
    clf_map_val = []
    malignant_map_val = []
    for i, j in zip(val_seg, y_pred_val_with_label):
        tmp_map = np.zeros(i.shape + (3, ), dtype='uint8')
        tmp_map[np.isin(i, j[j[:, 3] == 0, 2])] = [255, 0, 0]
        tmp_map[np.isin(i, j[j[:, 3] == 1, 2])] = [0, 0, 255]
        tmp_map[np.isin(i, j[j[:, 3] == 2, 2])] = [0, 255, 0]
        clf_map_val.append(tmp_map)
        malignant_map_val.append(np.isin(i, j[j[:, 3] == 0, 2]))
    clf_map_val = np.array(clf_map_val)
    malignant_map_val = np.array(malignant_map_val)
    malignant_map_val = np.array([binary_dilation(j, selem.disk(11)) for j in malignant_map_val])
    cellularity_val = np.array([np.sum(j) / (j.shape[0] * j.shape[1]) for j in malignant_map_val])
    if not os.path.exists('segmentation&classification/val/mapping'):
        os.mkdir('segmentation&classification/val/mapping')
    for i, j in enumerate(clf_map_val):
        skimage.io.imsave('segmentation&classification/val/mapping/{:03d}.tif'.format(i), j)

    # estimation for corr
    clf_map_corr = []
    malignant_map_corr = []
    for i, j in zip(corr_seg, y_pred_corr_with_label):
        tmp_map = np.zeros(i.shape + (3, ), dtype='uint8')
        tmp_map[np.isin(i, j[j[:, 3] == 0, 2])] = [255, 0, 0]
        tmp_map[np.isin(i, j[j[:, 3] == 1, 2])] = [0, 0, 255]
        tmp_map[np.isin(i, j[j[:, 3] == 2, 2])] = [0, 255, 0]
        clf_map_corr.append(tmp_map)
        malignant_map_corr.append(np.isin(i, j[j[:, 3] == 0, 2]))
    clf_map_corr = np.array(clf_map_corr)
    malignant_map_corr = np.array(malignant_map_corr)
    malignant_map_corr = np.array([binary_dilation(j, selem.disk(11)) for j in malignant_map_corr])
    cellularity_corr = np.array([np.sum(j) / (j.shape[0] * j.shape[1]) for j in malignant_map_corr])
    if not os.path.exists('segmentation&classification/corr/mapping'):
        os.mkdir('segmentation&classification/corr/mapping')
    for i, j in enumerate(clf_map_val):
        skimage.io.imsave('segmentation&classification/corr/mapping/{:03d}.tif'.format(i), j)

    # load true label
    y_cellu_val = np.load('data/val/y.npy')
    y_cellu_corr = np.load('data/corr/y.npy')
    # linear correction
    regr = LinearRegression().fit(np.expand_dims(cellularity_corr, 1), y_cellu_corr)
    cellularity_val_1 = regr.predict(np.expand_dims(cellularity_val, 1))
    cellularity_val_1 = np.clip(cellularity_val_1, .0, 1.)

    print('Performance of cellularity assessment:')
    metric, lcl, ucl = tau_ci(y_cellu_val, cellularity_val_1)
    print('Kendall\'s Tau:\t{:.2f} [{:.2f}, {:.2f}]'.format(metric, lcl, ucl))
    metric, lcl, ucl = predprob_ci(y_cellu_val, cellularity_val_1)
    print('PredProb:\t\t{:.2f} [{:.2f}, {:.2f}]'.format(metric, lcl, ucl))

    pd.DataFrame({'Truth': y_cellu_val, 'Prediction': cellularity_val_1}).\
        to_csv('segmentation&classification/pred_val.csv', index=False)