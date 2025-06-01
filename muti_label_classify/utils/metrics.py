import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix

def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_mAP(seg, num, return_each=False):
    gt_label = seg[:, num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims=True)

    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []

    for class_id in range(class_num):
        confidence = seg[:, class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i] > 0)
            fp[i] = (sorted_label[i] <= 0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / np.maximum(float(true_num), np.finfo(np.float64).eps)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = np.mean(aps)
    if return_each:
        return mAP, aps
    return mAP

def get_PR(seg, num,output):
    targets = seg[:, num:].astype(np.int32)
    preds = seg[:,:num]
    def prob_to_binary(probs, threshold):
        return (probs >= threshold).astype(int)
    thresholds = [i*0.1 for i in range(1,10)]
    f1_scores = [f1_score(targets, prob_to_binary(preds, t),average='micro') for t in thresholds]
    print(f1_scores)
    print(np.max(f1_scores))
    best_threshold = thresholds[np.argmax(f1_scores)]
    threshold = best_threshold
    preds = (preds >= threshold).astype(np.int32)

    # calculate P and R
    CF1 = f1_score(targets,preds, average='macro')
    OF1 = f1_score(targets,preds, average='micro')
    CP = precision_score(targets, preds, average='macro')
    OP = precision_score(targets, preds, average='micro')
    CR = recall_score(targets, preds, average='macro')
    OR = recall_score(targets, preds, average='micro')
    # precision = precision_score(targets, preds, average=None)
    # recall = recall_score(targets, preds, average=None)
    # metrics = {
    #     'Class': np.arange(num),  # 类别 0 到 79
    #     'Precision': precision,
    #     'Recall': recall,
    # }
    # metrics = pd.DataFrame(metrics)
    # metrics.to_csv(os.path.join(output,'PR_class.csv'), index=True)
    # cm = confusion_matrix(targets.flatten(), preds.flatten(), labels=[0, 1])
    # cm = pd.DataFrame(cm)
    # cm.to_csv(os.path.join(output,'confusion_matrix.csv'), index=True)
    return CF1,CP,CR,OF1,OP,OR


