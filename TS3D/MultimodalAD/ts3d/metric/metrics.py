
from .basic_metrics import basic_metricor, generate_curve
from sklearn.metrics import confusion_matrix
import pandas as pd


def calculate_weights(predicted, actual, k=0.5):
    predicted_anomalies = [i for i, x in enumerate(predicted) if x == 1]
    actual_anomalies = [i for i, x in enumerate(actual) if x == 1]
    overlap = set(predicted_anomalies).intersection(actual_anomalies)

    weights = [1] * len(predicted)

    if not overlap:
        return weights
    else:

        sorted_overlap = sorted(actual_anomalies)
        tp_segments = []
        if sorted_overlap:
            current_start = sorted_overlap[0]
            current_end = sorted_overlap[0]
            for idx in sorted_overlap[1:]:
                if idx == current_end + 1:
                    current_end = idx
                else:
                    tp_segments.append(
                        (current_start, current_end, current_end - current_start + 1)
                    )
                    current_start = idx
                    current_end = idx
            tp_segments.append(
                (current_start, current_end, current_end - current_start + 1)
            )

        fp_indices = set(predicted_anomalies) - overlap
        for i in fp_indices:
            if not tp_segments:
                weights[i] = 0  
            else:
                min_distance = float("inf")
                selected_length = 1
                for s, e, length in tp_segments:
                    if i < s:
                        d = s - i
                    elif i > e:
                        d = i - e
                    else:
                        d = 0
                    if d < min_distance:
                        min_distance = d
                        selected_length = length
                    elif d == min_distance and length > selected_length:
                        selected_length = length
                # weights[i] = (1 / selected_length) * min_distance if min_distance != 0 else 1
                weights[i] = (
                    selected_length / (selected_length + k * min_distance**2)
                    if min_distance != 0
                    else 1
                )

        fn_indices = set(actual_anomalies) - overlap
        for i in fn_indices:
            if not tp_segments:
                weights[i] = 0  
            else:
                min_distance = float("inf")
                selected_length = 1
                for s, e, length in tp_segments:
                    if i < s:
                        d = s - i
                    elif i > e:
                        d = i - e
                    else:
                        d = 0
                    if d < min_distance:
                        min_distance = d
                        selected_length = length
                    elif d == min_distance and length > selected_length:
                        selected_length = length
                # weights[i] = (1 / selected_length) ** min_distance if min_distance != 0 else 1
                weights[i] = (
                    selected_length / (selected_length + k * min_distance**2)
                    if min_distance != 0
                    else 1
                )

        return weights


def get_metrics(score, labels, slidingWindow=100, pred=None, version="opt", thre=250):
    metrics = {}

    pred_labels = pred.astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    if tp + fp == 0:
        precision = "NULL"
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = "NULL"
    else:
        recall = tp / (tp + fn)

    if precision == "NULL" or recall == "NULL" or precision + recall == 0:
        f1 = "NULL"
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")

    weights = calculate_weights(pred_labels, labels)

    # filtered_weights = [weight for weight in weights if weight > 1]
    # print("weights greater than 1:", filtered_weights)

    TP = TN = FP = FN = 0
    TS_class = []


    for i in range(len(labels)):
        if labels[i] == 1 and pred_labels[i] == 1:
            TP += weights[i]  
            TS_class.append(1)
        elif labels[i] == 0 and pred_labels[i] == 0:
            TN += weights[i]  
            TS_class.append(2)
        elif labels[i] == 0 and pred_labels[i] == 1:
            FP += weights[i]  
            TS_class.append(3)
        elif labels[i] == 1 and pred_labels[i] == 0:
            FN += weights[i]  
            TS_class.append(4)
    df = pd.DataFrame(TS_class, columns=["TS_class"])

    # df.to_csv("/home/yyy/baselines/TSB-AD/draw_pics/TP/TS_class.csv", index=False)

    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    if TP + FP == 0:
        Precision = "NULL"
    else:
        Precision = TP / (TP + FP)

    if TP + FN == 0:
        Recall = "NULL"
    else:
        Recall = TP / (TP + FN)

    if Precision == "NULL" or Recall == "NULL" or Precision + Recall == 0:
        F1 = "NULL"
    else:
        F1 = 2 * Precision * Recall / (Precision + Recall)

    print(f"Accuracy: {Accuracy}, Precision: {Precision}, Recall: {Recall}, F1: {F1}")

    """
    Threshold Independent
    """
    grader = basic_metricor()
    # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        labels, score, slidingWindow, version, thre
    )

    """
    Threshold Dependent
    if pred is None --> use the oracle threshold
    """

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)

    metrics["MA-F1"] = F1
    metrics["Standard-F1"] = PointF1
    metrics["PA-F1"] = PointF1PA
    metrics["Event-based-F1"] = EventF1PA
    metrics["R-based-F1"] = RF1
    metrics["Affiliation-F"] = Affiliation_F

    metrics["AUC-PR"] = AUC_PR
    metrics["AUC-ROC"] = AUC_ROC
    metrics["VUS-PR"] = VUS_PR
    metrics["VUS-ROC"] = VUS_ROC



    return metrics
