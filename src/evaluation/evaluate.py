
import json
import pandas as pd

def intervals_overlap(int1, int2):
    """
    Determine if two intervals (start, end) overlap.
    """
    start1, end1 = int1
    start2, end2 = int2
    return not (end1 < start2 or end2 < start1)

def window_wise_metrics(true_intervals, detected_intervals):
    """
    Compute the evaluation metrics:
      - True Positives (TP): for each detected interval, count how many true intervals it overlaps.
      - False Positives (FP): count detected intervals with no overlap.
      - False Negatives (FN): count true intervals that are not overlapped by any detected interval.
    
    Both true_intervals and detected_intervals are assumed to be lists of tuples (start, end).
    """
    # Ensure intervals are in tuple format.
    true_intervals = [tuple(i) for i in true_intervals]
    detected_intervals = [tuple(i) for i in detected_intervals]
    
    TP = 0
    FP = 0
    for d in detected_intervals:
        # Count overlaps for each detection.
        overlap_count = sum(1 for a in true_intervals if intervals_overlap(d, a))
        if overlap_count > 0:
            TP += overlap_count
        else:
            FP += 1
            
    # Count false negatives: true intervals with no detection overlapping.
    FN = sum(1 for a in true_intervals if not any(intervals_overlap(a, d) for d in detected_intervals))
    
    return {"TP": TP, "FP": FP, "FN": FN}

def compute_precision_recall_f1(agg):
    """
    Given aggregated counts for TP, FP, and FN, compute:
       - Precision = TP / (TP+FP)
       - Recall = TP / (TP+FN)
       - F1 = 2pr/(p+r)
    """
    TP = agg.get("TP", 0)
    FP = agg.get("FP", 0)
    FN = agg.get("FN", 0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "F1": f1}

def evaluate_intervals(ground_truth_intervals, detected_intervals):
    """
    Computes window-based precision, recall, and F1 score.

    Parameters
    ----------
    ground_truth_intervals : list of lists or tuples
        The ground truth anomaly intervals.
    detected_intervals : list of lists or tuples
        The detected anomaly intervals.

    Returns
    -------
    dict
        A dictionary with 'precision', 'recall', and 'F1' scores.
    """
    metrics = window_wise_metrics(ground_truth_intervals, detected_intervals)
    results = compute_precision_recall_f1(metrics)
    return results

def main():
    """
    Main function to run the evaluation.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection results.")
    parser.add_argument("ground_truth_file", help="Path to the JSON file containing ground truth intervals.")
    parser.add_argument("detected_file", help="Path to the CSV file containing detected anomaly intervals.")
    args = parser.parse_args()

    # Load ground truth intervals
    with open(args.ground_truth_file, 'r') as f:
        ground_truth_intervals = json.load(f)

    # Load detected intervals
    detected_intervals_df = pd.read_csv(args.detected_file)
    detected_intervals = detected_intervals_df[['start', 'end']].values.tolist()

    # Compute metrics
    results = evaluate_intervals(ground_truth_intervals, detected_intervals)

    print("Evaluation Results:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1 Score: {results['F1']:.4f}")

if __name__ == "__main__":
    main()
