import numpy as np
from scipy.stats import norm
import pandas as pd

def compute_detection_intervals(
    score_vector,
    alpha,
    method="mean",
    smoothing=True,
    sliding=False,
    anomaly_padding=0
):
    """
    Given an anomaly score vector, compute detection intervals using either
    global thresholding or a sliding-window threshold, and optionally
    smooth the scores first via an exponentially-weighted moving average.

    Parameters
    ----------
    score_vector : array-like, shape (T,)
        The aligned anomaly score vector.
    alpha : float
        The upper quantile for thresholding (e.g. 0.05 ⇒ top 5%).
    method : {'mean', 'median'}, default='mean'
        Whether to compute central tendency + spread as (mean, std) or
        (median, MAD).
    smoothing : bool, default=False
        If True, first replace `score_vector` with its EWMA (alpha = smoothing_alpha).
    smoothing_alpha : float in (0,1), default=0.3
        The alpha for the EWMA if `smoothing=True`.
    sliding : bool, default=False
        If True, compute a local threshold in a sliding window (size T/3, step T/10)
        and mark any point exceeding its window's threshold as anomalous.
        Otherwise use a single global threshold.
    anomaly_padding : int, default=0
        Number of time points to pad before and after each detected interval.

    Returns
    -------
    detection_intervals : list of (start, end) tuples
        Contiguous index ranges (padded) where the (smoothed) score exceeds the threshold.
    threshold : float or None
        The global threshold (if sliding=False), else None.
    scores : np.ndarray
        The (optionally smoothed) score vector.
    """
    scores = np.array(score_vector, dtype=float)
    T = len(scores)

    if smoothing:
        span = max(1, int(len(scores) * 0.01))
        scores = pd.Series(scores).ewm(span=span).mean().values

    # Precompute the Gaussian multiplier
    z = norm.ppf(1 - alpha)

    # 2) Build a boolean mask of anomalies
    anomaly_flags = np.zeros(T, dtype=bool)

    if not sliding:
        # 2a) Global threshold
        if method == "mean":
            central = scores.mean()
            spread  = scores.std()
        elif method == "median":
            central = np.median(scores)
            spread  = np.median(np.abs(scores - central))
        else:
            raise ValueError("method must be 'mean' or 'median'")
        threshold = central + z * spread
        anomaly_flags = scores > threshold

    else:
        # 2b) Sliding‑window threshold
        threshold = 0 # With sliding window based method, threshold varies.
        win = max(1, T // 3)
        step = max(1, T // 10)
        for start in range(0, T, step):
            end = min(start + win, T)
            segment = scores[start:end]
            if method == "mean":
                central = segment.mean()
                spread  = segment.std()
            else:  # median
                central = np.median(segment)
                spread  = np.median(np.abs(segment - central))
            thresh_local = central + z * spread
            # mark any point in [start:end) above its local thresh
            anomaly_flags[start:end] |= (segment > thresh_local)

    # 3) Extract contiguous intervals from the boolean mask
    detection_intervals = []
    in_int = False
    for i, flag in enumerate(anomaly_flags):
        if flag and not in_int:
            in_int = True
            start = i
        elif not flag and in_int:
            in_int = False
            detection_intervals.append((start, i-1))
    if in_int:
        detection_intervals.append((start, T-1))


    # 4) Apply padding if requested
    if anomaly_padding > 0:
        padded = []
        for (s, e) in detection_intervals:
            s_pad = max(0, s - anomaly_padding)
            e_pad = min(T - 1, e + anomaly_padding)
            padded.append((s_pad, e_pad))

        # If nothing was detected, stay empty
        if not padded:
            detection_intervals = []
        else:
            # 5) Merge overlapping or contiguous intervals
            padded.sort(key=lambda x: x[0])
            merged = []
            cur_s, cur_e = padded[0]
            for s_next, e_next in padded[1:]:
                if s_next <= cur_e + 1:   # overlap or contiguous
                    cur_e = max(cur_e, e_next)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s_next, e_next
            merged.append((cur_s, cur_e))

            detection_intervals = merged
    
    return detection_intervals, threshold, scores
    
def align_anomaly_vector(final_vector, T_full, window_size, step_size, n_windows):
    """
    Align the stitched anomaly vector to the original time series length T_full,
    by first interpolating it out to the full covered span of your sliding windows,
    then extrapolating or truncating to exactly T_full.

    Parameters
    ----------
    final_vector : np.ndarray
        The stitched anomaly vector of length L (one score per column in the overlap).
    T_full : int
        The total number of time points in the original series.
    window_size : int
        Number of time points per window.
    step_size : int
        Step size between windows.
    n_windows : int
        Number of windows used in the sliding-window imagery.

    Returns
    -------
    aligned_vector : np.ndarray
        A 1D array of length T_full.
    """
    # 1) compute the total span covered by your windows
    covered_length = window_size + (n_windows - 1) * step_size

    L = len(final_vector)
    # 2) interpolate final_vector (length L) → covered_length
    x_old = np.arange(L)
    x_new = np.linspace(0, L - 1, covered_length)
    interp = np.interp(x_new, x_old, final_vector)

    # 3) now align to T_full
    if covered_length < T_full:
        # linear extrapolation past the end
        if covered_length > 1:
            slope = interp[-1] - interp[-2]
        else:
            slope = 0.0
        extra = T_full - covered_length
        extrap = interp[-1] + slope * np.arange(1, extra + 1)
        aligned = np.concatenate([interp, extrap])
    else:
        # just truncate if we overshot
        aligned = interp[:T_full]

    return aligned

##### Metric Calculation
def format_intervals(intervals):
    """
    Convert a list of intervals (e.g. list of lists or tuples) 
    into a list of tuples.
    """
    return [tuple(interval) for interval in intervals]

def intervals_overlap(int1, int2):
    """
    Determine if two intervals (start, end) overlap.
    """
    start1, end1 = int1
    start2, end2 = int2
    return not (end1 < start2 or end2 < start1)


def merge_intervals(intervals):
    """
    Merge a list of intervals (each as a tuple (start, end)) so that overlapping or contiguous intervals
    are combined.
    """
    if not intervals:
        return []
    # Sort intervals by their start values.
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    current_start, current_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= current_end + 1:
            # Overlap or contiguous.
            current_end = max(current_end, e)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = s, e
    merged.append((current_start, current_end))
    return merged

def interval_length(interval):
    """Return the length of an interval assuming inclusive endpoints."""
    s, e = interval
    return e - s + 1

def interval_intersection(interval1, interval2):
    """Return the intersection of two intervals as a tuple, or None if they don't overlap."""
    s1, e1 = interval1
    s2, e2 = interval2
    s = max(s1, s2)
    e = min(e1, e2)
    if s <= e:
        return (s, e)
    else:
        return None
    

def compute_precision_recall_f1(agg):
    """
    Given aggregated counts for TP, FP, and FN, compute:
       - Precision = TP / (TP+FP)
       - Recall = TP / (TP+FN)
       - F1 = 2pr/(p+r)
       - F0.5 = (1+0.5^2)*p*r/(0.5^2 * p + r)
    """
    TP = agg.get("TP", 0)
    FP = agg.get("FP", 0)
    FN = agg.get("FN", 0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    beta = 0.5
    f0_5 = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)) if (beta**2 * precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "F1": f1, "F0.5": f0_5}

def compute_pr_auc(pr_points):
    """
    Given a list of (recall, precision) points, compute the area under the PR curve using trapezoidal rule.
    Assumes that pr_points is a list of tuples (r,p) that we sort by recall.
    """
    pr_points.sort(key=lambda x: x[0])
    recalls = np.array([pt[0] for pt in pr_points])
    precisions = np.array([pt[1] for pt in pr_points])
    auc = np.trapz(precisions, recalls)
    return auc

def aggregate_metrics(metrics_list):
    """
    Aggregate a list of metric dictionaries (each keyed by alpha) by summing component-wise.
    If any file reports a None for a given key (e.g. threshold), the aggregate for that key
    remains None.
    """
    aggregated = {}
    for metrics in metrics_list:
        for alpha, vals in metrics.items():
            if alpha not in aggregated:
                aggregated[alpha] = {}
            for key, value in vals.items():
                # first time we see this alpha/key
                if key not in aggregated[alpha]:
                    aggregated[alpha][key] = None if value is None else value
                else:
                    # once None, always None
                    if aggregated[alpha][key] is None or value is None:
                        aggregated[alpha][key] = None
                    else:
                        aggregated[alpha][key] += value
    return aggregated


### Window-wise evaluation ###
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


def make_window_eval_record(true_intervals, detected_intervals, threshold=None):
    """
    Package up everything for one alpha:
      - normalize both interval lists
      - compute TP/FP/FN via window_wise_metrics
      - count total detected length
      - include the (optional) threshold
    """
    # ensure both are list of (start,end) tuples
    true_fmt = format_intervals(true_intervals)
    det_fmt  = format_intervals(detected_intervals)

    # core metrics
    metrics = window_wise_metrics(true_fmt, det_fmt)

    # total #points in all detected windows
    total_detected = sum((end - start + 1) for start, end in det_fmt)

    return {
        "true_intervals":        true_fmt,
        "detected_intervals":    det_fmt,
        **metrics,
        "total_detected_length": total_detected,
        "threshold":             threshold,
    }


def evaluate_detections_window(true_intervals, aligned_anomaly_vector, alpha_list, print_intervals=False, smoothing=True):
    """
    For each alpha, run compute_detection_intervals, then build a record via
    make_window_eval_record.
    """
    results = {}
    for alpha in alpha_list:
        det_ints, thresh, _ = compute_detection_intervals(aligned_anomaly_vector, alpha, smoothing=smoothing)
        record = make_window_eval_record(true_intervals, det_ints, threshold=thresh)
        results[alpha] = record
        if print_intervals:
            print(f"--- alpha = {alpha} (thresh={thresh}) ---")
            print(f"True intervals:     {true_intervals}")
            print(f"Detected intervals: {det_ints}\n")
    return results