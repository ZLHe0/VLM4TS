import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
import warnings
from io import BytesIO

def draw_image(
    series_id,
    save_path,
    time_series,
    time_points,
    anomaly_intervals=None,
    override=True,
    save_image=False,
    image_size=(240,240),  # typical size for ViT input
    dpi=100,
    plot_params=('-', 1, '*', 2, 'black', None),  # For univariate: (linestyle, linewidth, marker, markersize, color, y_scale)
    transform_type='line'   # Supported types: 'line', 'gaf', 'rp', 'area'
):
    """
    Create an image of the time series and generate an anomaly segmentation mask.
    
    This function supports:
      - Univariate time series: can use 'line', 'gaf', 'rp', or 'area' transforms.
      - Multivariate time series: only a line plot is supported; each channel is plotted 
        using Matplotlib's default color cycle.
    
    The x-axis is always labeled as "Time". Anomaly intervals (if provided) are used only
    for mask generation (no shading on the plot).
    
    Parameters
    ----------
    series_id : str
        Unique identifier for the time series (used in file names).
    save_path : str
        Directory where the output image tensor (and optionally the PNG image) and mask will be saved.
    time_series : array-like
        For univariate: shape (T,). For multivariate: shape (T, F) where F > 1.
    time_points : array-like of shape (T,)
        The time values (e.g., Unix timestamps).
    anomaly_intervals : list of [start, end] pairs or None
        Anomaly intervals (each as two timestamps). Used only for generating the mask.
    override : bool, optional
        If False and files already exist, the function will skip saving.
    save_image : bool, optional
        If True, also save the PNG image.
    image_size : tuple (height, width)
        Desired image size in pixels.
    dpi : int, optional
        Dots per inch for the saved image.
    plot_params : tuple
        For univariate line plots: (linestyle, linewidth, marker, markersize, color, y_scale).
    transform_type : str, optional
        Plot transformation type: 'line' (default), 'gaf', 'rp', or 'area'. For multivariate, only 'line' is allowed.
    
    Returns
    -------
    tuple
        (img_tensor, mask) where:
         - img_tensor is a numpy array of shape [C, H, W].
         - mask is a binary anomaly mask of shape [H, W] (all zeros if anomaly_intervals is None).
    """
    
    # Ensure the output directory exists.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Create a base name.
    transform_label = ""
    if transform_type.lower() == 'gaf':
        transform_label = "_GAF"
    elif transform_type.lower() == 'rp':
        transform_label = "_RP"
    elif transform_type.lower() == 'area':
        transform_label = "_area"
    else:
        transform_label = "_line"
    
    base_name = f"{series_id}{transform_label}"
    tensor_filename = os.path.join(save_path, base_name + "_img.npy")
    png_filename = os.path.join(save_path, base_name + ".png")
    mask_filename = os.path.join(save_path, base_name + "_mask.npy")
    
    # If files already exist and override is False, skip.
    if (os.path.exists(tensor_filename) and
        (not save_image or os.path.exists(png_filename)) and
        (anomaly_intervals is None or os.path.exists(mask_filename))
       ) and not override:
        print(f"Files for {base_name} already exist. Skipping...")
        return None, None

    # Convert time_series and time_points to numpy arrays.
    time_series = np.array(time_series, dtype=float)
    time_points = np.array(time_points, dtype=float)
    T = time_series.shape[0]
    
    # --- Step 1: Compute anomaly flags if anomaly_intervals is provided ---
    anomaly_flags = np.zeros(T, dtype=bool)
    if anomaly_intervals is not None:
        for i, t in enumerate(time_points):
            for interval in anomaly_intervals:
                start, end = interval
                if start <= t <= end:
                    anomaly_flags[i] = True
                    break

    # --- Step 2: Generate the plot based on transform_type ---
    fig_width = image_size[1] / dpi
    fig_height = image_size[0] / dpi
    
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    if transform_type.lower() == 'line':
        linestyle, linewidth, marker, markersize, color, y_scale = plot_params
        plt.plot(time_points, time_series, linestyle=linestyle, linewidth=linewidth,
                    marker=marker, markersize=markersize, color=color)
        if y_scale is not None:
            plt.ylim(y_scale)
    elif transform_type.lower() == 'area':
        _, _, _, _, color, _ = plot_params
        plt.fill_between(time_points, time_series, color=color)
    elif transform_type.lower() == 'gaf':
        from pyts.image import GramianAngularField
        transformer = GramianAngularField(method='summation')
        window_data_2d = time_series[np.newaxis, :]  # shape: [1, T]
        transformed_image = transformer.fit_transform(window_data_2d)
        plt.imshow(transformed_image[0], cmap='rainbow', origin='lower')
    elif transform_type.lower() == 'rp':
        from pyts.image import RecurrencePlot
        transformer = RecurrencePlot(threshold='point', percentage=10, dimension=1, time_delay=1)
        window_data_2d = time_series[np.newaxis, :]
        transformed_image = transformer.fit_transform(window_data_2d)
        plt.imshow(transformed_image[0], cmap='gray', origin='lower')
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")
    
    # Remove ticks for a minimal context.
    plt.xticks([]); plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    # Save the plot to an in-memory buffer.
    buf = BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    # --- Step 3: Load the image from the buffer as a tensor ---
    img_pil = Image.open(buf).convert("RGB")
    transform_img = transforms.ToTensor()
    img_tensor = transform_img(img_pil)  # shape: [C, H, W]
    
    # --- Step 4: Generate anomaly mask ---
    H = img_tensor.shape[1]
    W = img_tensor.shape[2]
    mask = np.zeros((H, W), dtype=np.float32)  # Default mask: all zeros.
    if anomaly_intervals is not None:
        indices = np.linspace(0, T - 1, num=W, dtype=int)
        for col in range(W):
            ts_idx = indices[col]
            if anomaly_flags[ts_idx]:
                mask[:, col] = 1.0
    
    # # --- Step 5: Save image tensor and PNG image (if desired) ---
    if save_image:
        with open(png_filename, 'wb') as f:
            f.write(buf.getbuffer())
        print(f"Saved PNG image: {png_filename}")
    
    return img_tensor.cpu().numpy(), mask


def draw_windowed_images(
    base_series_id,
    save_path,
    time_series,         # For univariate: (T,), for multivariate: (T, F) with F > 1.
    time_points,         # 1D array-like of shape (T,)
    anomaly_intervals=None,
    window_size=200,
    step_size=100,
    override=True,
    save_image=False,
    image_size=(240,240),  # Typical size for ViT
    dpi=100,
    plot_params=('-', 1, '*', 2, 'black', None),  # For univariate: (linestyle, linewidth, marker, markersize, color, y_scale)
    transform_type='line'   # Supported: 'line', 'gaf', 'rp'
):
    """
    Generate images for sub-sequences of a time series using a sliding window.
    
    For each window (of length window_size, moving by step_size), a sub-sequence
    is extracted and passed to the draw_image() function. Each output file is named
    with the base_series_id and a suffix representing the window number.
    
    Anomaly intervals (if provided) are used solely to generate a binary anomaly mask.
    
    Parameters
    ----------
    base_series_id : str
        Base name for the time series (e.g. "Twitter_volume_AAPL").
    save_path : str
        Directory to save the image tensor, PNG image (if save_image True), and mask.
    time_series : array-like
        Full time series values. For univariate, shape (T,); for multivariate, shape (T, F).
    time_points : array-like of shape (T,)
        Time values corresponding to the full series.
    anomaly_intervals : list of [start, end] pairs or None
        Anomaly intervals for the full sequence (in Unix timestamps).
    window_size : int, optional
        Length of the sub-sequence window (default: 200).
    step_size : int, optional
        Step size for sliding the window (default: 100).
    override : bool, optional
        Whether to overwrite existing files.
    save_image : bool, optional
        If True, also save the PNG image.
    image_size : tuple, optional
        Desired output image size in pixels (height, width).
    dpi : int, optional
        Dots per inch for the saved image.
    plot_params : tuple, optional
        Plot style parameters for univariate line plots: 
          (linestyle, linewidth, marker, markersize, color, y_scale).
    transform_type : str, optional
        Type of transformation: 'line' (default), 'gaf', or 'rp'.
        For multivariate time series, only 'line' is supported.
    
    Returns
    -------
    bool
        True if processing is successful.
    """
    num_points = len(time_series)
    aggregated_imgs = []
    aggregated_masks = []
    window_id = 0

    # Iterate over windows.
    for start in range(0, num_points - window_size + 1, step_size):
        end = start + window_size
        # Extract the windowed sub-sequence and corresponding time points.
        window_series = time_series[start:end]      # shape: (window_size,) or (window_size, F)
        window_time = time_points[start:end]          # shape: (window_size,)
        
        # Clip anomaly intervals to the window, if provided.
        window_anomaly_intervals = None
        if anomaly_intervals is not None:
            window_anomaly_intervals = []
            window_start_time = window_time[0]
            window_end_time = window_time[-1]
            for interval in anomaly_intervals:
                a_start, a_end = interval
                if a_end >= window_start_time and a_start <= window_end_time:
                    clipped_start = max(a_start, window_start_time)
                    clipped_end = min(a_end, window_end_time)
                    window_anomaly_intervals.append([clipped_start, clipped_end])
            if len(window_anomaly_intervals) == 0:
                window_anomaly_intervals = None
        
        # Create a unique series ID for this window.
        window_id += 1
        window_series_id = f"{base_series_id}_{window_id}"
        
        # Call draw_image() for this window.
        ret = draw_image(
            series_id=window_series_id,
            save_path=save_path,
            time_series=window_series,
            time_points=window_time,
            anomaly_intervals=window_anomaly_intervals,
            override=override,
            save_image=save_image,
            image_size=image_size,
            dpi=dpi,
            plot_params=plot_params,
            transform_type=transform_type
        )
        
        if ret is not None:
            img_tensor, mask = ret  # img_tensor: numpy array [C, H, W], mask: numpy array [H, W]
            aggregated_imgs.append(img_tensor)
            aggregated_masks.append(mask)  # Note: even if no anomaly, mask is all zeros.
    
    if len(aggregated_imgs) == 0:
        warnings.warn("No windowed images were generated.")
        return False

    aggregated_imgs = np.stack(aggregated_imgs)  # shape: [num_windows, C, H, W]
    aggregated_masks = np.stack(aggregated_masks)  # shape: [num_windows, H, W]
    
    # Build base filename.
    base_filename = os.path.join(save_path, f"{base_series_id}_{transform_type}")
    np.save(base_filename + "_img.npy", aggregated_imgs)
    print(f"Saved aggregated image tensor to {base_filename}_img.npy")
    np.save(base_filename + "_mask.npy", aggregated_masks)
    print(f"Saved aggregated anomaly mask to {base_filename}_mask.npy")
    
    return True
