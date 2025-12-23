"""Time series preprocessing and visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import warnings
from io import BytesIO
from PIL import Image
from scipy.signal import detrend


def preprocess_time_series(time_series):
    """
    Preprocess a time series by detrending and min-max standardization.

    Parameters
    ----------
    time_series : array-like
        The raw time series values.

    Returns
    -------
    preprocessed_series : np.ndarray
        The detrended and min-max normalized time series.
    """
    # Convert to a NumPy array (in case it's not already)
    ts = np.array(time_series, dtype=float)

    # Detrend the time series
    ts_detrended = detrend(ts)

    # Min-max normalization
    ts_min = np.min(ts_detrended)
    ts_max = np.max(ts_detrended)
    if ts_max - ts_min > 0:
        ts_normalized = (ts_detrended - ts_min) / (ts_max - ts_min)
    else:
        # In case the series is constant after detrending, return zeros
        ts_normalized = np.zeros_like(ts_detrended)

    return ts_normalized


def draw_image(
    series_id,
    save_path,
    time_series,
    time_points,
    override=True,
    save_image=False,
    image_size=(240, 240),
    dpi=100,
    plot_params=('-', 1, '*', 2, 'black', None)
):
    """
    Create a line plot image of the time series.

    Parameters
    ----------
    series_id : str
        Unique identifier for the time series (used in file names).
    save_path : str
        Directory where the output image tensor (and optionally the PNG image) will be saved.
    time_series : array-like
        Time series values, shape (T,).
    time_points : array-like of shape (T,)
        The time values (e.g., timestamps).
    override : bool, optional
        If False and files already exist, the function will skip saving.
    save_image : bool, optional
        If True, also save the PNG image.
    image_size : tuple (height, width)
        Desired image size in pixels.
    dpi : int, optional
        Dots per inch for the saved image.
    plot_params : tuple
        Plot style parameters: (linestyle, linewidth, marker, markersize, color, y_scale).

    Returns
    -------
    np.ndarray or None
        Image tensor of shape [C, H, W], or None if skipped.
    """

    # Ensure the output directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create filenames (always use "_line" suffix)
    base_name = f"{series_id}_line"
    tensor_filename = os.path.join(save_path, base_name + "_img.npy")
    png_filename = os.path.join(save_path, base_name + ".png")

    # If files already exist and override is False, skip
    if os.path.exists(tensor_filename) and (not save_image or os.path.exists(png_filename)) and not override:
        print(f"Files for {base_name} already exist. Skipping...")
        return None

    # Convert time_series and time_points to numpy arrays
    time_series = np.array(time_series, dtype=float)
    time_points = np.array(time_points, dtype=float)

    # Generate line plot
    fig_width = image_size[1] / dpi
    fig_height = image_size[0] / dpi

    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    linestyle, linewidth, marker, markersize, color, y_scale = plot_params
    plt.plot(time_points, time_series, linestyle=linestyle, linewidth=linewidth,
                marker=marker, markersize=markersize, color=color)
    if y_scale is not None:
        plt.ylim(y_scale)

    # Remove ticks for a minimal context
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    # Save the plot to an in-memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close()
    buf.seek(0)

    # Load the image from the buffer as a tensor
    img_pil = Image.open(buf).convert("RGB")
    transform_img = transforms.ToTensor()
    img_tensor = transform_img(img_pil)  # shape: [C, H, W]

    # Save image tensor and PNG image (if desired)
    if save_image:
        with open(png_filename, 'wb') as f:
            f.write(buf.getbuffer())
        print(f"Saved PNG image: {png_filename}")

    return img_tensor.cpu().numpy()


def draw_windowed_images(
    base_series_id,
    save_path,
    time_series,
    time_points,
    window_size=200,
    step_size=100,
    override=True,
    save_image=False,
    image_size=(240, 240),
    dpi=100,
    plot_params=('-', 1, '*', 2, 'black', None)
):
    """
    Generate line plot images for sub-sequences of a time series using a sliding window.

    For each window (of length window_size, moving by step_size), a sub-sequence
    is extracted and passed to the draw_image() function.

    Parameters
    ----------
    base_series_id : str
        Base name for the time series (e.g. "series").
    save_path : str
        Directory to save the image tensor and PNG image (if save_image True).
    time_series : array-like
        Full time series values, shape (T,).
    time_points : array-like of shape (T,)
        Time values corresponding to the full series.
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
        Plot style parameters: (linestyle, linewidth, marker, markersize, color, y_scale).

    Returns
    -------
    bool
        True if processing is successful.
    """
    num_points = len(time_series)
    aggregated_imgs = []
    window_id = 0

    # Iterate over windows
    for start in range(0, num_points - window_size + 1, step_size):
        end = start + window_size
        # Extract the windowed sub-sequence and corresponding time points
        window_series = time_series[start:end]
        window_time = time_points[start:end]

        # Create a unique series ID for this window
        window_id += 1
        window_series_id = f"{base_series_id}_{window_id}"

        # Call draw_image() for this window
        img_tensor = draw_image(
            series_id=window_series_id,
            save_path=save_path,
            time_series=window_series,
            time_points=window_time,
            override=override,
            save_image=save_image,
            image_size=image_size,
            dpi=dpi,
            plot_params=plot_params
        )

        if img_tensor is not None:
            aggregated_imgs.append(img_tensor)

    if len(aggregated_imgs) == 0:
        warnings.warn("No windowed images were generated.")
        return False

    aggregated_imgs = np.stack(aggregated_imgs)  # shape: [num_windows, C, H, W]

    # Build base filename (always use "line" suffix)
    base_filename = os.path.join(save_path, f"{base_series_id}_line")
    np.save(base_filename + "_img.npy", aggregated_imgs)
    print(f"Saved aggregated image tensor to {base_filename}_img.npy")

    return True
