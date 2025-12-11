# check_block.py

import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def detect_mosaic_advanced(
    image_path: str,
    output_dir: Path | None = None,
    tuning_params: dict | None = None,
    debug: bool = False,
) -> int | None:
    """
    Advanced version with tuning parameters AND debug mode.
    """

    if tuning_params is None:
        tuning_params = {
            "median_blur_kernel": 3,
            "peak_min_distance": 5,
            "peak_prominence_factor": 1.5,
            "min_peaks_detected": 5,
            "max_std_dev": 3.0,
            "max_size_difference": 3,
            "flatness_gradient_threshold": 15,
            "flatness_min_ratio": 0.4,
        }

    k = tuning_params["median_blur_kernel"]
    prom_factor = tuning_params["peak_prominence_factor"]
    min_dist = tuning_params["peak_min_distance"]
    min_peaks = tuning_params["min_peaks_detected"]
    max_std = tuning_params["max_std_dev"]
    max_diff = tuning_params["max_size_difference"]
    flat_thresh = tuning_params["flatness_gradient_threshold"]
    flat_ratio = tuning_params["flatness_min_ratio"]

    base_name = Path(image_path).stem
    try:
        original_color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_color_img is None:
            logging.warning(f"Failed to load image: {image_path}")
            return None

        img_gray = cv2.cvtColor(original_color_img, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape

        img_blur = cv2.medianBlur(img_gray, k if k % 2 == 1 else k + 1)

        grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        sum_x = np.sum(abs_grad_x, axis=0)
        sum_y = np.sum(abs_grad_y, axis=1)

        prominence_x = np.std(sum_x) * prom_factor
        prominence_y = np.std(sum_y) * prom_factor

        peaks_x, _ = find_peaks(sum_x, prominence=prominence_x, distance=min_dist)
        peaks_y, _ = find_peaks(sum_y, prominence=prominence_y, distance=min_dist)

        if len(peaks_x) < min_peaks or len(peaks_y) < min_peaks:
            if debug:
                logging.warning(
                    f"{base_name}: FAILED Filter 1 (Min Peaks). "
                    f"X: {len(peaks_x)}, Y: {len(peaks_y)}. (Required >= {min_peaks})"
                )
            return None

        distances_x = np.diff(peaks_x)
        distances_y = np.diff(peaks_y)

        if len(distances_x) < 1 or len(distances_y) < 1:
            return None

        std_x = np.std(distances_x)
        std_y = np.std(distances_y)
        if std_x > max_std or std_y > max_std:
            if debug:
                logging.warning(
                    f"{base_name}: FAILED Filter 2 (Std Dev). "
                    f"X: {std_x:.2f}, Y: {std_y:.2f}. (Max: {max_std})"
                )
            return None

        median_size_x = np.median(distances_x)
        median_size_y = np.median(distances_y)

        if abs(median_size_x - median_size_y) > max_diff:
            if debug:
                logging.warning(
                    f"{base_name}: FAILED Filter 3 (Size Difference). "
                    f"MedX: {median_size_x:.1f}, MedY: {median_size_y:.1f}. (Max Diff: {max_diff})"
                )
            return None

        orig_grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        orig_grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_orig_grad_x = cv2.convertScaleAbs(orig_grad_x)
        abs_orig_grad_y = cv2.convertScaleAbs(orig_grad_y)

        flat_area_ratio = np.sum(
            (abs_orig_grad_x < flat_thresh) & (abs_orig_grad_y < flat_thresh)
        ) / (h * w)

        if flat_area_ratio < flat_ratio:
            if debug:
                logging.warning(
                    f"{base_name}: FAILED Filter 4 (Flatness). "
                    f"Ratio: {flat_area_ratio:.2f}. (Minimum: {flat_ratio})"
                )
            return None

        block_size = int(round((median_size_x + median_size_y) / 2.0))

        if output_dir is not None:
            output_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(output_dir / f"{base_name}_01_grad_x.png"), abs_grad_x)
            cv2.imwrite(str(output_dir / f"{base_name}_02_grad_y.png"), abs_grad_y)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f"Gradient Projections (Tuned) for {base_name}", fontsize=16)
            ax1.plot(sum_x, color="blue")
            ax1.set_title(
                f"Vertical - {len(peaks_x)} peaks (Distance StdDev: {std_x:.2f})"
            )
            ax1.plot(peaks_x, sum_x[peaks_x], "x", color="red", markersize=10)
            ax1.set_xlim(0, w)
            ax2.plot(sum_y, color="green")
            ax2.set_title(
                f"Horizontal - {len(peaks_y)} peaks (Distance StdDev: {std_y:.2f})"
            )
            ax2.plot(peaks_y, sum_y[peaks_y], "x", color="red", markersize=10)
            ax2.set_xlim(0, h)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(str(output_dir / f"{base_name}_03_projections_plot.png"))
            plt.close(fig)
            overlay_img = original_color_img.copy()
            for x in peaks_x:
                cv2.line(overlay_img, (x, 0), (x, h), (0, 255, 0), 1)
            for y in peaks_y:
                cv2.line(overlay_img, (0, y), (w, y), (0, 0, 255), 1)
            cv2.putText(
                overlay_img,
                f"Block Size: ~{block_size}px (StdDev X: {std_x:.2f}, Y: {std_y:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imwrite(str(output_dir / f"{base_name}_04_overlay.png"), overlay_img)

        return block_size

    except Exception as e:
        logging.error(
            f"Error during tuned processing of {base_name}: {e}", exc_info=True
        )
        return None


def process_folder(folder_path: str, tuning_params: dict, debug: bool = False):
    folder = Path(folder_path)
    image_files = (
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.png"))
        + list(folder.glob("*.jpeg"))
    )

    if not image_files:
        logging.warning(f"No image files found in: {folder_path}")
        return

    detected_count = 0
    for f in tqdm(image_files, desc="Analyzing (Fast)"):
        block_size = detect_mosaic_advanced(
            str(f), output_dir=None, tuning_params=tuning_params, debug=debug
        )
        if block_size is not None:
            detected_count += 1
            logging.info(f"Detected mosaic in {f.name} with block size ~{block_size}px")
    logging.info(f"Total detected: {detected_count}/{len(image_files)}")


def process_folder_with_visualization(
    folder_path: str, output_folder: str, tuning_params: dict, debug: bool = False
):
    folder = Path(folder_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True, parents=True)

    image_files = (
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.png"))
        + list(folder.glob("*.jpeg"))
    )

    if not image_files:
        logging.warning(f"No image files found in: {folder_path}")
        return

    detected_count = 0
    for f in tqdm(image_files, desc="Analyzing & Visualizing"):
        block_size = detect_mosaic_advanced(
            str(f), output_dir=output_dir, tuning_params=tuning_params, debug=debug
        )
        if block_size is not None:
            detected_count += 1
            logging.info(f"Detected mosaic in {f.name} with block size ~{block_size}px")


if __name__ == "__main__":
    FOLDER_PATH = "Samples/test_images"
    VISUALIZATION_OUTPUT_PATH = "Samples/visualization_output"

    DEBUG_MODE = False

    TUNING_PARAMETERS = {
        "median_blur_kernel": 3,
        "peak_prominence_factor": 1.0,
        "peak_min_distance": 5,
        "min_peaks_detected": 3,
        "max_std_dev": 65.0,
        "max_size_difference": 20,
        "flatness_gradient_threshold": 15,
        "flatness_min_ratio": 0.15,
    }

    print(f"--- RUNNING VISUALIZATION MODE (Debug: {DEBUG_MODE}) ---")
    process_folder_with_visualization(
        FOLDER_PATH,
        VISUALIZATION_OUTPUT_PATH,
        TUNING_PARAMETERS,
        debug=DEBUG_MODE,
    )
