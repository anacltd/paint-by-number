import cv2
import numpy as np
import io
import zipfile


def resize_max_side(img: np.ndarray, max_side=1000) -> np.ndarray:
    """Resize an image to a maximum side length without cropping."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return img


def edge_preserve(img_bgr: np.ndarray, iters=2) -> np.ndarray:
    """Apply bilateral filtering to an image to preserve edges while smoothing."""
    out = img_bgr.copy()
    for _ in range(iters):
        out = cv2.bilateralFilter(out, d=7, sigmaColor=40, sigmaSpace=40)
    return out


def kmeans_quantize(img_bgr: np.ndarray, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the number of colors in an image using k-means clustering.

    Args:
        img_bgr: Input image in BGR format.
        k: Number of colors in the quantized palette. Defaults to 12.

    Returns:
        - Quantized image (same shape as input).
        - Palette of shape (k, 3)
    """
    Z = img_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    quant = centers[labels.flatten()].reshape(img_bgr.shape)
    return quant, centers


def make_paint_by_numbers_per_color(
        quant_img: np.ndarray,
        palette: np.ndarray,
        min_region_size: int = 200
    ) -> np.ndarray:
    """For each palette index, run connected components on its mask separately,
    and place that index number at the centroid of each sufficiently large region.

    Args:
        quant_img: Quantized image (BGR format).
        palette: Color palette of shape (k, 3).
        min_region_size: Minimum pixel area of regions to be labeled. Defaults to 200.

    Returns:
        Paint-by-numbers template with numbered regions and outlines.
    """
    h, w = quant_img.shape[:2]
    edges = cv2.Canny(cv2.cvtColor(quant_img, cv2.COLOR_BGR2GRAY), 50, 150)
    edges = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)
    template = 255 * np.ones((h, w), dtype=np.uint8)
    num_img = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    num_img[edges > 0] = (0, 0, 0)

    # Map each pixel to palette index
    palette_list = [tuple(c) for c in palette.tolist()]
    labels = np.zeros((h, w), dtype=np.uint16)
    for i, c in enumerate(palette_list):
        mask = np.all(quant_img == c, axis=2)
        labels[mask] = i + 1

    # Font size relative to image width
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(1.2, w / 1200.0 * 0.8))
    thick_outline = 2
    thick_inner = 1

    for idx in range(1, len(palette_list) + 1):
        mask = (labels == idx).astype(np.uint8)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for comp_id in range(1, n):  # skip background
            x, y, w_box, h_box, area = stats[comp_id]
            if area < min_region_size:
                continue
            cx, cy = map(int, centroids[comp_id])
            if mask[cy, cx] == 0:
                ys, xs = np.where(mask[y : y + h_box, x : x + w_box] > 0)
                if len(xs) == 0:
                    continue
                cx = x + int(xs.mean())
                cy = y + int(ys.mean())

            text = str(idx)
            cv2.putText(
                num_img,
                text,
                (cx - 5, cy + 5),
                font,
                font_scale,
                (255, 255, 255),
                thick_outline,
                cv2.LINE_AA,
            )
            cv2.putText(
                num_img,
                text,
                (cx - 5, cy + 5),
                font,
                font_scale,
                (0, 0, 0),
                thick_inner,
                cv2.LINE_AA,
            )

    return num_img


def cv2_to_bytes(img) -> bytes | None:
    """Encode an OpenCV image (BGR format) to PNG bytes.

    Args:
        img: Image in BGR format.

    Returns:
        Encoded PNG image as bytes, or None if encoding fails.
    """
    success, buf = cv2.imencode(".png", img)
    return buf.tobytes() if success else None


def make_zip(images_dict: dict[str, np.ndarray]) -> io.BytesIO:
    """Create an in-memory ZIP file containing multiple images.

    Args:
        images_dict: Dictionary mapping filenames (without extension)
            to images in OpenCV BGR format.

    Returns:
        In-memory ZIP file containing all images as PNGs.
    """
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, img in images_dict.items():
            img_bytes = cv2_to_bytes(img)
            zf.writestr(f"{name}.png", img_bytes)
    mem_zip.seek(0)
    return mem_zip
