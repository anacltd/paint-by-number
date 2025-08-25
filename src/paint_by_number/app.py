import cv2
import os
import numpy as np
import streamlit as st
from paint_by_number.utils import (
    resize_max_side,
    edge_preserve,
    kmeans_quantize,
    make_zip,
    make_paint_by_numbers_per_color,
)

MAX_SIDE = 1000
BILATERAL_ITER = 2
EYE_PATCH_SCALE = 0.5
EYE_FEATHER = 1
MIN_REGION_SIZE = 250  # skip tiny regions when placing numbers


with st.sidebar:
    img_input = st.file_uploader(
        "Picture",
        type=["jpg", "jpeg", "png", "webp"],
        help="The picture to generate a paint-by-number version of.",
    )
    k_colours = st.slider(
        "Number of colours to use",
        6,
        24,
        6,
        1,
        help="The total number of colours in the palette.",
    )

if img_input:
    file_bytes = np.frombuffer(img_input.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    bgr = resize_max_side(bgr, MAX_SIDE)
    smoothed = edge_preserve(bgr, BILATERAL_ITER)
    quant, palette = kmeans_quantize(smoothed, k_colours)
    final_img = cv2.medianBlur(quant, 3)

    out_base = os.path.join(".", f"paintable_{k_colours}")

    swatch = np.zeros((60, 60 * k_colours, 3), dtype=np.uint8)
    for i, c in enumerate(palette):
        swatch[:, i * 60 : (i + 1) * 60, :] = c
        cv2.putText(
            swatch,
            str(i + 1),
            (i * 60 + 15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    pbn = make_paint_by_numbers_per_color(
        final_img, palette, min_region_size=MIN_REGION_SIZE
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        st.image(cv2.cvtColor(swatch, cv2.COLOR_BGR2RGB))
    with col2:
        st.image(cv2.cvtColor(pbn, cv2.COLOR_BGR2RGB))

    st.divider()
    results = {"simplified": final_img, "palette": swatch, "paint_by_numbers": pbn}

    zip_file = make_zip(results)

    st.download_button(
        label="Download results",
        data=zip_file,
        file_name="paint_by_number_results.zip",
        mime="application/zip",
    )
