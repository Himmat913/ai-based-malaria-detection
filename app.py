"""Malaria Detection System — Streamlit entry point.

Flow: upload → decode/validate → (on request) real model inference
→ Grad-CAM → results. Model code lives in utils/, presentation in ui/.
"""

from pathlib import Path

import streamlit as st

from ui import components as ui
from ui.imaging import (
    InvalidImageError,
    decode_upload,
    map_heatmap_to_original,
    preview_data_uri,
    to_data_uri,
    to_png_bytes,
)
from utils.gradcam import generate_gradcam
from utils.model_loader import load_malaria_model
from utils.predictor import predict_image

ROOT = Path(__file__).parent
LOGO_PATH = ROOT / "assets" / "logo.png"
CSS_PATH = ROOT / "ui" / "styles.css"
ACCEPTED_TYPES = ["jpg", "jpeg", "png"]

st.set_page_config(
    page_title="Malaria Detection System",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🦠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def get_model():
    return load_malaria_model()


@st.cache_data(show_spinner=False)
def logo_data_uri():
    if not LOGO_PATH.exists():
        return None
    from PIL import Image

    with Image.open(LOGO_PATH) as logo:
        logo = logo.copy()
    logo.thumbnail((96, 96))
    return to_data_uri(logo)


# ---------------------------------------------------------------- state
state = st.session_state
state.setdefault("uploader_nonce", 0)
state.setdefault("analysis", None)   # {"digest", "result", "heatmap", "heatmap_uri"}
state.setdefault("preview", None)    # (digest, data_uri)


def start_over():
    """Clear the uploader and any result so a new image can be analysed."""
    state.uploader_nonce += 1
    state.analysis = None
    state.preview = None


def upload_notes(upload) -> list[str]:
    notes = []
    longest, shortest = max(upload.width, upload.height), min(upload.width, upload.height)
    if shortest and longest / shortest > 1.3:
        notes.append(
            "This image is not square. The model resizes every input to 224 × 224 "
            "without keeping proportions, so the cell will look stretched to it."
        )
    if longest > 600:
        notes.append(
            "Large image. If it shows a whole smear field rather than one cell, the result "
            "is unlikely to be meaningful: the model was trained on single-cell crops."
        )
    return notes


def run_analysis(upload) -> bool:
    """Run the real model on the uploaded image. Returns True on success.

    Failures are reported below the status box (not inside it, where a
    collapsed status would hide them) with the full exception available.
    """
    stage = "loading the model"
    failure = None
    with st.status("Analyzing image…", expanded=True) as status:
        try:
            status.write("Loading model (the first run downloads the weights from Hugging Face)")
            model = get_model()

            stage = "classifying the image"
            status.write("Classifying the image")
            result = predict_image(model, upload.image)

            stage = "generating the Grad-CAM heatmap"
            status.write("Generating Grad-CAM heatmap")
            overlay = generate_gradcam(model, upload.image)
        except Exception as exc:  # reported to the user below, never swallowed
            failure = exc
            status.update(label=f"Analysis stopped while {stage}", state="error", expanded=False)
        else:
            heatmap = map_heatmap_to_original(overlay, (upload.width, upload.height))
            state.analysis = {
                "digest": upload.digest,
                "result": result,
                "heatmap": heatmap,
                "heatmap_uri": to_data_uri(heatmap),
            }
            status.update(label="Analysis complete", state="complete", expanded=False)

    if failure is not None:
        hint = (
            "On a first run this usually means the weights could not be downloaded "
            "from Hugging Face, so check your connection. "
            if stage == "loading the model"
            else ""
        )
        st.error(
            f"Something went wrong while {stage}. {hint}Press **Analyze image** to try again."
        )
        with st.expander("Technical details"):
            st.exception(failure)
        return False
    return True


# ---------------------------------------------------------------- page
st.markdown(ui.load_css(CSS_PATH), unsafe_allow_html=True)
st.html(ui.nav(logo_data_uri()))
st.html(ui.hero())

upload = None
left, right = st.columns([5, 7], gap="large")

with left:
    st.html(
        ui.section_intro(
            "analyze",
            "Analyze",
            "Upload a cell image",
            "Use one red blood cell cropped from a Giemsa-stained thin smear, the kind of "
            "image the model was trained on. JPG or PNG, up to 20 MB.",
        )
    )
    uploaded_file = st.file_uploader(
        "Cell image",
        type=ACCEPTED_TYPES,
        key=f"uploader_{state.uploader_nonce}",
        help="Drop a file here or browse. Remove it with the × to choose another.",
    )

    if uploaded_file is not None:
        try:
            upload = decode_upload(uploaded_file.name, uploaded_file.getvalue())
        except InvalidImageError as exc:
            st.error(str(exc))

    analysis = state.analysis
    has_result = upload is not None and analysis is not None and analysis["digest"] == upload.digest
    if not has_result:
        state.analysis = None  # drop results that belong to a different (or removed) file

    if upload is not None:
        st.html(ui.file_details(upload, upload_notes(upload)))

    if has_result:
        st.button(
            "Analyze another image",
            type="secondary",
            width="stretch",
            on_click=start_over,
        )
    else:
        analyze_clicked = st.button(
            "Analyze image",
            type="primary",
            width="stretch",
            disabled=upload is None,
            help=None if upload else "Upload an image first",
        )
        if analyze_clicked and upload is not None and run_analysis(upload):
            st.rerun()

with right:
    original_uri = None
    if upload is not None:
        if state.preview is None or state.preview[0] != upload.digest:
            state.preview = (upload.digest, preview_data_uri(upload.image))
        original_uri = state.preview[1]

    heatmap_uri = state.analysis["heatmap_uri"] if has_result else None
    st.html(
        ui.comparison(
            original_uri,
            heatmap_uri,
            upload.name if upload else None,
            (upload.width, upload.height) if upload else None,
        )
    )

    if has_result:
        st.download_button(
            "Download heatmap (PNG)",
            data=to_png_bytes(state.analysis["heatmap"]),
            file_name=f"{Path(upload.name).stem}_gradcam.png",
            mime="image/png",
            width="content",
        )

if has_result:
    st.html(ui.result(state.analysis["result"]))
else:
    st.html(ui.how_it_works())

st.html(ui.model_section())
st.html(ui.about_section())
st.html(ui.footer())
