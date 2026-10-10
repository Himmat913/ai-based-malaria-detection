"""Small HTML builders for the parts of the page Streamlit widgets can't express.

Every value that comes from the user (file names) or the model is escaped.
All blocks are rendered with ``st.html`` so no Markdown processing is applied.
"""

from html import escape
from pathlib import Path

GITHUB_URL = "https://github.com/Himmat913/ai-based-malaria-detection"

CLASS_STYLE = {
    "Parasitized": {
        "tone": "alert",
        "marker": "●",
        "headline": "Parasite features detected",
        "heatmap": (
            "Red and yellow areas pushed the model most strongly towards "
            "<em>Parasitized</em>; blue areas contributed little."
        ),
        "next_step": (
            "Confirm with microscopy or a rapid diagnostic test and consult a "
            "healthcare provider, especially if there is fever, chills or fatigue."
        ),
    },
    "Uninfected": {
        "tone": "clear",
        "marker": "○",
        "headline": "No parasite features detected",
        "heatmap": (
            "Brighter green areas contributed most to the <em>Uninfected</em> "
            "prediction; deep blue areas contributed little."
        ),
        "next_step": (
            "If symptoms are present, confirmatory laboratory testing is still "
            "advised. A negative model output does not rule out malaria."
        ),
    },
}


def load_css(path: Path) -> str:
    return f"<style data-mx-styles>{path.read_text(encoding='utf-8')}</style>"


def nav(logo_uri: str | None) -> str:
    logo = (
        f'<img class="mx-brand__logo" src="{logo_uri}" alt="" width="28" height="31">'
        if logo_uri
        else ""
    )
    return (
        '<nav class="mx-nav" aria-label="Primary">'
        f'<a class="mx-brand" href="#analyze">{logo}<span>Malaria Detection</span></a>'
        '<div class="mx-nav__links">'
        '<a href="#analyze">Analyze</a>'
        '<a href="#model">Model</a>'
        '<a href="#about">About</a>'
        "</div>"
        f'<a class="mx-pill mx-pill--dark" href="{GITHUB_URL}" target="_blank" '
        'rel="noopener noreferrer">Source<span aria-hidden="true"> ↗</span></a>'
        "</nav>"
    )


def hero() -> str:
    return (
        '<header class="mx-hero">'
        '<p class="mx-label">Research tool · Thin blood-smear cell images</p>'
        '<div class="mx-hero__row">'
        "<h1>Malaria cell analysis</h1>"
        '<p class="mx-lede">Upload an image of a single red blood cell. A VGG19-based model '
        "classifies it as <strong>Parasitized</strong> or <strong>Uninfected</strong>, and a "
        "Grad-CAM heatmap shows which regions shaped that decision.</p>"
        "</div>"
        "</header>"
    )


def section_intro(anchor: str, label: str, title: str, text: str) -> str:
    return (
        f'<div class="mx-section-intro" id="{anchor}">'
        f'<p class="mx-label">{escape(label)}</p>'
        f"<h2>{escape(title)}</h2>"
        f'<p class="mx-muted">{text}</p>'
        "</div>"
    )


def file_details(upload, notes: list[str]) -> str:
    from ui.imaging import format_bytes  # local import keeps this module light

    rows = [
        ("File", f'<span class="mx-filename">{escape(upload.name)}</span>'),
        ("Dimensions", f"{upload.width} × {upload.height} px"),
        ("Size", format_bytes(upload.size_bytes)),
        ("Format", f"{escape(upload.source_format)} · {escape(upload.source_mode)}"),
    ]
    items = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    note_html = "".join(f'<p class="mx-note">{n}</p>' for n in notes)
    return f'<dl class="mx-specs mx-specs--compact">{items}</dl>{note_html}'


def _stage(label: str, body: str, caption: str, ratio: str) -> str:
    return (
        '<figure class="mx-figure">'
        f'<figcaption class="mx-label">{label}</figcaption>'
        f'<div class="mx-stage" style="aspect-ratio: {ratio}">{body}</div>'
        f'<p class="mx-caption">{caption}</p>'
        "</figure>"
    )


def stage_ratio(size) -> str:
    """Stage proportions follow the image, clamped so extreme shapes stay usable.

    Both panels share the ratio, so the original and the heatmap always line up.
    The image inside is still `object-fit: contain`, so nothing is ever cropped.
    """
    if not size:
        return "1 / 1"
    width, height = size
    ratio = min(max(width / height, 3 / 4), 4 / 3)
    return f"{ratio:.4f} / 1"


def comparison(
    original_uri: str | None,
    heatmap_uri: str | None,
    file_name: str | None,
    image_size=None,
) -> str:
    ratio = stage_ratio(image_size)
    if original_uri:
        original = (
            f'<img src="{original_uri}" alt="Uploaded cell image {escape(file_name or "")}">'
        )
        original_caption = "As uploaded, proportions preserved."
    else:
        original = '<span class="mx-stage__empty">Your image will appear here</span>'
        original_caption = "JPG or PNG"

    if heatmap_uri:
        heat = (
            f'<img class="mx-reveal" src="{heatmap_uri}" '
            'alt="Grad-CAM heatmap overlaid on the uploaded cell image">'
        )
        heat_caption = "Computed at 224 × 224, mapped back to the original proportions."
    else:
        heat = '<span class="mx-stage__empty">Heatmap appears after analysis</span>'
        heat_caption = "Regions that influenced the prediction"

    return (
        '<div class="mx-compare">'
        f"{_stage('Original', original, original_caption, ratio)}"
        f"{_stage('Grad-CAM', heat, heat_caption, ratio)}"
        "</div>"
    )


def how_it_works() -> str:
    steps = [
        ("01", "Upload", "Add a single-cell crop from a thin blood smear, as a JPG or PNG."),
        (
            "02",
            "Analyze",
            "The model resizes the image to 224 × 224 and scores it between "
            "Parasitized and Uninfected.",
        ),
        (
            "03",
            "Compare",
            "Read the label and score, then check the heatmap to see which regions drove it.",
        ),
    ]
    cols = "".join(
        f'<div class="mx-col"><span class="mx-numeral">{n}</span><h3>{t}</h3>'
        f'<p class="mx-muted">{d}</p></div>'
        for n, t, d in steps
    )
    return f'<div class="mx-band mx-grid3" aria-label="How it works">{cols}</div>'


def result(result: dict) -> str:
    label = result["label"]
    style = CLASS_STYLE.get(label)
    if style is None:  # unexpected label: show it plainly rather than guess
        style = {"tone": "neutral", "marker": "•", "headline": label, "heatmap": "", "next_step": ""}

    score = result["confidence"] * 100
    p_uninfected = result["prob_uninfected"]

    return (
        f'<div class="mx-band mx-result mx-reveal" data-tone="{style["tone"]}" role="status" aria-live="polite">'
        '<div class="mx-grid3">'
        '<div class="mx-col">'
        '<p class="mx-label">Prediction</p>'
        f'<p class="mx-verdict"><span class="mx-verdict__marker" aria-hidden="true">{style["marker"]}</span>'
        f"{escape(label)}</p>"
        f'<p class="mx-muted">{style["headline"]}.</p>'
        "</div>"
        '<div class="mx-col">'
        '<p class="mx-label">Model score</p>'
        f'<p class="mx-score">{score:.1f}<span>%</span></p>'
        f'<div class="mx-meter" role="img" aria-label="Score {score:.1f} percent for {escape(label)}">'
        f'<span style="width:{score:.2f}%"></span></div>'
        f'<p class="mx-muted mx-small">Sigmoid output P(Uninfected) = <code>{p_uninfected:.4f}</code>, '
        "threshold 0.5. A model score, not a calibrated probability of infection.</p>"
        "</div>"
        '<div class="mx-col">'
        '<p class="mx-label">Reading the heatmap</p>'
        f'<p class="mx-muted">{style["heatmap"]} The overlay is limited to the central '
        "region of the image, where the cell normally sits.</p>"
        "</div>"
        "</div>"
        '<p class="mx-advice"><strong>Not a diagnosis.</strong> '
        f'{style["next_step"]}</p>'
        "</div>"
    )


def model_section() -> str:
    rows = [
        ("Backbone", "VGG19, ImageNet-initialised; blocks 4–5 fine-tuned"),
        (
            "Classifier head",
            "Global average pooling → batch norm → dense 256 → dense 64 → one sigmoid unit",
        ),
        ("Input", "RGB, resized to 224 × 224, VGG19 <code>preprocess_input</code>"),
        (
            "Output",
            "Sigmoid score = P(Uninfected). ≥ 0.5 → Uninfected, otherwise Parasitized",
        ),
        ("Labels", "0 · Parasitized &nbsp;&nbsp; 1 · Uninfected"),
        (
            "Explanation",
            "Grad-CAM on <code>block4_conv4</code>; smoothed, limited to the central region, "
            "values below 45% of the peak suppressed",
        ),
        (
            "Weights",
            '<a href="https://huggingface.co/himmat123/malaria-model" target="_blank" '
            'rel="noopener noreferrer">himmat123/malaria-model</a> · <code>malaria_model.h5</code>, '
            "downloaded on first run",
        ),
        (
            "Training data",
            "NIH “Cell Images for Detecting Malaria”: 27,558 single-cell images, "
            "balanced, split 70 / 15 / 15",
        ),
        (
            "Notebook test result",
            "Accuracy 96.8%, AUC 0.993 on 4,134 held-out images from the same dataset. "
            '<span class="mx-muted">Reported in the training notebook; not clinical validation, '
            "and not re-measured by this app.</span>",
        ),
    ]
    items = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return (
        '<section class="mx-split" id="model" aria-labelledby="model-title">'
        "<div>"
        '<p class="mx-label">Model</p>'
        '<h2 id="model-title">What runs behind the result</h2>'
        '<p class="mx-muted">A convolutional network fine-tuned for one task: telling parasitized '
        "and uninfected single-cell images apart. Details come from this app’s source and its "
        "training notebook.</p>"
        "</div>"
        f'<dl class="mx-specs">{items}</dl>'
        "</section>"
    )


def about_section() -> str:
    limits = [
        (
            "Not a diagnostic device.",
            "Use it for research and education. Any result must be confirmed by microscopy "
            "or a rapid diagnostic test.",
        ),
        (
            "Single cells only.",
            "The model was trained on segmented single-cell crops. Whole-field smear photos, "
            "other stains or other imaging set-ups are outside its training data.",
        ),
        (
            "Scores are not probabilities of disease.",
            "The percentage is the model’s sigmoid output for the predicted class.",
        ),
        (
            "Heatmaps show attention, not proof.",
            "Grad-CAM indicates where the model looked, not that a parasite is present there.",
        ),
    ]
    items = "".join(
        f'<li><span class="mx-index">{i:02d}</span><p><strong>{t}</strong> {d}</p></li>'
        for i, (t, d) in enumerate(limits, start=1)
    )
    return (
        '<section class="mx-split" id="about" aria-labelledby="about-title">'
        "<div>"
        '<p class="mx-label">About</p>'
        '<h2 id="about-title">Know the limits</h2>'
        '<p class="mx-muted">This app pairs a trained classifier with a visual explanation so '
        "each prediction can be inspected rather than taken on trust.</p>"
        "</div>"
        f'<ol class="mx-list">{items}</ol>'
        "</section>"
    )


def footer() -> str:
    return (
        '<footer class="mx-footer">'
        '<div class="mx-footer__links">'
        '<a href="#analyze">Analyze</a><a href="#model">Model</a><a href="#about">About</a>'
        f'<a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">GitHub ↗</a>'
        "</div>"
        '<div class="mx-footer__meta">'
        "<span>VGG19 · Grad-CAM · Streamlit</span>"
        "<span>Research and educational use only</span>"
        "</div>"
        "</footer>"
    )
