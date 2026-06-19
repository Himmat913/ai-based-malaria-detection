from pathlib import Path

import streamlit as st
from PIL import Image

from utils.gradcam import generate_gradcam
from utils.model_loader import load_malaria_model
from utils.predictor import predict_image

st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🦠",
    layout="wide",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Libre+Baskerville:wght@700&display=swap');

:root {
    --background: #0f1220;
    --foreground: #f3f6ff;
    --card: #171b2e;
    --card-soft: #1d2238;
    --card-foreground: #f3f6ff;

    --sidebar: #111624;
    --sidebar-soft: #171b2e;
    --sidebar-foreground: #f3f6ff;
    --sidebar-border: #2a3147;

    --border: #2a3147;
    --border-strong: #3a4564;

    --primary: #6c8cff;
    --primary-hover: #829dff;
    --primary-foreground: #ffffff;

    --secondary: #38bdf8;
    --secondary-foreground: #07111f;

    --accent: #7dd3fc;
    --accent-foreground: #07111f;

    --muted: #20263a;
    --muted-foreground: #b8c2d9;

    --destructive: #fb7185;
    --destructive-foreground: #ffffff;

    --success: #34d399;
    --warning: #fbbf24;

    --ring: rgba(108, 140, 255, 0.35);
    --radius: 0.75rem;
}

/* Base app polish */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(108, 140, 255, 0.10), transparent 30%),
        var(--background);
    color: var(--foreground);
}

/* Removes the odd Streamlit top decoration/stripe */
[data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stHeader"] {
    background: rgba(15, 18, 32, 0.92) !important;
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(42, 49, 71, 0.45);
}

.main .block-container {
    padding-top: 2.4rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Sidebar: make it feel attached to the dashboard instead of floating separately */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(23, 27, 46, 0.98) 0%, rgba(17, 22, 36, 0.98) 100%) !important;
    border-right: 1px solid var(--sidebar-border);
    box-shadow: 18px 0 40px rgba(0, 0, 0, 0.10);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

[data-testid="stSidebar"] img {
    max-width: 170px;
    margin: 0 auto 1rem auto;
    display: block;
    border-radius: 18px;
    padding: 10px;
    background: rgba(243, 246, 255, 0.04);
    border: 1px solid rgba(243, 246, 255, 0.08);
}

[data-testid="stSidebar"] h3 {
    text-align: center;
    margin-top: 0.25rem;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: var(--muted-foreground) !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(243, 246, 255, 0.035) !important;
    border-color: rgba(184, 194, 217, 0.16) !important;
}

/* Typography */
h1, h2, h3, h4 {
    color: var(--foreground) !important;
    font-weight: 750 !important;
}

p, span, label, li {
    color: var(--foreground);
}

code {
    color: #dbeafe !important;
    background: rgba(108, 140, 255, 0.12) !important;
    border-radius: 6px;
    padding: 0.12rem 0.3rem;
}

.stCaption,
[data-testid="stCaptionContainer"] {
    color: var(--muted-foreground) !important;
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes growBar {
    from { width: 0%; }
}

@keyframes pulseDot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(108, 140, 255, 0.38); }
    50% { box-shadow: 0 0 0 6px rgba(108, 140, 255, 0); }
}

/* Header */
.app-header {
    border: 1px solid rgba(42, 49, 71, 0.78);
    background: rgba(23, 27, 46, 0.76);
    border-radius: 18px;
    padding: 24px 26px;
    margin-bottom: 22px;
    animation: fadeInUp 0.4s ease both;
}

.app-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--muted-foreground);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--primary);
    animation: pulseDot 2.2s ease-in-out infinite;
}

.app-header h1 {
    font-family: 'Libre Baskerville', Georgia, serif;
    font-size: clamp(1.65rem, 3vw, 2.3rem);
    margin: 0 0 8px 0;
    color: var(--foreground) !important;
}

.app-header p {
    color: var(--muted-foreground);
    font-size: 0.98rem;
    max-width: 720px;
    margin: 0;
    line-height: 1.65;
}

/* Tabs: equal spacing and stable sizing */
.stTabs {
    animation: fadeInUp 0.45s ease both;
}

.stTabs [data-baseweb="tab-list"] {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    width: 100%;
    background: rgba(23, 27, 46, 0.78);
    padding: 7px;
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

.stTabs [data-baseweb="tab"] {
    width: 100%;
    justify-content: center;
    border-radius: 12px;
    color: var(--muted-foreground) !important;
    font-weight: 700;
    padding: 0.7rem 0.9rem;
    transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
}

.stTabs [data-baseweb="tab"] p {
    color: inherit !important;
    width: 100%;
    text-align: center;
    font-weight: 700;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(108, 140, 255, 0.10);
    color: var(--foreground) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: var(--primary-foreground) !important;
    box-shadow: 0 8px 22px rgba(108, 140, 255, 0.18);
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.25rem;
}

/* Stepper */
.stepper {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 0.6rem 0 1.5rem 0;
    flex-wrap: wrap;
    animation: fadeInUp 0.45s ease both;
}

.step-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 2px;
}

.step-dot {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.76rem;
    font-weight: 800;
    border: 1.5px solid var(--border);
    color: var(--muted-foreground);
    background: var(--card);
    transition: all 0.3s ease;
}

.step-pill span:last-child {
    font-size: 0.86rem;
    font-weight: 650;
    color: var(--muted-foreground);
    transition: color 0.3s ease;
}

.step-active .step-dot {
    background: var(--primary);
    border-color: var(--primary);
    color: var(--primary-foreground);
    box-shadow: 0 0 0 4px rgba(108, 140, 255, 0.16);
}

.step-active span:last-child,
.step-done span:last-child {
    color: var(--foreground);
}

.step-done .step-dot {
    background: rgba(52, 211, 153, 0.16);
    border-color: rgba(52, 211, 153, 0.75);
    color: #8af7c8;
}

.step-line {
    width: 48px;
    height: 2px;
    background: var(--border);
    margin: 0 8px;
    transition: background 0.3s ease;
}

.step-line-done {
    background: rgba(52, 211, 153, 0.72);
}

/* Cards */
.card {
    background: rgba(23, 27, 46, 0.86);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    transition: border-color 0.2s ease, transform 0.2s ease;
    animation: fadeInUp 0.4s ease both;
}

.card:hover {
    border-color: var(--border-strong);
    transform: translateY(-1px);
}

.label {
    font-size: 0.72rem;
    color: var(--muted-foreground);
    margin-bottom: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 800;
}

.value {
    font-size: 1.08rem;
    font-weight: 750;
    color: var(--foreground);
    line-height: 1.35;
}

.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.meta-chip {
    background: rgba(243, 246, 255, 0.06);
    border: 1px solid rgba(184, 194, 217, 0.18);
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.75rem;
    color: var(--muted-foreground);
    font-weight: 650;
}

.placeholder-card {
    background: rgba(23, 27, 46, 0.70);
    border: 1.5px dashed var(--border-strong);
    border-radius: var(--radius);
    padding: 56px 20px;
    text-align: center;
    color: var(--muted-foreground);
    animation: fadeInUp 0.4s ease both;
}

.placeholder-icon {
    font-size: 1.8rem;
    margin-bottom: 10px;
    opacity: 0.82;
    color: var(--accent);
}

.placeholder-text {
    font-size: 0.9rem;
    color: var(--muted-foreground);
}

/* Results */
.result-banner {
    border-radius: var(--radius);
    padding: 22px 24px;
    border: 1px solid;
    margin: 18px 0 16px 0;
    animation: fadeInUp 0.4s ease both;
}

.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 0.92rem;
    font-weight: 800;
    border: 1px solid;
}

.result-meta {
    color: var(--muted-foreground);
    font-size: 0.84rem;
    margin-top: 10px;
}

.conf-track {
    width: 100%;
    height: 10px;
    background: rgba(15, 18, 32, 0.78);
    border: 1px solid var(--border);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 16px;
}

.conf-fill {
    height: 100%;
    border-radius: 999px;
    animation: growBar 1s cubic-bezier(.22,1,.36,1);
}

.conf-label {
    margin-top: 8px;
    color: var(--muted-foreground);
    font-size: 0.82rem;
    font-weight: 650;
}

.recommendation-card {
    background: rgba(23, 27, 46, 0.86);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 16px 18px;
    margin-top: 14px;
    animation: fadeInUp 0.45s ease both;
}

.recommendation-card.is-flagged {
    border-left-color: var(--destructive);
}

/* Streamlit components */
div[data-testid="stMetric"] {
    background: rgba(23, 27, 46, 0.86);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: var(--border-strong);
    transform: translateY(-1px);
}

[data-testid="stMetricLabel"] {
    color: var(--muted-foreground) !important;
}

[data-testid="stMetricValue"] {
    color: var(--foreground) !important;
}

.stButton button,
.stDownloadButton button {
    background: var(--primary) !important;
    color: var(--primary-foreground) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: var(--radius) !important;
    padding: 0.68rem 1.4rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
    transition: all 0.2s ease;
    box-shadow: 0 10px 24px rgba(108, 140, 255, 0.18);
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: var(--primary-hover) !important;
    transform: translateY(-1px);
    box-shadow: 0 14px 28px rgba(108, 140, 255, 0.24);
}

.stButton button:focus,
.stDownloadButton button:focus,
button:focus {
    box-shadow: 0 0 0 4px var(--ring) !important;
    outline: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(23, 27, 46, 0.82) !important;
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s ease, background 0.2s ease;
    padding: 1rem !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--primary) !important;
    background: rgba(108, 140, 255, 0.06) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--muted-foreground) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] small {
    display: none !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: rgba(243, 246, 255, 0.06) !important;
    color: var(--foreground) !important;
    border: 1px solid var(--border-strong) !important;
    box-shadow: none !important;
}

/* Shared container style */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important;
    border-radius: var(--radius) !important;
    background-color: rgba(23, 27, 46, 0.76);
}

/* Expander */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary {
    background: rgba(23, 27, 46, 0.90) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--foreground) !important;
    font-weight: 700 !important;
}

[data-testid="stExpander"] {
    border-radius: var(--radius);
    border-color: var(--border) !important;
}

/* Alert/status */
[data-testid="stAlert"] {
    background: rgba(23, 27, 46, 0.90) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    color: var(--foreground) !important;
}

[data-testid="stStatusWidget"] {
    color: var(--muted-foreground) !important;
}

/* Pipeline */
.pipe-row {
    display: flex;
    align-items: stretch;
    gap: 10px;
    overflow-x: auto;
    padding: 4px 2px 14px 2px;
}

.pipe-card {
    flex: 1 1 150px;
    min-width: 145px;
    background: rgba(23, 27, 46, 0.86);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 14px;
    text-align: center;
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.pipe-card:hover {
    border-color: var(--border-strong);
    transform: translateY(-1px);
}

.pipe-icon {
    font-size: 1.55rem;
    margin-bottom: 8px;
}

.pipe-title {
    font-weight: 750;
    color: var(--foreground);
    font-size: 0.89rem;
    margin-bottom: 5px;
}

.pipe-desc {
    color: var(--muted-foreground);
    font-size: 0.76rem;
    line-height: 1.45;
}

.pipe-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted-foreground);
    font-size: 1.25rem;
    font-weight: 800;
}

.disclaimer-card {
    background: rgba(23, 27, 46, 0.86);
    border: 1px solid var(--border);
    border-left: 3px solid var(--destructive);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin-bottom: 12px;
}

.footer-note {
    text-align: center;
    color: var(--muted-foreground);
    font-size: 0.76rem;
    padding: 22px 0 4px 0;
}

::-webkit-scrollbar {
    height: 8px;
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--background);
}

::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 8px;
}

/* Smaller screens */
@media (max-width: 768px) {
    .pipe-arrow {
        display: none;
    }

    .app-header {
        padding: 20px;
    }

    .app-header h1 {
        font-size: 1.55rem;
    }

    .step-line {
        width: 22px;
    }

    .stTabs [data-baseweb="tab-list"] {
        grid-template-columns: 1fr;
    }

    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
}

/* Custom command box */
.custom-code-box {
    background: #171b2e;
    border: 1px solid #2a3147;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 8px;
    margin-bottom: 16px;
    font-family: Consolas, Monaco, "Courier New", monospace;
}

.custom-code-box code {
    color: #dbeafe;
    background: transparent;
    font-size: 0.92rem;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

STATUS = {
    "Parasitized": {"color": "#fb7185", "icon": "⚠", "verdict": "Positive", "summary": "Parasite Detected"},
    "Uninfected": {"color": "#34d399", "icon": "✓", "verdict": "Negative", "summary": "No Parasite Detected"},
}


def workflow_stepper(stage):
    steps = ["Upload", "Analyze", "Results"]
    parts = []
    for i, label in enumerate(steps, start=1):
        state = "done" if i < stage else ("active" if i == stage else "pending")
        parts.append(f'<div class="step-pill step-{state}"><span class="step-dot">{i}</span><span>{label}</span></div>')
        if i < len(steps):
            line_state = "done" if i < stage else "pending"
            parts.append(f'<div class="step-line step-line-{line_state}"></div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def format_bytes(n):
    try:
        n = float(n)
    except Exception:
        return "Unknown size"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def render_pipeline(steps):
    parts = []
    for i, (icon, title, desc) in enumerate(steps):
        parts.append(
            f'<div class="pipe-card">'
            f'<div class="pipe-icon">{icon}</div>'
            f'<div class="pipe-title">{title}</div>'
            f'<div class="pipe-desc">{desc}</div>'
            f'</div>'
        )
        if i < len(steps) - 1:
            parts.append('<div class="pipe-arrow">→</div>')
    st.markdown(f'<div class="pipe-row">{"".join(parts)}</div>', unsafe_allow_html=True)


def get_recommendation(label):
    if label == "Parasitized":
        return (
            "Parasite features were detected in this sample. Confirm this result with microscopy or an RDT at a "
            "certified laboratory and consult a healthcare provider promptly, particularly if symptoms such as "
            "fever, chills, or fatigue are present.",
            True,
        )
    return (
        "No parasite features were detected by the model. If clinical symptoms are present, confirmatory "
        "laboratory testing is still advised, as this tool is a screening aid and not a diagnostic substitute.",
        False,
    )


@st.cache_resource(show_spinner=False)
def get_model():
    return load_malaria_model()


logo_path = Path("assets/logo.png")

with st.sidebar:
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    st.markdown("### Malaria Detection System")
    st.caption("VGG19 transfer learning with Grad-CAM explainability.")

    with st.container(border=True):
        st.markdown(
            '<div class="label">Model source</div>'
            '<div class="value" style="font-size:0.9rem; font-family:monospace;">himmat123/malaria-model</div>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown(
            '<div class="label">Classes</div>'
            '<span class="meta-chip" style="border-color:#fb718566; color:#fb7185;">0 · Parasitized</span>&nbsp;'
            '<span class="meta-chip" style="border-color:#34d39966; color:#34d399;">1 · Uninfected</span>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown(
            '<div class="label">Input size</div>'
            '<div class="value" style="font-size:0.9rem;">224 × 224 × 3</div>',
            unsafe_allow_html=True,
        )

    st.info("The model downloads from Hugging Face the first time the app runs.")

st.markdown(
    '<div class="app-header">'
    '<div class="app-eyebrow"><span class="status-dot"></span>AI-Assisted Diagnostic Tool</div>'
    '<h1>Malaria Cell Detection</h1>'
    '<p>Upload a blood smear image to receive an AI-generated parasite classification with a Grad-CAM visual explanation.</p>'
    '</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Info", "About"])

with tab1:
    uploaded_file = st.file_uploader("Upload a blood smear image", type=["jpg", "jpeg", "png"])

    has_file = uploaded_file is not None
    has_result = (
        has_file
        and st.session_state.get("malaria_file") == uploaded_file.name
        and st.session_state.get("malaria_result") is not None
    )
    stage = 3 if has_result else (2 if has_file else 1)
    workflow_stepper(stage)

    if has_file:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns(2)

        with left:
            with st.container(border=True):
                st.image(image, caption="Uploaded image", use_container_width=True)
                st.markdown(
                    f'<div class="meta-row">'
                    f'<span class="meta-chip">{uploaded_file.name}</span>'
                    f'<span class="meta-chip">{image.size[0]} × {image.size[1]} px</span>'
                    f'<span class="meta-chip">{format_bytes(uploaded_file.size)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        analyze = st.button("Analyze image", type="primary", use_container_width=True)

        if analyze:
            with st.status("Running diagnostic analysis", expanded=True) as status:
                status.write("Loading model")
                model = get_model()
                status.write("Preprocessing image")
                status.write("Running inference")
                result = predict_image(model, image)
                status.write("Generating Grad-CAM heatmap")
                gradcam_image = generate_gradcam(model, image)
                status.update(label="Analysis complete", state="complete", expanded=False)
            st.session_state["malaria_result"] = result
            st.session_state["malaria_gradcam"] = gradcam_image
            st.session_state["malaria_file"] = uploaded_file.name
            st.rerun()

        with right:
            if has_result:
                with st.container(border=True):
                    st.image(st.session_state["malaria_gradcam"], caption="Grad-CAM heatmap", use_container_width=True)
            else:
                st.markdown(
                    '<div class="placeholder-card">'
                    '<div class="placeholder-icon">◎</div>'
                    '<div class="placeholder-text">Grad-CAM heatmap will appear here after analysis</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        if has_result:
            result = st.session_state["malaria_result"]
            label = result["label"]
            confidence = result["confidence"] * 100
            prob_uninfected = result["prob_uninfected"] * 100
            info = STATUS.get(label, {"color": "#7a88a1", "icon": "•", "verdict": "Unknown", "summary": label})
            color = info["color"]

            st.markdown(
                f'<div class="result-banner" style="background:{color}14; border-color:{color}55;">'
                f'<span class="result-badge" style="background:{color}26; border-color:{color}66; color:{color};">{info["icon"]} {info["verdict"]} — {label}</span>'
                f'<div class="result-meta">{info["summary"]}</div>'
                f'<div class="conf-track"><div class="conf-fill" style="width:{confidence:.2f}%; background:{color};"></div></div>'
                f'<div class="conf-label">{confidence:.2f}% model confidence</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Class", label)
            with m2:
                st.metric("Confidence", f"{confidence:.2f}%")
            with m3:
                st.metric("P(Uninfected)", f"{prob_uninfected:.2f}%")

            rec_text, flagged = get_recommendation(label)
            flag_class = " is-flagged" if flagged else ""
            st.markdown(
                f'<div class="recommendation-card{flag_class}">'
                f'<div class="label">Recommendation</div>'
                f'<div style="color:var(--foreground); font-size:0.92rem; line-height:1.6;">{rec_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                st.markdown(
                    '<div class="label">Analysis details</div>'
                    '<div style="color:var(--foreground); font-size:0.9rem; line-height:1.65;">'
                    'The Grad-CAM heatmap highlights the regions of the cell that most influenced the prediction. '
                    'Warmer colors indicate areas the model weighted more heavily when distinguishing Parasitized '
                    'from Uninfected.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="disclaimer-card">'
                '<div class="label" style="margin-bottom:4px;">Important</div>'
                '<div style="color:var(--foreground); font-size:0.88rem; line-height:1.6;">'
                'This tool is for research and educational purposes only and is not a substitute for professional '
                'medical diagnosis.'
                '</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="placeholder-card">'
            '<div class="placeholder-icon">⤴</div>'
            '<div class="placeholder-text">Upload a blood smear image above to begin analysis</div>'
            '</div>',
            unsafe_allow_html=True,
        )

with tab2:
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="card"><div class="label">Architecture</div><div class="value">VGG19-based binary classifier</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="label">Preprocessing</div><div class="value">VGG19 preprocess_input</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="card"><div class="label">Grad-CAM layer</div><div class="value">block4_conv4</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    st.markdown("### Inference pipeline")
    render_pipeline([
        ("📤", "Upload", "Blood smear image (JPG/PNG)"),
        ("🧹", "Preprocess", "Resize to 224×224, VGG19 preprocess_input"),
        ("🧠", "VGG19 Backbone", "Convolutional feature extraction"),
        ("🎯", "Sigmoid Output", "Single neuron binary probability"),
        ("🏷️", "Class Label", "Parasitized or Uninfected"),
    ])

    st.markdown("### Model details")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="card"><div class="label">Input size</div><div class="value">224 × 224 × 3</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="label">Output layer</div><div class="value">Single sigmoid neuron</div></div>', unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="card"><div class="label">Class mapping</div><div class="value" style="font-size:0.95rem;">0 = Parasitized · 1 = Uninfected</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="label">Hosting</div><div class="value" style="font-size:0.95rem;">Hugging Face (downloaded on demand)</div></div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### Project overview")
    st.markdown(
        '<div class="card">'
        'This app performs real malaria cell inference using the trained model stored on Hugging Face.'
        '<ul style="margin-top:10px; line-height:1.8;">'
        '<li>Downloads the <code>.h5</code> model at runtime</li>'
        '<li>Preprocesses the image exactly like the training notebook</li>'
        '<li>Produces a real binary prediction</li>'
        '<li>Generates a Grad-CAM heatmap for explainability</li>'
        '</ul></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    st.markdown("### Run locally")
    st.markdown(
        '<div class="custom-code-box"><code>streamlit run app.py</code></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Deployment")
    st.markdown(
        '<div class="card">Push the code to GitHub, connect the repo to Streamlit Cloud, and keep the model on Hugging Face.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    st.markdown("### Frequently asked questions")
    with st.expander("What does a Positive (Parasitized) result mean?"):
        st.write("The model detected visual features in the cell image that are consistent with the presence of a malaria parasite.")
    with st.expander("What does a Negative (Uninfected) result mean?"):
        st.write("The model did not detect the visual features it associates with parasitized cells.")
    with st.expander("Is this a substitute for medical diagnosis?"):
        st.write("No. This app is intended for research and educational purposes only and should never replace a certified laboratory diagnosis.")

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer-card">'
        '<div class="label" style="margin-bottom:4px;">Disclaimer</div>'
        '<div style="color:var(--foreground); font-size:0.88rem; line-height:1.6;">'
        'This tool is for research and educational purposes only and is not a substitute for professional medical diagnosis.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="footer-note">VGG19 · Grad-CAM · Streamlit</div>', unsafe_allow_html=True)