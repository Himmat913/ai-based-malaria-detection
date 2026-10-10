# Malaria Detection System

A Streamlit app that classifies single red-blood-cell images from thin blood smears as **Parasitized** or **Uninfected** with a VGG19-based model, and shows a Grad-CAM heatmap of the regions that influenced each prediction.

Research and educational use only. It is not a diagnostic device.

## Run locally

Requires Python 3.10–3.12. TensorFlow 2.16.1 and NumPy 1.26.4 do not support 3.13+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints, usually http://localhost:8501.

### Model weights

The weights are **not** in this repository. On the first analysis the app downloads `malaria_model.h5` (about 80 MB) from the Hugging Face repo [`himmat123/malaria-model`](https://huggingface.co/himmat123/malaria-model) and caches it. The first run therefore needs internet access.

To run offline, download the file once and point the app at it:

```bash
# macOS/Linux
export MALARIA_MODEL_PATH=/path/to/malaria_model.h5
# Windows (PowerShell)
$env:MALARIA_MODEL_PATH="C:\path\to\malaria_model.h5"
```

### Deploying to Streamlit Community Cloud

Push the repository to GitHub and create an app with `app.py` as the entry point. Choose Python 3.12 in the app's advanced settings. The weights download from Hugging Face on first use.

## Using the app

1. Upload a JPG or PNG of a **single cell**, the kind of crop the model was trained on. Whole-field smear photos are outside its training data.
2. Press **Analyze image**.
3. Read the prediction and model score, and compare the original with the Grad-CAM overlay.
4. Download the heatmap if you need it, or press **Analyze another image** to start over.

## Project structure

```
app.py                  Page layout and the upload → analyze → results flow
utils/
  model_loader.py       Downloads/loads the Keras model (cached); optional MALARIA_MODEL_PATH
  predictor.py          Preprocessing (224×224, VGG19 preprocess_input) and prediction
  gradcam.py            Grad-CAM heatmap on block4_conv4
ui/
  components.py         HTML builders for the custom page sections (all user text escaped)
  imaging.py            Display-only image helpers: decoding, previews, heatmap aspect mapping
  styles.css            Layout and custom-block styles
.streamlit/config.toml  Theme (colours, fonts, radii) and client settings
assets/logo.png         App logo, also used as the favicon
tests/test_pipeline.py  Preprocessing, display-mapping and model/Grad-CAM contract checks
REDESIGN_NOTES.md       What changed in the redesign, why, and how it was tested
```

Model code in `utils/` is independent of presentation code in `ui/`. Preview sizes, CSS and browser zoom never affect what the model receives.

## Model summary

| | |
|---|---|
| Backbone | VGG19, ImageNet-initialised, blocks 4–5 fine-tuned |
| Head | Global average pooling → BatchNorm → Dense 256 → Dense 64 → 1 sigmoid unit |
| Input | RGB, resized to 224 × 224, `tf.keras.applications.vgg19.preprocess_input` |
| Output | Sigmoid score = P(Uninfected); ≥ 0.5 → Uninfected, else Parasitized |
| Classes | `0 = Parasitized`, `1 = Uninfected` |
| Explanation | Grad-CAM on `block4_conv4`, Gaussian-smoothed, central circular mask, values < 45 % of the peak suppressed |
| Training data | NIH "Cell Images for Detecting Malaria", 27,558 images, balanced, 70/15/15 split |

The training notebook reports 96.8 % accuracy and 0.993 AUC on 4,134 held-out images from the same dataset. These figures are not clinical validation, and this app does not re-measure them.

## Tests

```bash
python tests/test_pipeline.py          # or: python -m pytest tests
```

The model test uses the Hugging Face weights unless `MALARIA_MODEL_PATH` is set.
