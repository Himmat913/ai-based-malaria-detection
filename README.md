# Malaria Detection System

A Streamlit app for malaria cell classification using a VGG19-based deep learning model with Grad-CAM explainability.

## Features

- Real malaria image classification
- Hugging Face model download at runtime
- Grad-CAM heatmap visualization
- Streamlit dashboard
- Solarized-inspired UI theme

## Classes

- `0 = Parasitized`
- `1 = Uninfected`

## Model

The trained model is hosted on Hugging Face:

- `himmat123/malaria-model`
- file: `malaria_model.h5`

## Local setup

Create and activate your virtual environment, then install dependencies:

```bash
pip install -r requirements.txt