import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg19 import preprocess_input

IMAGE_SIZE = (224, 224)


def prepare_image(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image).astype(np.float32)
    processed = preprocess_input(image_array)
    batch = np.expand_dims(processed, axis=0)
    return batch


def predict_image(model, image):
    batch = prepare_image(image)
    probability_uninfected = float(model.predict(batch, verbose=0)[0][0])

    if probability_uninfected >= 0.5:
        label = "Uninfected"
        confidence = probability_uninfected
    else:
        label = "Parasitized"
        confidence = 1.0 - probability_uninfected

    return {
        "label": label,
        "confidence": confidence,
        "prob_uninfected": probability_uninfected,
    }