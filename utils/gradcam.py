import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg19 import preprocess_input

IMG_SIZE = 224


def generate_gradcam(model, image):

    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_rgb = np.array(image)

    processed = preprocess_input(
        img_rgb.astype(np.float32)
    )

    processed = np.expand_dims(
        processed,
        axis=0
    )

    last_conv_layer_name = "block4_conv4"

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.outputs[0]
        ]
    )

    prediction = float(
        model.predict(
            processed,
            verbose=0
        )[0][0]
    )

    target_class = (
        "uninfected"
        if prediction >= 0.5
        else "parasitized"
    )

    threshold = 0.45

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            processed
        )

        score = predictions[:, 0]

        if target_class == "uninfected":
            loss = score
        else:
            loss = -score

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    ).numpy()

    heatmap = cv2.GaussianBlur(
        heatmap,
        (11, 11),
        0
    )

    h, w = heatmap.shape

    Y, X = np.ogrid[:h, :w]

    center_x = w / 2
    center_y = h / 2

    radius = min(h, w) * 0.42

    distance = np.sqrt(
        (X - center_x) ** 2 +
        (Y - center_y) ** 2
    )

    mask = distance <= radius

    heatmap = heatmap * mask

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap[heatmap < threshold] = 0

    heatmap_resized = cv2.resize(
        heatmap,
        (
            img_rgb.shape[1],
            img_rgb.shape[0]
        )
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        img_rgb,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    return Image.fromarray(overlay)