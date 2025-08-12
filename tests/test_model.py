import keras_cv
import tensorflow as tf
import keras
import numpy as np

# Construct an EfficientNetV2 from a preset:
efficientnet = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2_b0"
)
images = tf.ones((1, 256, 256, 3))
outputs = efficientnet.predict(images)
efficientnet.save("my_model.keras")
