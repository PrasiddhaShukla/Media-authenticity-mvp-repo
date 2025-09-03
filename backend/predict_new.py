import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "notebooks", "models", "new_classifier.h5")
DATA_PATH = os.path.join(BASE_DIR, "data", "data2")

# ---------------------------
# Load model
# ---------------------------
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (160, 160)

# ---------------------------
# Predict a single image
# ---------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        print(f"{img_path} → REAL (confidence: {prediction:.4f})")
        return "REAL", prediction
    else:
        print(f"{img_path} → FAKE (confidence: {1 - prediction:.4f})")
        return "FAKE", prediction

# ---------------------------
# CLI mode (predict single image) OR evaluation
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"❌ File not found: {img_path}")
            sys.exit(1)
        predict_image(img_path)
    else:
        # ---------------------------
        # Test with sample images
        # ---------------------------
        predict_image(os.path.join(DATA_PATH, "test", "FAKE", "3 (9).jpg"))
        predict_image(os.path.join(DATA_PATH, "test", "REAL", "0003.jpg"))

        # ---------------------------
        # Evaluate on dataset
        # ---------------------------
        val_ds = image_dataset_from_directory(
            DATA_PATH,
            image_size=IMG_SIZE,
            batch_size=32
        ).prefetch(buffer_size=tf.data.AUTOTUNE)

        val_loss, val_acc = model.evaluate(val_ds)
        print(f"Validation Accuracy: {val_acc:.2f}")

        # ---------------------------
        # Classification report
        # ---------------------------
        y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
        y_pred = (model.predict(val_ds) > 0.5).astype("int32").flatten()

        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=["FAKE", "REAL"]))
