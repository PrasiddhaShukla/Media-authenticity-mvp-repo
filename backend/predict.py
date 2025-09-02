import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix

# Always build path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "notebooks", "models", "classifier.h5")
DATA_PATH = os.path.join(BASE_DIR, "data", "data1")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------
# Function to predict single image
# ---------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize
    
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        print(f"{img_path} → AI-generated (confidence: {prediction:.2f})")
    else:
        print(f"{img_path} → Real (confidence: {1 - prediction:.2f})")

# ---------------------------
# Test with sample images
# ---------------------------
predict_image(os.path.join(DATA_PATH, "AI", "-s-fluffy-fur-and-round-features-immediately-melted-the-viewer-s-heart-photo.jpg"))
predict_image(os.path.join(DATA_PATH, "real", "-Skills-Promo-A-Total-Artist-All-of-North-Wests-Impressive-Drawings-Photos-5.jpg"))

# ---------------------------
# Evaluate on dataset (data/AI and data/real)
# ---------------------------
val_ds = image_dataset_from_directory(
    DATA_PATH,
    image_size=(224, 224),
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
print(classification_report(y_true, y_pred, target_names=["AI", "Real"]))
