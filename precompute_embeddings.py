import os
import json
import mysql.connector
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load MobileNetV2
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "fishencyclopedia"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)[0].tolist()

def generate_augmented_embeddings(img_path):
    """Generate embeddings for original + rotated/flipped images and average them"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    emb_original = model.predict(x)[0]

    # Convert to tensor for rotations/flips
    x_tensor = tf.convert_to_tensor(x)

    # Rotations: 90, 180, 270 degrees
    emb_rotations = []
    for k in range(1, 4):
        rotated = tf.image.rot90(x_tensor, k=k)
        emb_rot = model.predict(rotated)[0]
        emb_rotations.append(emb_rot)

    # Horizontal flip
    flipped = tf.image.flip_left_right(x_tensor)
    emb_flipped = model.predict(flipped)[0]

    # Average all embeddings
    all_embeddings = [emb_original] + emb_rotations + [emb_flipped]
    avg_emb = np.mean(all_embeddings, axis=0)

    return avg_emb.tolist()

def precompute_embeddings():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT id, image_url, image_male_url, embedding, embedding_male FROM fishes")
    fishes = cursor.fetchall()

    for fish in fishes:
        fish_id = fish["id"]

        # Female embedding
        if not fish["embedding"] and fish["image_url"] and os.path.exists(fish["image_url"]):
            emb = generate_augmented_embeddings(fish["image_url"])
            cursor.execute(
                "UPDATE fishes SET embedding=%s WHERE id=%s",
                (json.dumps(emb), fish_id)
            )
            print(f"Processed female embedding for {fish_id}")

        # Male embedding
        if not fish["embedding_male"] and fish["image_male_url"] and os.path.exists(fish["image_male_url"]):
            emb_male = generate_augmented_embeddings(fish["image_male_url"])
            cursor.execute(
                "UPDATE fishes SET embedding_male=%s WHERE id=%s",
                (json.dumps(emb_male), fish_id)
            )
            print(f"Processed male embedding for {fish_id}")

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Finished saving embeddings to database.")

if __name__ == "__main__":
    precompute_embeddings()
