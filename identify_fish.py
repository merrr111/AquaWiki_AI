import sys
import json
import os
import mysql.connector
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

# Load MobileNetV2
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="fishencyclopedia"
    )

def get_embedding(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)[0]

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2 + 1e-10)

def find_similar_fish(image_path, threshold=0.7):
    print("🔍 Identifying fish... Please wait...", file=sys.stderr)
    query_emb = get_embedding(image_path)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, name, scientific_name, origin, image_url, image_male_url,
               embedding, embedding_male, description, male_description
        FROM fishes
    """)
    fishes = cursor.fetchall()
    conn.close()

    best_match = None
    best_score = -1

    for fish in fishes:
        # Female embedding
        if fish.get("embedding"):
            try:
                fish_emb = np.array(json.loads(fish["embedding"]))
                score = cosine_similarity(query_emb, fish_emb)
                if score > best_score:
                    best_score = score
                    best_match = {
                        "id": fish["id"],
                        "name": fish["name"],
                        "scientific_name": fish.get("scientific_name"),
                        "origin": fish.get("origin"),
                        "matched_image_url": fish["image_url"],
                        "match_type": "female",
                        "description": fish.get("description"),
                        "score": float(score)
                    }
            except:
                pass

        # Male embedding
        if fish.get("embedding_male"):
            try:
                fish_emb_male = np.array(json.loads(fish["embedding_male"]))
                score = cosine_similarity(query_emb, fish_emb_male)
                if score > best_score:
                    best_score = score
                    best_match = {
                        "id": fish["id"],
                        "name": fish["name"],
                        "scientific_name": fish.get("scientific_name"),
                        "origin": fish.get("origin"),
                        "matched_image_url": fish.get("image_male_url") or fish["image_url"],
                        "match_type": "male",
                        "description": fish.get("male_description"),
                        "score": float(score)
                    }
            except:
                pass

    return best_match if best_score > threshold else None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python identify_fish.py <image_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]
    result = find_similar_fish(image_path)
    print(json.dumps({"matched_fish": result}, ensure_ascii=False))
