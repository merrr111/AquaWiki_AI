from fastapi import FastAPI, UploadFile, File
import numpy as np
import json, io, os, tempfile
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import mysql.connector

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI(title="Fish Identification API")

# Load the model once
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def get_db_connection():
    ca_content = os.getenv("CA_CERT")

    # Write the CA cert to a temp file if available
    ca_path = None
    if ca_content:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as temp_ca:
            temp_ca.write(ca_content.encode())
            ca_path = temp_ca.name

    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME"),
        ssl_ca=ca_path if os.getenv("DB_SSL", "false").lower() == "true" else None
    )

def get_embedding(img_data):
    img = Image.open(io.BytesIO(img_data)).convert("RGB").resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)[0]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    img_data = await file.read()
    query_emb = get_embedding(img_data)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM fishes")
    fishes = cursor.fetchall()
    conn.close()

    matches = []

    for fish in fishes:
        for sex, emb_field, img_field in [
            ("female", "embedding", "image_url"),
            ("male", "embedding_male", "image_male_url")
        ]:
            emb_str = fish.get(emb_field)
            if not emb_str:
                continue
            fish_emb = np.array(json.loads(emb_str))
            score = cosine_similarity(query_emb, fish_emb)
            matches.append({
                "id": fish["id"],
                "name": fish["name"],
                "matched_image_url": fish.get(img_field),
                "match_type": sex,
                "description": fish["description"] if sex == "female" else fish.get("male_description"),
                "score": float(score)
            })

    matches.sort(key=lambda x: x["score"], reverse=True)

    best_match = next((m for m in matches if m["score"] > 0.5), None)
    other_similar = [m for m in matches if m != best_match][:5]

    return {
        "matched_fish": best_match,
        "other_similar_fishes": other_similar
    }

# ---------------------------
# Run directly (Render entry point)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fish_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        log_level="info"
    )
