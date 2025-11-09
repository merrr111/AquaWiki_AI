from fastapi import FastAPI, UploadFile, File, Request
import numpy as np
import json, io, os, traceback
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import mysql.connector
import tempfile

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI(title="Fish Identification API")

# Load the model once
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def get_db_connection():
    return mysql.connector.connect(
        host="srv2088.hstgr.io",      # Hostinger MySQL server
        port=3306,
        user="u915767734_admin",      # Your DB user
        password="Hk76Yg78*",         # Your DB password
        database="u915767734_aquawiki"
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
    try:
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
                try:
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
                except Exception as e:
                    print(f"Skipping fish ID {fish['id']} due to embedding error: {e}")
                    continue

        matches.sort(key=lambda x: x["score"], reverse=True)
        best_match = next((m for m in matches if m["score"] > 0.5), None)
        other_similar = [m for m in matches if m != best_match][:5]

        return {
            "matched_fish": best_match,
            "other_similar_fishes": other_similar
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": "Internal server error", "details": str(e)}

@app.get("/")
async def root():
    return {"message": "Welcome to the Fish Identification API"}

@app.get("/update_fish_data")
async def update_fish_data_get():
    return {"message": "This endpoint only accepts POST requests."}

@app.post("/update_fish_data")
async def update_fish_data(request: Request):
    """Generate embeddings & hashes for a fish after upload (async image download)"""
    import imagehash
    import httpx

    try:
        data = await request.json()
        fish_id = data.get("fish_id")
        image_url = data.get("image_url")
        image_male_url = data.get("image_male_url")

        async def download_image(url):
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")

        def calculate_hash(img):
            return str(imagehash.phash(img))

        def compute_embedding(img):
            img_resized = img.resize((224, 224))
            x = image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return model.predict(x)[0].tolist()

        conn = get_db_connection()
        cursor = conn.cursor()

        # Female image
        female_img = await download_image(image_url)
        female_hash = calculate_hash(female_img)
        female_emb = compute_embedding(female_img)

        # Male image (optional)
        male_hash, male_emb = None, None
        if image_male_url:
            male_img = await download_image(image_male_url)
            male_hash = calculate_hash(male_img)
            male_emb = compute_embedding(male_img)

        # Save to database
        cursor.execute("""
            UPDATE fishes SET
                image_hash = %s,
                image_male_hash = %s,
                embedding = %s,
                embedding_male = %s
            WHERE id = %s
        """, (
            female_hash,
            male_hash,
            json.dumps(female_emb),
            json.dumps(male_emb) if male_emb else None,
            fish_id
        ))
        conn.commit()

        return {"status": "success", "message": f"Updated embeddings and hashes for fish ID {fish_id}"}

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.post("/update_all_fishes")
async def update_all_fishes():
    """
    Update embeddings and hashes for all fishes in the database.
    """
    import httpx

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, image_url, image_male_url FROM fishes")
    fishes = cursor.fetchall()
    cursor.close()
    conn.close()

    results = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for fish in fishes:
            try:
                resp = await client.post(
                    "https://aquawiki-ai-1bh3.onrender.com/update_fish_data",
                    json={
                        "fish_id": fish["id"],
                        "image_url": fish["image_url"],
                        "image_male_url": fish.get("image_male_url") or ""
                    }
                )
                results.append(await resp.json())
            except Exception as e:
                results.append({"fish_id": fish["id"], "status": "error", "message": str(e)})

    return {"status": "success", "results": results}

# Run directly (Render entry point)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fish_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        log_level="info"
    )
