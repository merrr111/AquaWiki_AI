from fastapi import FastAPI, UploadFile, File, Request
import numpy as np
import json, io, os, traceback
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import mysql.connector
import tempfile

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI(title="Fish Identification API")

# Load the model once
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# === Warm up the model ===
print("Warming up MobileNetV2 model...")
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = model.predict(dummy)
print("Model warmed up and ready for predictions!")

def get_db_connection():
    return mysql.connector.connect(
        host="srv2088.hstgr.io",
        port=3306,
        user="u915767734_admin",
        password="Hk76Yg78*",
        database="u915767734_aquawiki"
    )

def get_embedding(img_data):
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    emb = model.predict(x)[0]
    return emb / (np.linalg.norm(emb) + 1e-10)

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    try:
        img_data = await file.read()
        query_emb = get_embedding(img_data)

        # Also load image for histogram comparison
        query_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        query_img = ImageOps.exif_transpose(query_img)

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
                img_url = fish.get(img_field)

                if not emb_str or not img_url:
                    continue

                try:
                    fish_emb = np.array(json.loads(emb_str))
                    fish_emb = fish_emb / (np.linalg.norm(fish_emb) + 1e-10)

                    final_score = cosine_similarity(query_emb, fish_emb)

                    # --- Histogram similarity ---
                    try:
                        # Load fish image
                        from urllib.request import urlopen
                        fish_img = Image.open(urlopen(img_url)).convert("RGB")
                        fish_img = ImageOps.exif_transpose(fish_img)
                        # Resize to same size
                        fish_img_resized = fish_img.resize((224, 224))
                        query_img_resized = query_img.resize((224, 224))
                        # Compute histograms
                        h1 = np.array(query_img_resized.histogram())
                        h2 = np.array(fish_img_resized.histogram())
                        hist_score = float(np.dot(h1, h2) / (np.linalg.norm(h1)*np.linalg.norm(h2) + 1e-10))
                    except:
                        hist_score = 0.0

                    matches.append({
                        "id": fish["id"],
                        "name": fish["name"],
                        "matched_image_url": img_url,
                        "match_type": sex,
                        "description": fish["description"] if sex == "female" else fish.get("male_description"),
                        "score": float(final_score),
                        "histogram_score": hist_score
                    })

                except:
                    continue

        # Sort matches by cosine similarity
        matches.sort(key=lambda x: x["score"], reverse=True)

        best_overall_score = matches[0]["score"] if matches else 0.0

        if best_overall_score < 0.25:
            return {
                "matched_fish": None,
                "other_similar_fishes": [],
                "not_fish": True,
                "reason": "The image does not appear to be a fish."
            }

        # ---- NORMAL FISH IDENTIFICATION ----
        best_match = next((m for m in matches if m["score"] > 0.4), None)

        # Only include other similar fishes with score above 0.5
        min_similarity = 0.5
        other_similar = [m for m in matches if m != best_match and m["score"] > min_similarity][:5]

        return {
            "matched_fish": best_match,
            "other_similar_fishes": other_similar,
            "not_fish": False
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
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img = ImageOps.exif_transpose(img)
                return img

        def calculate_hash(img):
            return str(imagehash.phash(img))

        def compute_embedding(img):
            img_resized = img.resize((224, 224))
            x = image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            emb = model.predict(x)[0]
            return emb / (np.linalg.norm(emb) + 1e-10)

        conn = get_db_connection()
        cursor = conn.cursor()

        female_img = await download_image(image_url)
        female_hash = calculate_hash(female_img)
        female_emb = compute_embedding(female_img)

        male_hash, male_emb = None, None
        if image_male_url:
            male_img = await download_image(image_male_url)
            male_hash = calculate_hash(male_img)
            male_emb = compute_embedding(male_img)

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
            json.dumps(female_emb.tolist()),
            json.dumps(male_emb.tolist()) if male_emb else None,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fish_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        log_level="info"
    )
