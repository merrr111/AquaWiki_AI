from fastapi import FastAPI, UploadFile, File, Request
import numpy as np
import json, io, os, traceback
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import mysql.connector
import tempfile

# New imports
import cv2
import requests

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI(title="Fish Identification API")

# Embedding model (existing) - use for embeddings (no top)
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Classification model (for fish / not-fish checks) - includes top
clf_model = MobileNetV2(weights="imagenet", include_top=True)

# === Warm up the models on startup to prevent first-upload delay ===
print("Warming up MobileNetV2 models...")
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = model.predict(dummy)
_ = clf_model.predict(dummy)  # warm up classifier head
print("Models warmed up and ready for predictions!")

def get_db_connection():
    return mysql.connector.connect(
        host="srv2088.hstgr.io",
        port=3306,
        user="u915767734_admin",
        password="Hk76Yg78*",
        database="u915767734_aquawiki"
    )

def get_embedding(img_data):
    # Fix EXIF rotation
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    emb = model.predict(x)[0]
    return emb / (np.linalg.norm(emb) + 1e-10)  # normalize

def get_color_histogram(img, bins=16):
    """Compute normalized RGB histogram"""
    hist_r = np.histogram(np.array(img)[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(np.array(img)[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(np.array(img)[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    return hist / (np.linalg.norm(hist) + 1e-10)

def cosine_similarity(vec1, vec2):
    vec1 = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2 = vec2 / (np.linalg.norm(vec2) + 1e-10)
    return float(np.dot(vec1, vec2))

# Fish keywords found in ImageNet labels (common forms)
FISH_KEYWORDS = {
    "fish", "shark", "ray", "eel", "puffer", "lionfish", "gar", "goby",
    "seahorse", "goldfish", "tench", "coho", "barracuda", "mackerel",
    "anchovy", "stingray", "electric_ray", "hammerhead", "tiger_shark"
}

def _label_indicates_fish(decoded_preds, threshold=0.1):
    """
    decoded_preds: list of (class_id, label, prob) tuples (from decode_predictions)
    threshold: minimum probability for considering the label meaningful
    """
    for (_, label, prob) in decoded_preds:
        lab = label.lower().replace(" ", "_")
        for keyword in FISH_KEYWORDS:
            if keyword in lab and prob >= threshold:
                return True, float(prob), label
    return False, 0.0, None

def is_fish_by_classification(img_data, prob_threshold=0.1):
    """
    Use MobileNetV2 classifier top predictions to see if image contains a fish.
    Returns True if classification suggests fish.
    """
    try:
        img = Image.open(io.BytesIO(img_data)).convert("RGB").resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = clf_model.predict(x)
        decoded = decode_predictions(preds, top=3)[0]
        is_fish, prob, label = _label_indicates_fish(decoded, threshold=prob_threshold)
        return is_fish
    except Exception as e:
        # If classification fails for any reason, return False (we'll fallback to heuristic)
        print("Classification check failed:", e)
        return False

def is_probably_fish_heuristic(img_data):
    """
    Heuristic method using color (water detection) + contours (elongated body).
    Returns True if heuristic indicates fish-like image.
    """
    try:
        # Decode image with OpenCV
        data = np.frombuffer(img_data, np.uint8)
        img_cv = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_cv is None:
            return False

        h, w = img_cv.shape[:2]
        if h == 0 or w == 0:
            return False

        # ---- 1) Water-color detection (blue/green presence) ----
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([80, 30, 20])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(mask_blue > 0) / (h * w)

        # ---- 2) Find large contours (object detection) ----
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return blue_ratio >= 0.08  # relaxed threshold

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < (0.005 * h * w):  # allow smaller fish
            return False

        x, y, cw, ch = cv2.boundingRect(largest)
        aspect_ratio = float(max(cw, ch)) / (min(cw, ch) + 1e-8)
        elongated = aspect_ratio >= 1.4  # slightly relaxed

        # Combine cues: elongated contour + water or strong water presence
        if elongated and blue_ratio >= 0.04:
            return True
        if blue_ratio >= 0.12:
            return True

        return False

    except Exception as e:
        print("Heuristic fish check failed:", e)
        return False

def is_fish_image(img_data):
    """
    Master decision function:
    1) Try classification-based check (fast)
    2) If inconclusive, try heuristic (shape + water color)
    Returns True if either method suggests fish.
    """
    return is_fish_by_classification(img_data) or is_probably_fish_heuristic(img_data)

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    try:
        img_data = await file.read()

        # --------------- Relaxed fish-only gate ----------------
        if not is_fish_image(img_data):
            return {
                "matched_fish": None,
                "other_similar_fishes": [],
                "error": "Uploaded image does not appear to be a fish. Please upload a fish image."
            }
        # -----------------------------------------------------

        query_emb = get_embedding(img_data)
        query_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        query_hist = get_color_histogram(query_img)

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

                    # Compute histogram for comparison
                    try:
                        resp = requests.get(img_url, timeout=5)
                        fish_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                        fish_hist = get_color_histogram(fish_img)
                    except Exception:
                        fish_hist = None

                    emb_score = cosine_similarity(query_emb, fish_emb)
                    color_score = cosine_similarity(query_hist, fish_hist) if fish_hist is not None else 0.0
                    final_score = 0.7 * emb_score + 0.3 * color_score

                    matches.append({
                        "id": fish["id"],
                        "name": fish["name"],
                        "matched_image_url": img_url,
                        "match_type": sex,
                        "description": fish["description"] if sex == "female" else fish.get("male_description"),
                        "score": float(final_score)
                    })

                except Exception as e:
                    print(f"Skipping fish ID {fish['id']} due to error: {e}")
                    continue

        matches.sort(key=lambda x: x["score"], reverse=True)
        best_match = next((m for m in matches if m["score"] > 0.3), None)
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
