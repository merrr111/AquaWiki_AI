import mysql.connector
import json, io
import numpy as np
import imagehash
import requests
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

# Load the model once
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# --- DB Connection ---
def get_db_connection():
    return mysql.connector.connect(
        host="srv2088.hstgr.io",      
        port=3306,
        user="u915767734_admin",       
        password="Hk76Yg78*",          
        database="u915767734_aquawiki"
    )

#Image Embedding
def get_embedding(img):
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)[0].tolist()

#Image Hash 
def get_image_hash(img):
    return str(imagehash.phash(img))

#Download image from URL
def download_image(url):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

#Main function
def update_all_fishes():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT id, image_url, image_male_url FROM fishes")
    fishes = cursor.fetchall()

    for fish in fishes:
        try:
            print(f"Processing Fish ID {fish['id']}...")
            
            # Female image
            female_img = download_image(fish['image_url'])
            female_hash = get_image_hash(female_img)
            female_emb = get_embedding(female_img)

            # Male image
            male_hash, male_emb = None, None
            if fish.get('image_male_url'):
                male_img = download_image(fish['image_male_url'])
                male_hash = get_image_hash(male_img)
                male_emb = get_embedding(male_img)

            # Update DB
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
                fish['id']
            ))
            conn.commit()
            print(f"✅ Updated Fish ID {fish['id']}")

        except Exception as e:
            print(f"⚠️ Error updating Fish ID {fish['id']}: {e}")

    cursor.close()
    conn.close()
    print("All fishes processed!")

#Run script
if __name__ == "__main__":
    update_all_fishes()
