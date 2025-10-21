from PIL import Image
import imagehash
import mysql.connector
import os

# ✅ Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",        # adjust if needed
    password="",        # adjust if needed
    database="fishencyclopedia"
)
cursor = db.cursor(dictionary=True)

def calculate_hash(image_path):
    """Calculate perceptual hash of an image (normalized)"""
    try:
        image = Image.open(image_path).convert("RGB").resize((256, 256))
        return str(imagehash.phash(image))
    except Exception as e:
        print(f"❌ Error hashing {image_path}: {e}")
        return None

# ✅ Fetch all fishes
cursor.execute("SELECT id, image_url, image_male_url FROM fishes")
fishes = cursor.fetchall()

for fish in fishes:
    fish_id = fish["id"]
    female_path = fish["image_url"]
    male_path = fish["image_male_url"]
    updated = False

    # ✅ Female image → image_hash
    if female_path and os.path.isfile(female_path):
        hash_val = calculate_hash(female_path)
        if hash_val:
            cursor.execute(
                "UPDATE fishes SET image_hash=%s WHERE id=%s",
                (hash_val, fish_id)
            )
            updated = True
            print(f"✅ Updated fish ID {fish_id} with female hash {hash_val}")

    # ✅ Male image → image_male_hash
    if male_path and os.path.isfile(male_path):
        hash_val = calculate_hash(male_path)
        if hash_val:
            cursor.execute(
                "UPDATE fishes SET image_male_hash=%s WHERE id=%s",
                (hash_val, fish_id)
            )
            updated = True
            print(f"✅ Updated fish ID {fish_id} with male hash {hash_val}")

    if updated:
        db.commit()
    else:
        print(f"⚠️ No valid images found for fish ID {fish_id}")

cursor.close()
db.close()
