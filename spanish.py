import os
import requests
import json
import pandas as pd
import mimetypes
import time

# --- CONFIGURATION ---
INPUT_FOLDER = "./dataset/canciones/spanish_songs"
OUTPUT_CSV = "final_benchmark_results_spanish_songs.csv"
JSON_OUTPUT_DIR = "./processed_jsons/spanish_songs"

# API Endpoints
OPENAI_URL = "http://localhost:8001/transcribe"
DEEPGRAM_URL = "http://localhost:8000/transcribe"
ASSEMBLY_URL = "http://localhost:8002/transcribe"

# Pricing (Estimated per minute)
PRICE_OPENAI = 0.006
PRICE_DEEPGRAM = 0.0043
PRICE_ASSEMBLY = 0.0061

# Ensure output directory exists
if not os.path.exists(JSON_OUTPUT_DIR):
    os.makedirs(JSON_OUTPUT_DIR)

full_results = []

def save_json(data, filename, suffix):
    """Saves raw JSON response."""
    if not data: return
    base_name = os.path.splitext(filename)[0]
    safe_name = f"{base_name}_{suffix}.json"
    save_path = os.path.join(JSON_OUTPUT_DIR, safe_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_openai_data(filepath):
    try:
        with open(filepath, "rb") as f:
            response = requests.post(
                OPENAI_URL,
                files={"file": f},
                # 🔴 FORCE SPANISH: GPT-4o will handle the sentiment
                params={"smart_analysis": True, "language": "es"}
            )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"   [OpenAI] Failed: {e}")
    return None

def get_deepgram_data(filepath):
    try:
        # 🔴 SPANISH OPTIMIZATION: Disable Sentiment (English only)
        params = {
            "model": "nova-2",
            "smart_format": True,
            "diarize": True,
            "language": "es"
        }
        mime_type = mimetypes.guess_type(filepath)[0] or "audio/mpeg"
        with open(filepath, "rb") as f:
            response = requests.post(
                DEEPGRAM_URL, 
                files={"file": (os.path.basename(filepath), f, mime_type)},
                params=params
            )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"   [Deepgram] Failed: {e}")
    return None

def get_assembly_data(filepath):
    try:
        # 🔴 SPANISH OPTIMIZATION: Disable unsupported features to prevent 500 Error
        params = {
            "language_code": "es",
            "speaker_labels": True,
            "sentiment_analysis": False, # Not supported in ES
            "auto_chapters": False       # Not supported in ES
        }
        mime_type = mimetypes.guess_type(filepath)[0] or "audio/mpeg"
        with open(filepath, "rb") as f:
            response = requests.post(
                ASSEMBLY_URL, 
                files={"file": (os.path.basename(filepath), f, mime_type)},
                params=params
            )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"   [Assembly] Failed: {e}")
    return None

# --- MAIN PROCESS ---
print(f"🚀 Starting SPANISH SONGS Benchmark on: {INPUT_FOLDER}")
print(f"📂 JSONs will be saved to: {JSON_OUTPUT_DIR}")

# Check if folder exists
if not os.path.exists(INPUT_FOLDER):
    print(f"❌ Error: Folder '{INPUT_FOLDER}' not found!")
    exit()

files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.mp3', '.wav', '.m4a'))]
files.sort()

for i, filename in enumerate(files):
    filepath = os.path.join(INPUT_FOLDER, filename)
    print(f"\n[{i+1}/{len(files)}] Processing: {filename} ...")
    
    # 1. OPENAI (Spanish)
    print("   👉 1. OpenAI (Forced ES)...", end=" ")
    openai_data = get_openai_data(filepath)
    if not openai_data:
        print("❌ Failed. Skipping.")
        continue
    save_json(openai_data, filename, "openai")
    print("✅ Saved.")

    # 2. DEEPGRAM (Spanish)
    print("   👉 2. Deepgram (Forced ES)...", end=" ")
    deepgram_data = get_deepgram_data(filepath)
    if deepgram_data:
        save_json(deepgram_data, filename, "deepgram")
        print("✅ Saved.")
    else:
        print("❌ Failed.")

    # 3. ASSEMBLYAI (Spanish)
    print("   👉 3. AssemblyAI (Forced ES)...", end=" ")
    assembly_data = get_assembly_data(filepath)
    if assembly_data:
        save_json(assembly_data, filename, "assembly")
        print("✅ Saved.")
    else:
        print("❌ Failed.")
    
    # 4. BUILD CSV ROW
    duration = openai_data.get("audio_duration", 0)
    
    row = {
        "Filename": filename,
        "Language": "es",  # Hardcoded Spanish
        "Duration (s)": round(duration, 2),
        
        # OpenAI
        "OAI_Confidence": round(openai_data.get("confidence", 0)*100, 2),
        "OAI_Sentiment": "Yes" if openai_data.get("sentiment_analysis_results") else "No",
        "OAI_Cost": round((duration/60)*PRICE_OPENAI, 5),
        
        # Deepgram
        "DG_Confidence": 0,
        "DG_Sentiment": "No", # Always No for Spanish
        "DG_Cost": round((duration/60)*PRICE_DEEPGRAM, 5),
        
        # Assembly
        "AAI_Confidence": 0,
        "AAI_Sentiment": "No", # Always No for Spanish
        "AAI_Cost": round((duration/60)*PRICE_ASSEMBLY, 5)
    }
    
    # Extract Deepgram Metrics
    if deepgram_data:
        try:
            res = deepgram_data["results"]["channels"][0]["alternatives"][0]
            row["DG_Confidence"] = round(res["confidence"] * 100, 2)
        except: pass

    # Extract Assembly Metrics
    if assembly_data:
        words = assembly_data.get("words", [])
        if words:
            avg = sum(w["confidence"] for w in words) / len(words)
            row["AAI_Confidence"] = round(avg * 100, 2)

    full_results.append(row)
    
    # Save continuously
    pd.DataFrame(full_results).to_csv(OUTPUT_CSV, index=False)

print(f"\n🎉 Batch Complete! \n   📄 Summary: {OUTPUT_CSV} \n   📂 Details: {JSON_OUTPUT_DIR}/")