import os
import requests
import json
import pandas as pd
import mimetypes
import time

# --- CONFIGURATION ---
INPUT_FOLDER = "./dataset/audios"
OUTPUT_CSV = "final_benchmark_results_audios.csv"
JSON_OUTPUT_DIR = "./processed_jsons/audios"  # Folder to save all JSONs

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
    """Saves the raw JSON response to a file with special character support."""
    if not data: return
    
    # Create name like: "1_openai.json"
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
                params={"smart_analysis": True} # Auto-detects language
            )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"   [OpenAI] Failed: {e}")
    return None

def get_deepgram_data(filepath, detected_lang):
    try:
        # Config based on language
        params = {
            "model": "nova-2",
            "smart_format": True,
            "diarize": True,
            "language": detected_lang
        }
        # Only enable sentiment if English (Limitation)
        if detected_lang == "en":
            params["sentiment"] = True
            
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

def get_assembly_data(filepath, detected_lang):
    try:
        params = {
            "language_code": detected_lang,
            "speaker_labels": True
        }
        # Only enable Intelligence if English (Prevents 500 Error)
        if detected_lang == "en":
            params["sentiment_analysis"] = True
            params["auto_chapters"] = True
            
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
print(f"🚀 Starting Batch Process on: {INPUT_FOLDER}")
print(f"📂 JSONs will be saved to: {JSON_OUTPUT_DIR}")

files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.mp3', '.wav', '.m4a'))]
files.sort()

for i, filename in enumerate(files):
    filepath = os.path.join(INPUT_FOLDER, filename)
    print(f"\n[{i+1}/{len(files)}] Processing: {filename} ...")
    
    # 1. OPENAI (Router & Analysis)
    print("   👉 1. OpenAI...", end=" ")
    openai_data = get_openai_data(filepath)
    
    if not openai_data:
        print("❌ Failed. Skipping.")
        continue
    
    save_json(openai_data, filename, "openai")
    print("✅ Saved.")

    # Detect Language
    text_sample = openai_data.get("text", "")[:50].lower()
    # Simple heuristic: if " the " is present, assume English, else Spanish
    detected_lang = "en" if " the " in text_sample or " and " in text_sample else "es"
    print(f"      Detected Language: {detected_lang.upper()}")

    # 2. DEEPGRAM
    print("   👉 2. Deepgram...", end=" ")
    deepgram_data = get_deepgram_data(filepath, detected_lang)
    if deepgram_data:
        save_json(deepgram_data, filename, "deepgram")
        print("✅ Saved.")
    else:
        print("❌ Failed.")

    # 3. ASSEMBLYAI
    print("   👉 3. AssemblyAI...", end=" ")
    assembly_data = get_assembly_data(filepath, detected_lang)
    if assembly_data:
        save_json(assembly_data, filename, "assembly")
        print("✅ Saved.")
    else:
        print("❌ Failed.")
    
    # 4. BUILD CSV ROW
    duration = openai_data.get("audio_duration", 0)
    
    row = {
        "Filename": filename,
        "Language": detected_lang,
        "Duration (s)": round(duration, 2),
        
        # OpenAI
        "OAI_Confidence": round(openai_data.get("confidence", 0)*100, 2),
        "OAI_Sentiment": "Yes" if openai_data.get("sentiment_analysis_results") else "No",
        "OAI_Cost": round((duration/60)*PRICE_OPENAI, 5),
        
        # Deepgram
        "DG_Confidence": 0,
        "DG_Sentiment": "No",
        "DG_Cost": round((duration/60)*PRICE_DEEPGRAM, 5),
        
        # Assembly
        "AAI_Confidence": 0,
        "AAI_Sentiment": "No",
        "AAI_Cost": round((duration/60)*PRICE_ASSEMBLY, 5)
    }
    
    # Extract Deepgram Metrics
    if deepgram_data:
        try:
            res = deepgram_data["results"]["channels"][0]["alternatives"][0]
            row["DG_Confidence"] = round(res["confidence"] * 100, 2)
            if "sentiments" in deepgram_data["results"]: row["DG_Sentiment"] = "Yes"
        except: pass

    # Extract Assembly Metrics
    if assembly_data:
        words = assembly_data.get("words", [])
        if words:
            avg = sum(w["confidence"] for w in words) / len(words)
            row["AAI_Confidence"] = round(avg * 100, 2)
        if assembly_data.get("sentiment_analysis_results"):
            row["AAI_Sentiment"] = "Yes"

    full_results.append(row)
    
    # Save continuously
    pd.DataFrame(full_results).to_csv(OUTPUT_CSV, index=False)

print(f"\n🎉 Batch Complete! \n   📄 Summary: {OUTPUT_CSV} \n   📂 Details: {JSON_OUTPUT_DIR}/")