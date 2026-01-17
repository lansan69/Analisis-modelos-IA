import os
import requests
import pandas as pd
import json
import mimetypes

# --- CONFIGURATION ---
API_URL = "http://localhost:8002/transcribe"
TEST_FILE_PATH = "1.mp3"
PRICE_PER_MINUTE = 0.0061

# 🔴 Language Selection
LANGUAGE = "es"  

results = []

print(f"Processing file with AssemblyAI: {TEST_FILE_PATH} (Language: {LANGUAGE})...")

if os.path.exists(TEST_FILE_PATH):
    try:
        mime_type, _ = mimetypes.guess_type(TEST_FILE_PATH)
        if mime_type is None:
            mime_type = "audio/mpeg"

        with open(TEST_FILE_PATH, "rb") as f:
            print(f"Sending request...", end=" ")
            
            files_payload = {
                "file": (os.path.basename(TEST_FILE_PATH), f, mime_type)
            }
            
            # --- FIX: Conditionally enable features ---
            # AssemblyAI supports Transcription & Diarization in Spanish.
            # But Sentiment & Chapters are usually English-only.
            
            params_payload = {
                "language_code": LANGUAGE,
                "speaker_labels": True, # Diarization WORKS in Spanish
            }
            
            # Only enable these if English, otherwise the API crashes
            if LANGUAGE == "en":
                params_payload["auto_chapters"] = True
                params_payload["sentiment_analysis"] = True
            else:
                print("\n   (⚠️ Disabling Sentiment/Chapters: Not supported in Spanish by AssemblyAI)...", end=" ")

            response = requests.post(
                API_URL, 
                files=files_payload,
                params=params_payload
            )
        
        if response.status_code == 200:
            data = response.json()
            
            # --- 1. SAVE JSON ---
            json_filename = f"{os.path.splitext(TEST_FILE_PATH)[0]}_full_data_assembly.json"
            with open(json_filename, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)
                
            print(f"✅ Done.")
            print(f"   └─ Full details saved to: {json_filename}")

            # --- 2. EXTRACT METRICS ---
            duration_sec = data.get("audio_duration", 0)
            
            # Confidence
            words = data.get("words", [])
            word_count = len(words)
            if words:
                avg_confidence = sum(w.get("confidence", 0) for w in words) / word_count
                confidence_score = avg_confidence * 100
            else:
                confidence_score = 0.0
                
            # Diarization
            utterances = data.get("utterances", [])
            speakers = set()
            for utt in utterances:
                if utt.get("speaker"):
                    speakers.add(utt.get("speaker"))
            
            diarization_status = f"Yes ({len(speakers)} speakers)" if len(speakers) > 0 else "No"

            # Sentiment (Will be 'No' for Spanish)
            sentiment_results = data.get("sentiment_analysis_results", [])
            has_sentiment = "Yes" if len(sentiment_results) > 0 else "No (Not Supported)"
            
            cost = (duration_sec / 60) * PRICE_PER_MINUTE

            results.append({
                "Song": TEST_FILE_PATH,
                "Duration (s)": round(duration_sec, 2),
                "Confidence (%)": round(confidence_score, 2),
                "Diarization": diarization_status,
                "Sentiment Available": has_sentiment,
                "Estimated Cost ($)": round(cost, 5),
                "Words Count": word_count,
                "Full JSON File": json_filename
            })

            # Print Summary
            print("\n--- SUMMARY ---")
            print(f"File: {TEST_FILE_PATH}")
            print(f"Duration: {round(duration_sec, 2)}s")
            print(f"Confidence: {round(confidence_score, 2)}%")
            print(f"Words: {word_count}")
            print(f"Sentiment: {has_sentiment}")
            
        else:
            print(f"❌ Server Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
else:
    print(f"❌ File not found: {TEST_FILE_PATH}")

if results:
    df = pd.DataFrame(results)
    df.to_csv("test_result_assembly.csv", index=False)