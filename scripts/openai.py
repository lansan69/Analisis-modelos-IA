import os
import requests
import pandas as pd
import json  # Added json library

# --- CONFIGURATION ---
API_URL = "http://localhost:8001/transcribe"
TEST_FILE_PATH = "1.mp3"  
PRICE_PER_MINUTE = 0.006

results = []

print(f"Processing single file: {TEST_FILE_PATH} ...")

if os.path.exists(TEST_FILE_PATH):
    try:
        with open(TEST_FILE_PATH, "rb") as f:
            print(f"Sending request...", end=" ")
            
            # Send request
            response = requests.post(
                API_URL, 
                files={"file": f},
                params={
                    "language": "es", 
                    "smart_analysis": True 
                }
            )
        
        if response.status_code == 200:
            data = response.json()
            
            # --- 1. SAVE FULL DETAILED DATA (JSON) ---
            # This saves the transcription, words, sentiment, chapters - EVERYTHING.
            json_filename = f"{os.path.splitext(TEST_FILE_PATH)[0]}_full_data.json"
            
            with open(json_filename, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)
                
            print(f"✅ Done.")
            print(f"   └─ Full details saved to: {json_filename}")

            # --- 2. PREPARE SUMMARY DATA (CSV) ---
            duration_sec = data.get("audio_duration", 0)
            confidence_score = data.get("confidence", 0) * 100
            cost = (duration_sec / 60) * PRICE_PER_MINUTE
            
            sentiment_found = "Yes" if data.get("sentiment_analysis_results") else "No"
            chapters_found = "Yes" if data.get("chapters") else "No"
            
            results.append({
                "Song": TEST_FILE_PATH,
                "Duration (s)": round(duration_sec, 2),
                "Confidence (%)": round(confidence_score, 2),
                "Sentiment Available": sentiment_found,
                "Chapters Available": chapters_found,
                "Estimated Cost ($)": round(cost, 5),
                "Words Count": len(data.get("words", [])),
                "Full JSON File": json_filename  # Reference to the full file
            })
            
            # Print Result
            print("\n--- SUMMARY ---")
            print(f"File: {TEST_FILE_PATH}")
            print(f"Duration: {round(duration_sec, 2)}s")
            print(f"Confidence: {round(confidence_score, 2)}%")
            print(f"Sentiment: {sentiment_found}")
            
        else:
            print(f"❌ Server Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
else:
    print(f"❌ File not found: {TEST_FILE_PATH}")

# Export CSV
if results:
    df = pd.DataFrame(results)
    df.to_csv("test_result_openai_fixed.csv", index=False)
    print("   └─ Summary saved to: test_result_openai_fixed.csv")