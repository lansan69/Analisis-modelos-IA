import os
import requests
import pandas as pd
import json

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/transcribe"  # Deepgram runs on port 8000
TEST_FILE_PATH = "1.mp3"                      # Your test file
PRICE_PER_MINUTE = 0.0043                     # Deepgram Nova-2/3 Pricing (approx)

results = []

print(f"Processing single file with Deepgram: {TEST_FILE_PATH} ...")

if os.path.exists(TEST_FILE_PATH):
    try:
        with open(TEST_FILE_PATH, "rb") as f:
            print(f"Sending request...", end=" ")
            
            # Deepgram API expects 'file' parameter
            response = requests.post(
                API_URL, 
                files={"file": f},
                params={
                    "language": "es",       # English for your songs
                    "diarize": True,        # Enable Speaker Detection
                    "sentiment": True       # Enable Sentiment Analysis
                }
            )
        
        if response.status_code == 200:
            data = response.json()
            
            # --- 1. SAVE FULL DETAILED DATA (JSON) ---
            json_filename = f"{os.path.splitext(TEST_FILE_PATH)[0]}_full_data_deepgram.json"
            with open(json_filename, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)
                
            print(f"✅ Done.")
            print(f"   └─ Full details saved to: {json_filename}")

            # --- 2. EXTRACT METRICS (Deepgram Normalization Logic) ---
            # Deepgram structure is: results -> channels[0] -> alternatives[0]
            try:
                result_data = data.get("results", {}).get("channels", [])[0].get("alternatives", [])[0]
                metadata = data.get("metadata", {})
                
                # A. Duration
                duration_sec = metadata.get("duration", 0)
                
                # B. Confidence (0.0 to 1.0) -> Convert to %
                confidence_score = result_data.get("confidence", 0) * 100
                
                # C. Words Count
                words_list = result_data.get("words", [])
                word_count = len(words_list)
                
                # D. Diarization Check (Count unique speakers)
                # Look into 'words' or 'paragraphs' for speaker tags
                speakers = set()
                if "paragraphs" in result_data and "paragraphs" in result_data["paragraphs"]:
                    # Deepgram usually structures speaker changes in paragraphs
                    for p in result_data["paragraphs"].get("paragraphs", []):
                        if "speaker" in p:
                            speakers.add(p["speaker"])
                elif words_list:
                    # Fallback: check individual words
                    for w in words_list:
                        if "speaker" in w:
                            speakers.add(w["speaker"])
                            
                speaker_count = len(speakers)
                diarization_status = f"Yes ({speaker_count} speakers)" if speaker_count > 0 else "No"

                # E. Sentiment Check
                # Deepgram usually puts average sentiment in the result or per utterance
                sentiment_segments = data.get("results", {}).get("sentiments", {}).get("segments", [])
                # Sometimes it's attached to utterances if requested differently, 
                # but 'sentiment=True' usually adds a top-level analysis or segment analysis.
                # We'll check if any sentiment data exists in the raw structure.
                has_sentiment = "No"
                if "sentiments" in data.get("results", {}) or "sentiment" in result_data:
                    has_sentiment = "Yes"
                
                # F. Cost
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
                print(f"Speakers Found: {speaker_count}")
                print(f"Sentiment Data: {has_sentiment}")
                
            except IndexError:
                print("❌ Error: Deepgram response format was unexpected (empty channels/alternatives).")
            
        else:
            print(f"❌ Server Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
else:
    print(f"❌ File not found: {TEST_FILE_PATH}")

# Export CSV
if results:
    df = pd.DataFrame(results)
    df.to_csv("test_result_deepgram.csv", index=False)
    print("   └─ Summary saved to: test_result_deepgram.csv")