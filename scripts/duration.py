import os
import datetime
from mutagen import File

# --- CONFIGURATION ---
ROOT_DIR = "./dataset"  # Path to your main folder

def calculate_total_duration():
    total_seconds = 0
    file_count = 0
    
    print(f"Scanning '{ROOT_DIR}' for audio files...")
    
    # Walk recursively through all subfolders
    for root, dirs, files in os.walk(ROOT_DIR):
        for filename in files:
            if filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac')):
                filepath = os.path.join(root, filename)
                
                try:
                    # mutagen.File attempts to detect format automatically
                    audio = File(filepath)
                    
                    if audio is not None and audio.info is not None:
                        duration = audio.info.length
                        total_seconds += duration
                        file_count += 1
                        # Optional: Print individual file duration
                        # print(f"Found: {filename} ({round(duration, 2)}s)")
                        
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {e}")

    # Convert total seconds to readable format
    total_duration = datetime.timedelta(seconds=int(total_seconds))
    
    print("-" * 40)
    print(f"✅ Scanning Complete")
    print(f"📂 Total Files Processed: {file_count}")
    print(f"⏱️  Total Duration: {total_duration}")
    print(f"   (Hours:Minutes:Seconds)")
    print("-" * 40)

if __name__ == "__main__":
    if os.path.exists(ROOT_DIR):
        calculate_total_duration()
    else:
        print(f"❌ Error: Folder '{ROOT_DIR}' not found.")
