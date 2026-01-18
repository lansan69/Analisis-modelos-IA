import os
import json
import glob
import pandas as pd
import difflib
import statistics

# --- CONFIGURACIÓN ---
# Carpeta donde están los JSONs (puede ser ./processed_jsons/audios o la que uses)
JSON_FOLDERS = [
    "./processed_jsons/audios", 
]
OUTPUT_CSV = "detailed_analysis_report.csv"

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def normalize_data(data, provider):
    """Estandariza la estructura de los 3 proveedores."""
    if not data: return None
    
    norm = {
        "text": "",
        "start": 0.0,
        "end": 0.0,
        "duration": 0.0,
        "confidence": 0.0,
        "speakers": 0,
        "word_count": 0
    }

    words = []
    try:
        # OPENAI
        if provider == "OpenAI":
            norm["text"] = data.get("text", "")
            words = data.get("words", [])
            if words:
                norm["start"] = words[0].get("start", 0)
                norm["end"] = words[-1].get("end", 0)
                # OpenAI a veces usa logprob, normalizamos si es negativo
                confs = [w.get("confidence", 1.0) for w in words]
                # Fix para tu caso donde confidence era ~0.36 (logprob)
                # Asumimos que si el promedio es < 0.8 y > 0, podría ser un error de formato,
                # pero usaremos el valor crudo o 0 si falla.
                norm["confidence"] = statistics.mean(confs) * 100 if confs else 0
            norm["speakers"] = 1 # Whisper v1 no diariza nativo en API simple

        # DEEPGRAM
        elif provider == "Deepgram":
            alt = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0]
            norm["text"] = alt.get("transcript", "")
            words = alt.get("words", [])
            if words:
                norm["start"] = words[0].get("start", 0)
                norm["end"] = words[-1].get("end", 0)
                speakers = set(w.get("speaker") for w in words if "speaker" in w)
                norm["speakers"] = len(speakers) if speakers else 1
                confs = [w.get("confidence", 0) for w in words]
                norm["confidence"] = statistics.mean(confs) * 100

        # ASSEMBLYAI
        elif provider == "AssemblyAI":
            norm["text"] = data.get("text", "")
            words = data.get("words", [])
            if words:
                # Assembly da milisegundos -> convertir a segundos
                norm["start"] = words[0].get("start", 0) / 1000.0
                norm["end"] = words[-1].get("end", 0) / 1000.0
                speakers = set(w.get("speaker") for w in words if w.get("speaker") is not None)
                norm["speakers"] = len(speakers) if speakers else 1
                confs = [w.get("confidence", 0) for w in words]
                norm["confidence"] = statistics.mean(confs) * 100

        norm["word_count"] = len(words)
        norm["duration"] = norm["end"] - norm["start"]
        
    except Exception as e:
        print(f"Error parseando {provider}: {e}")
        return None

    return norm

def get_similarity(text1, text2):
    if not text1 or not text2: return 0.0
    return difflib.SequenceMatcher(None, text1, text2).ratio()

# --- PROCESO PRINCIPAL ---
all_rows = []

# Buscar todos los archivos JSON en las carpetas
files = []
for folder in JSON_FOLDERS:
    if os.path.exists(folder):
        files.extend(glob.glob(os.path.join(folder, "*_*.json")))

# Extraer IDs únicos (ej: "1", "10", "song_1")
# Asume formato: {id}_{provider}.json
unique_ids = set()
for f in files:
    filename = os.path.basename(f)
    # Estrategia simple: tomar lo que está antes del primer guion bajo
    # Ajusta esto si tus IDs tienen guiones bajos (ej: english_song_1)
    if "_openai" in filename: unique_ids.add(filename.split("_openai")[0])
    elif "_deepgram" in filename: unique_ids.add(filename.split("_deepgram")[0])
    elif "_assembly" in filename: unique_ids.add(filename.split("_assembly")[0])

print(f"🚀 Iniciando análisis de {len(unique_ids)} audios únicos...")

for audio_id in sorted(unique_ids):
    # Intentar encontrar los archivos en cualquiera de las carpetas
    found_files = {}
    for prov in ["openai", "deepgram", "assembly"]:
        for folder in JSON_FOLDERS:
            path = os.path.join(folder, f"{audio_id}_{prov}.json")
            if os.path.exists(path):
                found_files[prov] = path
                break
    
    # Cargar y Normalizar
    d_oai = normalize_data(load_json(found_files.get("openai")), "OpenAI")
    d_dg  = normalize_data(load_json(found_files.get("deepgram")), "Deepgram")
    d_aai = normalize_data(load_json(found_files.get("assembly")), "AssemblyAI")

    row = {"Audio_ID": audio_id}

    # Métricas Individuales
    row["OAI_Words"] = d_oai["word_count"] if d_oai else 0
    row["DG_Words"]  = d_dg["word_count"] if d_dg else 0
    row["AAI_Words"] = d_aai["word_count"] if d_aai else 0

    row["OAI_Conf"] = round(d_oai["confidence"], 2) if d_oai else 0
    row["DG_Conf"]  = round(d_dg["confidence"], 2) if d_dg else 0
    row["AAI_Conf"] = round(d_aai["confidence"], 2) if d_aai else 0

    # Comparativa de Texto (Similitud %)
    # Cuánto se parecen Deepgram y Assembly a OpenAI (asumiendo OAI como "base" por inteligencia)
    row["Sim_OAI_vs_DG"] = round(get_similarity(d_oai["text"], d_dg["text"]) * 100, 2) if d_oai and d_dg else 0
    row["Sim_OAI_vs_AAI"] = round(get_similarity(d_oai["text"], d_aai["text"]) * 100, 2) if d_oai and d_aai else 0
    row["Sim_DG_vs_AAI"]  = round(get_similarity(d_dg["text"], d_aai["text"]) * 100, 2) if d_dg and d_aai else 0

    # Diferencias de Tiempos (Diarización / Latencia de inicio)
    # Diferencia absoluta en el tiempo de inicio (Start Time Delta)
    if d_oai and d_dg:
        row["Start_Diff_OAI_DG"] = round(abs(d_oai["start"] - d_dg["start"]), 3)
    else: row["Start_Diff_OAI_DG"] = None

    if d_dg and d_aai:
        row["Start_Diff_DG_AAI"] = round(abs(d_dg["start"] - d_aai["start"]), 3)
    else: row["Start_Diff_DG_AAI"] = None

    # Speakers
    row["DG_Speakers"] = d_dg["speakers"] if d_dg else 0
    row["AAI_Speakers"] = d_aai["speakers"] if d_aai else 0
    
    all_rows.append(row)

# Generar CSV Final
df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ Análisis Completo.")
print(f"📄 Reporte guardado en: {OUTPUT_CSV}")
print("\n--- RESUMEN RÁPIDO ---")
print(f"Similitud Promedio (DG vs AAI): {df['Sim_DG_vs_AAI'].mean():.2f}%")
print(f"Diferencia Promedio Inicio (DG vs AAI): {df['Start_Diff_DG_AAI'].mean():.3f} segundos")