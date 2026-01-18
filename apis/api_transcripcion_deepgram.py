from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import json
import tempfile
from dotenv import load_dotenv
from deepgram import DeepgramClient

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Deepgram Transcription API (Enhanced)",
    description="API para transcribir audios usando Deepgram con Diarización y Sentimientos",
    version="1.1.0"
)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY no está configurada en las variables de entorno")


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Deepgram Transcription API",
        "version": "1.1.0",
        "endpoints": {
            "/transcribe": "POST - Subir audio (Incluye Diarización y Sentimientos)",
            "/health": "GET - Verificar estado de la API"
        }
    }


@app.get("/health")
async def health_check():
    """Verificar el estado de la API"""
    return {"status": "healthy", "service": "deepgram-transcription"}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Archivo de audio a transcribir (mp3, wav, etc.)"),
    model: str = "nova-3",
    language: str = "es",
    smart_format: bool = True,
    diarize: bool = True,       # NEW: Default to True for your evaluation
    sentiment: bool = True      # NEW: Default to True for your evaluation
):
    """
    Transcribir un archivo de audio usando Deepgram.
    
    Features activados por defecto para evaluación:
    - Diarization (Identificación de hablantes)
    - Sentiment Analysis (Detección de emociones)
    - Utterances (Segmentación por tiempos)
    """
    
    # Validar que se subió un archivo
    if not file:
        raise HTTPException(status_code=400, detail="No se proporcionó ningún archivo")
    
    # Validar tipo de archivo
    allowed_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm']
    # Usamos 'or ""' para asegurar que si es None, pase un string vacío
    file_extension = os.path.splitext(file.filename or "")[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de archivo no soportado. Use uno de: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Crear cliente de Deepgram
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        
        # Leer el contenido del archivo
        audio_data = await file.read()
        
        # --- CRITICAL UPDATE FOR YOUR CHECKLIST ---
        # Added: diarize, utterances, sentiment, punctuate
        response = deepgram.listen.v1.media.transcribe_file(
            request=audio_data,
            model=model,
            language=language,
            smart_format=smart_format,
            diarize=diarize,         # Identifies distinct speakers
            sentiment=sentiment,     # Analyzes Pos/Neg/Neutral sentiment
            utterances=True,         # Required for structured segments/timestamps
            punctuate=True           # Adds proper punctuation
        )
        
        # Convertir respuesta a JSON
        response_json = response.json()
        response_dict = json.loads(response_json)
        
        # Retornar la transcripción
        return JSONResponse(content=response_dict)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al transcribir el audio: {str(e)}"
        )


@app.post("/transcribe/text-only")
async def transcribe_audio_text_only(
    file: UploadFile = File(..., description="Archivo de audio a transcribir"),
    model: str = "nova-3",
    language: str = "es"
):
    """
    Transcribir un archivo de audio y retornar solo el texto transcrito
    """
    if not file:
        raise HTTPException(status_code=400, detail="No se proporcionó ningún archivo")
    
    try:
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        audio_data = await file.read()
        
        # Transcripción simple para text-only
        response = deepgram.listen.v1.media.transcribe_file(
            request=audio_data,
            model=model,
            language=language,
            smart_format=True
        )
        
        response_json = response.json()
        response_dict = json.loads(response_json)
        
        # Logic to extract clean text safely
        transcript = ""
        if "results" in response_dict and "channels" in response_dict["results"]:
            channels = response_dict["results"]["channels"]
            if channels and "alternatives" in channels[0]:
                alternatives = channels[0]["alternatives"]
                if alternatives:
                    transcript = alternatives[0].get("transcript", "")
        
        return {
            "filename": file.filename,
            "transcript": transcript,
            "language": language,
            "model": model
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al transcribir el audio: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)