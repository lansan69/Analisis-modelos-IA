from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import json
import tempfile
from dotenv import load_dotenv
import assemblyai as aai

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="AssemblyAI Transcription API",
    description="API para transcribir audios usando AssemblyAI",
    version="1.0.0"
)

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not ASSEMBLYAI_API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY no está configurada en las variables de entorno")

# Configurar la API key de AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "AssemblyAI Transcription API",
        "version": "1.0.0",
        "endpoints": {
            "/transcribe": "POST - Subir audio para transcribir",
            "/health": "GET - Verificar estado de la API"
        }
    }


@app.get("/health")
async def health_check():
    """Verificar el estado de la API"""
    return {"status": "healthy", "service": "assemblyai-transcription"}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Archivo de audio a transcribir (mp3, wav, etc.)"),
    language_code: str = "es",
    speaker_labels: bool = False,
    auto_chapters: bool = False,
    sentiment_analysis: bool = False
):
    """
    Transcribir un archivo de audio usando AssemblyAI
    
    Parameters:
    - file: Archivo de audio
    - language_code: Código del idioma (default: es)
    - speaker_labels: Identificar diferentes hablantes (default: False)
    - auto_chapters: Generar capítulos automáticamente (default: False)
    - sentiment_analysis: Análisis de sentimiento (default: False)
    
    Returns:
    - JSON con la transcripción completa de AssemblyAI
    """
    
    # Validar que se subió un archivo
    if not file:
        raise HTTPException(status_code=400, detail="No se proporcionó ningún archivo")
    
    # Validar tipo de archivo
    allowed_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de archivo no soportado. Use uno de: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Leer el contenido del archivo
        audio_data = await file.read()
        
        # Guardar temporalmente el archivo
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Configurar opciones de transcripción
            config = aai.TranscriptionConfig(
                language_code=language_code,
                speaker_labels=speaker_labels,
                auto_chapters=auto_chapters,
                sentiment_analysis=sentiment_analysis
            )
            
            # Crear transcriber
            transcriber = aai.Transcriber()
            
            # Transcribir el audio
            transcript = transcriber.transcribe(temp_file_path, config=config)
            
            # Construir respuesta
            response_dict = {
                "id": transcript.id,
                "status": transcript.status,
                "text": transcript.text,
                "words": [
                    {
                        "text": word.text,
                        "start": word.start,
                        "end": word.end,
                        "confidence": word.confidence,
                        "speaker": getattr(word, 'speaker', None)
                    }
                    for word in (transcript.words or [])
                ] if transcript.words else [],
                "utterances": [
                    {
                        "text": utt.text,
                        "start": utt.start,
                        "end": utt.end,
                        "confidence": utt.confidence,
                        "speaker": utt.speaker
                    }
                    for utt in (transcript.utterances or [])
                ] if speaker_labels and transcript.utterances else [],
                "chapters": [
                    {
                        "summary": chapter.summary,
                        "headline": chapter.headline,
                        "gist": chapter.gist,
                        "start": chapter.start,
                        "end": chapter.end
                    }
                    for chapter in (transcript.chapters or [])
                ] if auto_chapters and transcript.chapters else [],
                "sentiment_analysis_results": [
                    {
                        "text": sent.text,
                        "start": sent.start,
                        "end": sent.end,
                        "sentiment": sent.sentiment,
                        "confidence": sent.confidence
                    }
                    for sent in (transcript.sentiment_analysis or [])
                ] if sentiment_analysis and transcript.sentiment_analysis else [],
                "audio_duration": transcript.audio_duration,
                "language_code": language_code
            }
            
            # Retornar la transcripción
            return JSONResponse(content=response_dict)
        
        finally:
            # Eliminar archivo temporal
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al transcribir el audio: {str(e)}"
        )


@app.post("/transcribe/text-only")
async def transcribe_audio_text_only(
    file: UploadFile = File(..., description="Archivo de audio a transcribir"),
    language_code: str = "es"
):
    """
    Transcribir un archivo de audio y retornar solo el texto transcrito
    
    Parameters:
    - file: Archivo de audio
    - language_code: Código del idioma (default: es)
    
    Returns:
    - JSON con solo el texto transcrito
    """
    
    if not file:
        raise HTTPException(status_code=400, detail="No se proporcionó ningún archivo")
    
    try:
        # Leer el contenido del archivo
        audio_data = await file.read()
        
        # Guardar temporalmente el archivo
        file_extension = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Configurar opciones de transcripción
            config = aai.TranscriptionConfig(language_code=language_code)
            
            # Crear transcriber
            transcriber = aai.Transcriber()
            
            # Transcribir el audio
            transcript = transcriber.transcribe(temp_file_path, config=config)
            
            return {
                "filename": file.filename,
                "transcript": transcript.text,
                "language_code": language_code,
                "audio_duration": transcript.audio_duration
            }
        
        finally:
            # Eliminar archivo temporal
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al transcribir el audio: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
