from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import json
import tempfile
import math
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

app = FastAPI(
    title="OpenAI Enhanced Transcription API",
    description="Whisper (Transcription) + GPT-4o (Intelligence)",
    version="2.1.0"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env")

# Initialize Client
client = OpenAI(api_key=OPENAI_API_KEY)

def analyze_with_gpt(transcript_text: str):
    """
    Uses GPT-4o-mini to hallucinate the 'Analysis' features 
    that Whisper doesn't have (Sentiment, Chapters, etc).
    """
    system_prompt = """
    You are an audio analysis engine. Return a JSON object with:
    1. 'sentiment_analysis_results': A list of segments with sentiment (POSITIVE, NEGATIVE, NEUTRAL) and confidence (0.0-1.0).
    2. 'chapters': A list of chapters with 'headline', 'summary', 'start', 'end'.
    3. 'summary': A brief summary of the entire text.
    
    Structure the JSON exactly like AssemblyAI's response format.
    If the text is too short, provide a single sentiment/chapter.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this transcript: {transcript_text[:15000]}"} 
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"GPT Analysis failed: {e}")
        return {"sentiment_analysis_results": [], "chapters": []}

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file"),
    language: str = None, # 🔴 CAMBIO: Default es None (Auto-detectar)
    smart_analysis: bool = True 
):
    """
    Hybrid Endpoint: Whisper (Audio) + GPT-4o (Analysis)
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    # 1. Save Temp File
    file_extension = os.path.splitext(file.filename or "")[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # 2. Transcribe with Whisper (Get WORD Timestamps)
        # Construimos los argumentos dinámicamente para mayor seguridad
        transcription_args = {
            "model": "whisper-1",
            "file": open(temp_path, "rb"),
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"]
        }
        
        # 🔴 LÓGICA DE DETECCIÓN:
        # Solo pasamos el idioma si el cliente lo pidió explícitamente.
        # Si language es None, Whisper entrará en modo "Auto-Detect".
        if language:
            transcription_args["language"] = language

        transcript_response = client.audio.transcriptions.create(**transcription_args)
        
        # 3. Calculate Confidence & Duration
        segments = getattr(transcript_response, 'segments', [])
        # Safe logprob calculation
        avg_logprob = -1.0
        if segments:
            valid_logprobs = [s['avg_logprob'] for s in segments if 'avg_logprob' in s]
            if valid_logprobs:
                avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
        
        confidence_score = math.exp(avg_logprob)
        
        # 4. Optional: Intelligence Layer (GPT-4o)
        analysis_data = {}
        if smart_analysis:
            analysis_data = analyze_with_gpt(transcript_response.text)

        # 5. Build Response
        response_dict = {
            "text": transcript_response.text,
            "audio_duration": transcript_response.duration,
            "confidence": confidence_score,  # Normalized 0.0 - 1.0
            
            # Map Whisper 'words' to your list
            "words": [
                {
                    "text": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": 0.99 
                } 
                for w in getattr(transcript_response, 'words', [])
            ],
            
            # GPT-4o Generated Data
            "sentiment_analysis_results": analysis_data.get("sentiment_analysis_results", []),
            "chapters": analysis_data.get("chapters", []),
            
            # Metadata
            "model_used": "whisper-1 + gpt-4o-mini",
            # Agregamos esto para debuggear si quieres ver qué idioma detectó
            "detected_language": getattr(transcript_response, "language", "auto") 
        }

        return JSONResponse(content=response_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)