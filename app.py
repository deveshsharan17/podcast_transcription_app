import os
import uuid
import json
from datetime import timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from typing import List
from pathlib import Path
from pydub import AudioSegment
import torch
import whisper
import yt_dlp
from pyannote.audio import Pipeline
from openai import OpenAI
from transformers import pipeline as hf_pipeline
import assemblyai as aai
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ------------- Config ---------------- #
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

WHISPER_MODEL_NAME = "base.en"
CHUNK_MINUTES = 1
NUM_THREADS = 4

BART_MODEL = "facebook/bart-large-cnn"
T5_MODEL = "google/flan-t5-base"

OPENAI_DEFAULT_MODEL = "gpt-4o"
OPENAI_ALT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo-preview",
    "gpt-3.5-turbo",
]

CHAT_MAX_CHARS = 15_000  # context length sent to the chatbot

# Get API keys from environment variables
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

if not ASSEMBLYAI_API_KEY:
    print("Warning: ASSEMBLYAI_API_KEY not found in environment variables")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables")
if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in environment variables")

# Configure AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY

# ----------- Flask app ------------- #
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# ----------- Helper functions (adapted from your script) -------------- #

def get_video_info(url: str):
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    duration = info.get("duration", 0)
    title = info.get("title", "Podcast")
    return duration, title

def download_audio(url: str) -> Path:
    out = OUTPUT_DIR / f"{uuid.uuid4()}.%(ext)s"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return next(OUTPUT_DIR.glob(out.name.replace("%(ext)s", "wav")))

def clip_audio(src_wav: Path, start_s: int, end_s: int) -> Path:
    audio = AudioSegment.from_file(src_wav)
    clipped = audio[start_s * 1000 : end_s * 1000]
    dst = src_wav.with_name(src_wav.stem + f"_{start_s}s_{end_s}s.wav")
    clipped.export(dst, format="wav")
    return dst

def chunk_audio(wav_file: Path, minutes: int = CHUNK_MINUTES):
    audio = AudioSegment.from_file(wav_file)
    ms = minutes * 60_000
    for i in range(0, len(audio), ms):
        chunk = audio[i : i + ms]
        chunk_path = wav_file.with_suffix(f".part{i//ms}.wav")
        chunk.export(chunk_path, format="wav")
        yield i / 1000.0, chunk_path

def load_whisper_model():
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    try:
        model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
    except Exception:
        device = "cpu"
        model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
    return model, device

def transcribe_chunks(chunks, model, device):
    import concurrent.futures
    lock = threading.Lock()

    def _one(offset_s, path):
        with lock:
            res = model.transcribe(str(path), word_timestamps=True, verbose=False, fp16=device == "cuda")
        for seg in res["segments"]:
            seg["start"] += offset_s
            seg["end"] += offset_s
            for w in seg["words"]:
                w["start"] += offset_s
                w["end"] += offset_s
        return res["segments"]

    segments = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(_one, off, p) for off, p in chunks]
        for f in futures:
            segments.extend(f.result())
    segments.sort(key=lambda s: s["start"])
    return segments

def load_diarization_pipeline(token):
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
        cache_dir=str(Path.home() / ".cache/pyannote"),
    )

def assign_speakers(segments, diarization):
    from itertools import groupby
    time_speakers = []
    for tpl in diarization.itertracks(yield_label=True):
        if len(tpl) == 3:
            segment, _, label = tpl
        else:
            segment, label = tpl
        time_speakers.append((segment.start, segment.end, label))
    time_speakers.sort(key=lambda x: x[0])

    idx = 0
    for seg in segments:
        for wd in seg["words"]:
            while idx < len(time_speakers) - 1 and wd["start"] >= time_speakers[idx][1]:
                idx += 1
            wd["speaker"] = time_speakers[idx][2]

    all_words = sorted((w for s in segments for w in s["words"]), key=lambda w: w["start"])
    speaker_turns = []
    for spk, words in groupby(all_words, key=lambda w: w["speaker"]):
        words = list(words)
        speaker_turns.append({
            "speaker": spk,
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": " ".join(w["word"] for w in words),
        })
    return speaker_turns

def format_transcript(turns: List[dict]) -> str:
    return "\n".join(f"{t['speaker']}: {t['text'].strip()}" for t in turns if t.get("text"))

def generate_insights(turns: List[dict], api_key: str, model_name: str, temperature: float = 0.3) -> dict:
    client = OpenAI(api_key=api_key)
    prompt = """You are an expert conversation analyst. Analyse the following podcast transcript and return **only** valid JSON with this schema:

{
  "topic": string,
  "speaker_opinions": { "<speaker>": [string, …] }
}

Transcript:
```
%s
```""" % format_transcript(turns)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

def summarise_with_gpt(text, api_key, model_name):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": "Summarise the following podcast transcript in 3‑5 sentences:\n\n" + text,
        }],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def load_hf_summarizer(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summarizer = hf_pipeline(
        "summarization",
        model=model_name,
        device=device,
        framework="pt"
    )
    return summarizer

def summarise_with_hf(text: str, summariser) -> str:
    try:
        max_len = 1024
        if len(text.split()) <= max_len:
            result = summariser(
                text,
                max_length=180,
                min_length=60,
                do_sample=False,
                truncation=True
            )
            return result[0]["summary_text"].strip()

        # For longer texts, split into chunks
        parts = []
        words = text.split()
        for i in range(0, len(words), max_len):
            chunk = " ".join(words[i:i+max_len])
            result = summariser(
                chunk,
                max_length=180,
                min_length=60,
                do_sample=False,
                truncation=True
            )
            parts.append(result[0]["summary_text"].strip())
        return " ".join(parts)
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        return "Error: Could not generate summary. Please try a different summarization model."

# ----------- Chatbot helper -------------- #
def answer_question(question: str, transcript: str, summary: str, api_key: str, model_name: str) -> str:
    """Send the user's question + context to OpenAI and return the answer."""
    client = OpenAI(api_key=api_key)

    context = (summary + "\n\n" + transcript)[:CHAT_MAX_CHARS]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant. Use the provided podcast transcript "
                "to answer the user's questions. If the answer is not in the transcript, say so."
            ),
        },
        {"role": "system", "content": f"Transcript context:\n{context}"},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# -------- Flask routes -------- #

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        url = request.form.get("youtube_url")
        start_min = int(request.form.get("start_min", 0))
        end_min = int(request.form.get("end_min", 5))
        transcriber_choice = request.form.get("transcriber_choice", "Whisper")
        summariser_choice = request.form.get("summariser_choice", "GPT")
        gpt_summary_model = request.form.get("gpt_summary_model", OPENAI_DEFAULT_MODEL)
        gpt_insight_model = request.form.get("gpt_insight_model", OPENAI_DEFAULT_MODEL)

        print(f"DEBUG: Received URL: {url}")
        print(f"DEBUG: start_min={start_min}, end_min={end_min}")
        print(f"DEBUG: Transcriber={transcriber_choice}, Summariser={summariser_choice}")

        # Validate URL
        if not url:
            flash("YouTube URL is required.", "error")
            return redirect(url_for("index"))

        # Get video info
        duration, title = get_video_info(url)
        if start_min >= end_min or end_min * 60 > duration:
            flash("Invalid start/end times.", "error")
            return redirect(url_for("index"))

        # Download & clip audio
        wav_path = download_audio(url)
        clip_path = clip_audio(wav_path, start_min * 60, end_min * 60)

        # Transcribe + diarise
        if transcriber_choice == "AssemblyAI":
            if not ASSEMBLYAI_API_KEY:
                flash("AssemblyAI API key not found. Please check your environment variables.", "error")
                return redirect(url_for("index"))
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(str(clip_path))
            turns = [{
                "speaker": "Speaker 1",
                "start": 0,
                "end": transcript.audio_duration,
                "text": transcript.text
            }]
        else:  # Whisper
            whisper_model, device = load_whisper_model()
            if not HF_TOKEN:
                flash("HuggingFace token not found. Speaker diarization will be limited.", "warning")
                turns = [{
                    "speaker": "Speaker",
                    "start": 0,
                    "end": duration,
                    "text": whisper_model.transcribe(str(clip_path))["text"]
                }]
            else:
                dia_pipe = load_diarization_pipeline(HF_TOKEN)
                chunks = list(chunk_audio(clip_path))
                segments = transcribe_chunks(chunks, whisper_model, device)
                dia_result = dia_pipe(str(clip_path))
                turns = assign_speakers(segments, dia_result)

        # Prepare full transcript text
        transcript_text = format_transcript(turns)

        # Summarise
        if summariser_choice == "GPT":
            if not OPENAI_API_KEY:
                flash("OpenAI API key not found. Using BART summarizer instead.", "warning")
                summariser_choice = "BART"
            else:
                summary = summarise_with_gpt(transcript_text, OPENAI_API_KEY, gpt_summary_model)

        if summariser_choice in ["BART", "T5"]:
            model_name = BART_MODEL if summariser_choice == "BART" else T5_MODEL
            summariser = load_hf_summarizer(model_name)
            summary = summarise_with_hf(transcript_text, summariser)

        # Generate insights
        if not OPENAI_API_KEY:
            insights = {
                "topic": "OpenAI API key not found in environment variables",
                "speaker_opinions": {
                    "Note": ["Please check your environment variables for OpenAI API key"]
                }
            }
        else:
            insights = generate_insights(turns, OPENAI_API_KEY, gpt_insight_model)

        # Render results page
        return render_template(
            "result.html",
            title=title,
            transcript=transcript_text,
            summary=summary,
            insights=insights,
            start_min=start_min,
            end_min=end_min,
            url=url,
        )
    except Exception as e:
        flash(f"Error during processing: {e}", "error")
        return redirect(url_for("index"))

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        question = data.get("question")
        transcript = data.get("transcript")
        summary = data.get("summary")
        model_name = data.get("model_name", OPENAI_DEFAULT_MODEL)
        
        if not all([question, transcript, summary]):
            return jsonify({"error": "Missing required parameters"}), 400
            
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return jsonify({"error": "OpenAI API key not found"}), 500
            
        answer = answer_question(
            question=question,
            transcript=transcript,
            summary=summary,
            api_key=openai_key,
            model_name=model_name
        )
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)


