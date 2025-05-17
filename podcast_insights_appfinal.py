# podcast_insights_app.py – YouTube → transcript → summary + insights
"""
A self‑contained Streamlit UI that lets you:
1. Paste a YouTube podcast URL and immediately preview the video.
2. Pick a start / end time range with a slider.
3. Choose **Whisper (local)** or **AssemblyAI** for transcription + diarisation (left sidebar).
4. Choose **BART**, **T5** or **GPT** for the summary model (left sidebar).
5. Choose which OpenAI chat‑model to use for insights generation (and for summary if you picked GPT).
6. Click **Generate** to run the full pipeline and get:
   • transcript summary (model‑dependent)
   • topic
   • per‑speaker opinions

Environment variables / secrets required (set in `.streamlit/secrets.toml` or your shell):
- `OPENAI_API_KEY`  – for OpenAI chat models
- `HF_TOKEN`         – HuggingFace token for `pyannote/speaker-diarization-3.1`
- `AssAI` *or* `ASSEMBLYAI_API_KEY` – AssemblyAI key (only if you pick the cloud engine)

Install deps:
```bash
pip install streamlit yt-dlp pydub tqdm "torch>=2.1" torchaudio git+https://github.com/openai/whisper.git \
            pyannote.audio transformers openai assemblyai
```
Run:
```bash
streamlit run podcast_insights_app.py
```
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import uuid
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import torch
import whisper
import yt_dlp
from openai import OpenAI
from pydub import AudioSegment
from pyannote.audio import Pipeline
from tqdm.auto import tqdm
from transformers import pipeline as hf_pipeline
import assemblyai as aai

# ───────────────────────── Configuration ────────────────────────── #
WHISPER_MODEL_NAME = "base.en"   # adjust if you have a beefier GPU
CHUNK_MINUTES      = 1           # chunk size for Whisper
NUM_THREADS        = 4           # parallel Whisper workers
OUTPUT_DIR         = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# summarisation model hub ids
BART_MODEL = "facebook/bart-large-cnn"
T5_MODEL   = "google/flan-t5-base"  # light enough for CPU ↔ quality trade-off

OPENAI_DEFAULT_MODEL = "gpt-4o"
OPENAI_ALT_MODELS    = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo-preview",
    "gpt-3.5-turbo",
]

# ─────────────────────── Cached helpers ─────────────────────────── #
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str = WHISPER_MODEL_NAME):
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        print(f"⚠️  Could not load on {device}: {e!r}\n   Falling back to CPU.")
        device = "cpu"
        model = whisper.load_model(model_size, device="cpu")
    return model, device

@st.cache_resource(show_spinner=False)
def load_diarization_pipeline(token: str):
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
        cache_dir=str(Path.home() / ".cache/pyannote"),
    )

@st.cache_resource(show_spinner=False)
def load_hf_summarizer(model_name: str):
    """Return a HuggingFace transformers summariser pipeline."""
    device = 0 if torch.cuda.is_available() else -1
    return hf_pipeline("summarization", model=model_name, device=device)

@st.cache_data(show_spinner=False)
def get_video_info(url: str) -> Tuple[int, str]:
    """Return duration in seconds and video title (yt‑dlp metadata)."""
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    duration = info.get("duration", 0)
    title = info.get("title", "Podcast")
    return duration, title

# ──────────────────────── Audio helpers ─────────────────────────── #

def download_audio(url: str) -> Path:
    """Download best‑quality audio as .wav and return path."""
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
        yield i / 1000.0, chunk_path  # offset‑seconds, path

# ───────────────────── Whisper transcription ───────────────────── #

def transcribe_chunks(chunks: List[Tuple[float, Path]], model, device) -> List[dict]:
    lock = threading.Lock()

    def _one(offset_s, path):
        with lock:
            res = model.transcribe(
                str(path), word_timestamps=True, verbose=False, fp16=device == "cuda"
            )
        for seg in res["segments"]:
            seg["start"] += offset_s
            seg["end"]   += offset_s
            for w in seg["words"]:
                w["start"] += offset_s
                w["end"]   += offset_s
        return res["segments"]

    segments = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        futs = [ex.submit(_one, off, p) for off, p in chunks]
        for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
            segments.extend(f.result())
    segments.sort(key=lambda s: s["start"])
    return segments

# ──────────────────── Speaker assignment ───────────────────────── #

def assign_speakers(segments: List[dict], diarization) -> List[dict]:
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
            "start":   words[0]["start"],
            "end":     words[-1]["end"],
            "text":    " ".join(w["word"] for w in words),
        })
    return speaker_turns

# ───────────────────── GPT insight generation ──────────────────── #

def format_transcript(turns: List[dict]) -> str:
    return "\n".join(f"{t['speaker']}: {t['text'].strip()}" for t in turns if t.get("text"))



def generate_insights(turns: List[dict], api_key: str, model_name: str, temperature: float = 0.3) -> dict:
    """Return topic + speaker opinions using a GPT model."""
    client = OpenAI(api_key=api_key)
    prompt = """You are an expert conversation analyst. Analyse the following podcast transcript and return **only** valid JSON with this schema:\n\n{\n  \"topic\": string,\n  \"speaker_opinions\": {\n    \"<speaker>\": [string, …]\n  }\n}\n\nTranscript:\n```\n%s\n```""" % format_transcript(turns)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# ──────────────────── Summarisation helpers ────────────────────── #

def summarise_with_gpt(text: str, api_key: str, model_name: str) -> str:
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


def summarise_with_hf(text: str, summariser) -> str:
    # transformers pipelines limit input length; chunk if needed
    max_len = 1024  # tokens ~ words approximation
    if len(text.split()) <= max_len:
        return summariser(text, max_length=180, min_length=60, do_sample=False)[0]["summary_text"].strip()

    # chunking for long texts
    parts = []
    words = text.split()
    for i in range(0, len(words), max_len):
        chunk = " ".join(words[i : i + max_len])
        parts.append(summariser(chunk, max_length=180, min_length=60, do_sample=False)[0]["summary_text"].strip())
    return " " .join(parts)

# ────────────────────────── Streamlit UI ────────────────────────── #

st.set_page_config(page_title="Podcast Insight Generator", layout="centered")

# Sidebar – configuration
with st.sidebar:
    st.header("⚙️ Settings")

    transcriber_choice = st.radio(
        "Transcription / diarisation engine",
        ["Whisper (local)", "AssemblyAI"],
        index=0,
    )

    summariser_choice = st.radio(
        "Summarisation model",
        ["GPT", "BART", "T5"],
        index=0,
        help="Choose which model to generate the transcript summary.",
    )

    # GPT model selection for summary (only shown if chosen)
    if summariser_choice == "GPT":
        gpt_summary_model = st.selectbox(
            "OpenAI model for summary",
            OPENAI_ALT_MODELS,
            index=OPENAI_ALT_MODELS.index(OPENAI_DEFAULT_MODEL),
        )
    else:
        gpt_summary_model = None  # not used

    # GPT model for insights
    gpt_insight_model = st.selectbox(
        "OpenAI model for insights",
        OPENAI_ALT_MODELS,
        index=OPENAI_ALT_MODELS.index(OPENAI_DEFAULT_MODEL),
    )

    st.markdown("---")
    st.caption("Made with Whisper, PyAnnote, AssemblyAI, HuggingFace & OpenAI ✨")

st.title("🎙️ Podcast Insight Generator")

url = st.text_input("Paste a YouTube podcast URL", placeholder="https://www.youtube.com/watch?v=…")

if url:
    try:
        duration, title = get_video_info(url)
    except Exception as e:
        st.error(f"Failed to fetch video info: {e}")
        st.stop()

    st.video(url)
    st.markdown(f"**Video title:** {title}  •  **Duration:** {timedelta(seconds=duration)}")

    dur_min = duration // 60
    start_min, end_min = st.slider(
        "Select portion to analyse (minutes)",
        0,
        max(1, dur_min),
        (0, min(5, dur_min)),
        step=1,
        key="range_slider",
    )

    if start_min >= end_min:
        st.error("Start time must be < end time")
        st.stop()

    if st.button("Generate", type="primary"):
        # ── Download + clip ─────────────────────────────────────── #
        with st.status("Downloading audio…"):
            wav_path = download_audio(url)
        with st.status("Clipping audio…"):
            clip_path = clip_audio(wav_path, start_min * 60, end_min * 60)

        # ── Transcription & diarisation ────────────────────────── #
        if transcriber_choice == "Whisper (local)":
            with st.status("Loading Whisper & diarisation pipeline…"):
                whisper_model, device = load_whisper()
                dia_pipe = load_diarization_pipeline(os.getenv("HF_TOKEN", ""))

            st.info("Transcribing with Whisper… this can take a while.")
            chunks = list(chunk_audio(clip_path))
            segments = transcribe_chunks(chunks, whisper_model, device)

            with st.status("Running speaker diarisation…"):
                dia_result = dia_pipe(str(clip_path))
            turns = assign_speakers(segments, dia_result)

        else:  # AssemblyAI
            with st.status("Transcribing with AssemblyAI… this can take a while."):
                aai_key = (
                    st.secrets["AssAI"] if "AssAI" in st.secrets else os.getenv("ASSEMBLYAI_API_KEY", "")
                )
                if not aai_key:
                    st.error("AssemblyAI key not found in secrets or env vars.")
                    st.stop()

                aai.settings.api_key = aai_key
                config = aai.TranscriptionConfig(
                    speech_model=aai.SpeechModel.best,
                    speaker_labels=True,
                )
                transcript = aai.Transcriber(config=config).transcribe(str(clip_path))

                if transcript.status == "error":
                    st.error(f"Transcription failed: {transcript.error}")
                    st.stop()

                turns = [
                    {
                        "speaker": f"Speaker {u.speaker}",
                        "start":   u.start / 1000.0,
                        "end":     u.end / 1000.0,
                        "text":    u.text.strip(),
                    }
                    for u in transcript.utterances
                    if u.text.strip()
                ]

        # ── Generate summary ───────────────────────────────────── #
        st.subheader("Summary")
        full_transcript_text = " ".join(t["text"] for t in turns if t.get("text"))

        if summariser_choice == "GPT":
            summary_text = summarise_with_gpt(
                full_transcript_text,
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model_name=gpt_summary_model,
            )
        elif summariser_choice == "BART":
            bart_summariser = load_hf_summarizer(BART_MODEL)
            summary_text = summarise_with_hf(full_transcript_text, bart_summariser)
        else:  # T5
            t5_summariser = load_hf_summarizer(T5_MODEL)
            summary_text = summarise_with_hf(full_transcript_text, t5_summariser)

        st.write(summary_text)

        # ── GPT insights ───────────────────────────────────────── #
        with st.spinner("Generating insights with GPT…"):
            insights = generate_insights(
                turns,
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model_name=gpt_insight_model,
            )

        # ── Display insights ───────────────────────────────────── #
        st.subheader("Topic")
        st.write(insights.get("topic"))

        st.subheader("Speaker opinions")
        for spk, bullets in insights.get("speaker_opinions", {}).items():
            st.markdown(f"**{spk}**")
            for b in bullets:
                st.markdown(f"- {b}")

        # ── Allow downloading JSON ─────────────────────────────── #
        export = {
            "summary": summary_text,
            "topic": insights.get("topic"),
            "speaker_opinions": insights.get("speaker_opinions", {}),
        }
        st.download_button(
            "Download raw JSON",
            data=json.dumps(export, indent=2, ensure_ascii=False),
            file_name="podcast_insights.json",
            mime="application/json",
        )
