# Streamlit app: YouTube → transcript → insights
"""
A self‑contained Streamlit UI that lets you:
1. Paste a YouTube podcast URL and immediately preview the video.
2. Pick a start / end time range with a slider.
3. Click **Generate insights** to run Whisper + PyAnnote + GPT‑4o and get a topic, summary, and per‑speaker opinions.

Environment variables required (set in `.streamlit/secrets.toml` or your shell):
- `OPENAI_API_KEY` – for GPT‑4o
- `HF_TOKEN`        – HuggingFace token for `pyannote/speaker-diarization-3.1`

Install deps:
```bash
pip install streamlit yt-dlp pydub tqdm "torch>=2.1" torchaudio git+https://github.com/openai/whisper.git pyannote.audio
```
Run:
```bash
streamlit run podcast_insights_app.py
```
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Tuple, List

import streamlit as st
from pydub import AudioSegment
from tqdm.auto import tqdm
from openai import OpenAI
import whisper
import yt_dlp
from pyannote.audio import Pipeline
import torch


# ───────────────────────── Configuration ────────────────────────── #
WHISPER_MODEL_NAME = "base.en"   # adjust if you have a beefier GPU
CHUNK_MINUTES      = 1           # chunk size for Whisper
NUM_THREADS        = 4           # parallel Whisper workers
OUTPUT_DIR         = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Cache‑able helpers ------------------------------------------------ #
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str = "base.en"):
    # Prefer GPU if possible, but we know MPS may break.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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

@st.cache_data(show_spinner=False)
def get_video_info(url: str) -> Tuple[int, str]:
    """Return duration in seconds and video title using yt‑dlp metadata extraction."""
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    duration = info.get("duration", 0)  # seconds
    title = info.get("title", "Podcast")
    return duration, title

# Audio download / chunk utils ------------------------------------- #

def download_audio(url: str) -> Path:
    """Download best‑quality audio as a .wav and return path."""
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
    """Return a trimmed .wav covering [start_s, end_s]."""
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

# Whisper transcription -------------------------------------------- #

def transcribe_chunks(chunks: List[Tuple[float, Path]], model, device) -> List[dict]:
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        futs = [ex.submit(_one, off, p) for off, p in chunks]
        for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
            segments.extend(f.result())
    segments.sort(key=lambda s: s["start"])
    return segments

# Speaker assignment ------------------------------------------------ #

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
            "text":    " ".join(w["word"] for w in words)
        })
    return speaker_turns

# GPT‑4o insight generation ---------------------------------------- #

def format_transcript(turns: List[dict]) -> str:
    return "\n".join(f"{t['speaker']}: {t['text'].strip()}" for t in turns if t.get("text"))


def generate_insights(turns: List[dict], api_key: str, temperature: float = 0.3) -> dict:
    client = OpenAI(api_key=api_key)
    prompt = f"""You are an expert conversation analyst. Analyze the following podcast transcript and return **only** valid JSON (no commentary) with this schema:\n\n{{\n  \"topic\": string,\n  \"speaker_opinions\": {{\n     \"<speaker>\": [string, …]\n  }},\n  \"summary\": string\n}}\n\nTranscript:\n```\n{format_transcript(turns)}\n```"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# ────────────────────────── Streamlit UI ────────────────────────── #

st.set_page_config(page_title="Podcast Insight Generator", layout="centered")
st.title("🎙️ Podcast Insight Generator")

url = st.text_input("Paste a YouTube podcast URL", placeholder="https://www.youtube.com/watch?v=…")

if url:
    try:
        duration, title = get_video_info(url)
    except Exception as e:
        st.error(f"Failed to fetch video info: {e}")
        st.stop()

    st.video(url)
    dur_min = duration // 60
    st.markdown(f"**Video title:** {title}  •  **Duration:** {timedelta(seconds=duration)}")

    start_min, end_min = st.slider(
        "Select portion to analyze (minutes)",
        0,
        max(1, dur_min),
        (0, min(5, dur_min)),
        step=1,
        key="range_slider",
    )

    if start_min >= end_min:
        st.error("Start time must be < end time")
        st.stop()

    if st.button("Generate insights", type="primary"):
        with st.status("Downloading audio…"):
            wav_path = download_audio(url)
        with st.status("Clipping audio…"):
            clip_path = clip_audio(wav_path, start_min * 60, end_min * 60)
        with st.status("Loading models…"):
            whisper_model, device = load_whisper()
            dia_pipe = load_diarization_pipeline(os.getenv("HF_TOKEN", ""))

        # transcription with progress bar
        st.info("Transcribing with Whisper… this can take a while.")
        chunks = list(chunk_audio(clip_path))
        segments = transcribe_chunks(chunks, whisper_model, device)

        # diarization
        with st.status("Running speaker diarization…"):
            dia_result = dia_pipe(str(clip_path))
        turns = assign_speakers(segments, dia_result)

        # GPT‑4o insights
        with st.spinner("Talking to GPT‑4o…"):
            insights = generate_insights(turns, os.getenv("OPENAI_API_KEY", ""))

        # ── Display results ──────────────────────────────────────── #
        st.subheader("Topic")
        st.write(insights.get("topic"))

        st.subheader("Summary")
        st.write(insights.get("summary"))

        st.subheader("Speaker opinions")
        for spk, bullets in insights.get("speaker_opinions", {}).items():
            st.markdown(f"**{spk}**")
            for b in bullets:
                st.markdown(f"- {b}")

        # Allow downloading JSON
        st.download_button(
            "Download raw JSON",
            data=json.dumps(insights, indent=2, ensure_ascii=False),
            file_name="insights.json",
            mime="application/json",
        )

        # footer
        st.caption("Made with Whisper, PyAnnote, and GPT‑4o ✨")
