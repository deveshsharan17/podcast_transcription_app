# podcast_insights_app.py  –  optimised CPU‑only version
from __future__ import annotations
import json, os, tempfile, uuid, threading, concurrent.futures
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from pydub import AudioSegment
from tqdm.auto import tqdm
from openai import OpenAI
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import yt_dlp
import torch


# ─────────────────── Config ──────────────────── #
WHISPER_MODEL_SIZE = "base.en"      # faster‑whisper size id
WHISPER_COMPUTE    = "int8"         # int8 quantisation → fastest CPU
CHUNK_MINUTES      = 1
NUM_THREADS        = min(8, os.cpu_count() or 4)
OUTPUT_DIR         = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ───────────── Cacheable helpers ─────────────── #
@st.cache_resource(show_spinner=False)
def load_whisper_fast():
    """faster‑whisper model (+ device hint)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=device,
        compute_type=WHISPER_COMPUTE   # "int8" on CPU, "int8_float16" on GPU
    )
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
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("duration", 0), info.get("title", "Podcast")


# ───────────── I/O utilities ──────────────── #
@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def clip_audio(src_wav: Path, start_s: int, end_s: int) -> Path:
    audio = AudioSegment.from_file(src_wav)
    audio = audio.set_channels(1).set_frame_rate(16_000)   # mono‑16 kHz early
    clipped = audio[start_s * 1000 : end_s * 1000]
    dst = src_wav.with_name(f"{src_wav.stem}_{start_s}s_{end_s}s.wav")
    clipped.export(dst, format="wav")
    return dst


def chunk_audio(wav_file: Path, minutes: int = CHUNK_MINUTES):
    audio = AudioSegment.from_file(wav_file)
    ms = minutes * 60_000
    for i in range(0, len(audio), ms):
        chunk = audio[i:i + ms]
        path = wav_file.with_suffix(f".part{i//ms}.wav")
        chunk.export(path, format="wav")
        yield i / 1000.0, path


# ───────────── Processing primitives ───────────── #
def transcribe_chunks(chunks: List[Tuple[float, Path]], model) -> List[dict]:
    def _one(offset_s: float, path: Path):
        # faster-whisper returns List[Segment], Segment.words is List[Word]
        segments, _ = model.transcribe(
            str(path),
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
        )

        converted = []
        for seg in segments:
            # build a dict matching your old schema
            seg_dict = {
                "start": seg.start + offset_s,
                "end":   seg.end   + offset_s,
                "text":  seg.text,
                "words": []
            }
            for w in seg.words:
                seg_dict["words"].append({
                    "start": w.start + offset_s,
                    "end":   w.end   + offset_s,
                    "word":  w.word
                })
            converted.append(seg_dict)
        return converted

    segments: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(NUM_THREADS) as ex:
        futs = [ex.submit(_one, off, p) for off, p in chunks]
        for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
            segments.extend(f.result())
    segments.sort(key=lambda s: s["start"])
    return segments



def diarize_chunks(chunks: List[Tuple[float, Path]], dia_pipe) -> List[Tuple[float, any]]:
    def _run(offset_s, path):
        diar = dia_pipe(str(path))
        return offset_s, diar
    diar_results = []
    with concurrent.futures.ThreadPoolExecutor(NUM_THREADS) as ex:
        futs = [ex.submit(_run, off, p) for off, p in chunks]
        for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
            diar_results.append(f.result())
    return diar_results


def assign_speakers(segments: List[dict], diar_results) -> List[dict]:
    from itertools import groupby
    # Flatten diarization with absolute times
    time_speakers = []
    for off, diar in diar_results:
        for segment, _, label in diar.itertracks(yield_label=True):
            time_speakers.append((segment.start + off, segment.end + off, label))
    time_speakers.sort(key=lambda x: x[0])

    idx = 0
    for seg in segments:
        for wd in seg["words"]:
            while idx < len(time_speakers)-1 and wd["start"] >= time_speakers[idx][1]:
                idx += 1
            wd["speaker"] = time_speakers[idx][2]

    # Collapse to speaker turns
    words_sorted = sorted(
        (w for s in segments for w in s["words"]), key=lambda w: w["start"]
    )
    speaker_turns = []
    for spk, ws in groupby(words_sorted, key=lambda w: w["speaker"]):
        ws = list(ws)
        speaker_turns.append({
            "speaker": spk,
            "start":   ws[0]["start"],
            "end":     ws[-1]["end"],
            "text":    " ".join(w["word"] for w in ws)
        })
    return speaker_turns


# ───────────── LLM insights ───────────── #
def format_transcript(turns: List[dict]) -> str:
    return "\n".join(f"{t['speaker']}: {t['text'].strip()}" for t in turns if t.get("text"))


def generate_insights(turns, api_key: str, temperature: float = .3):
    client = OpenAI(api_key=api_key)
    prompt = f"""You are an expert conversation analyst. Analyse the following podcast transcript
    and return **only** valid JSON with this schema:

    {{
      "topic": string,
      "speaker_opinions": {{"<speaker>": [string, …]}},
      "summary": string
    }}

    Transcript:
    ```
    {format_transcript(turns)}
    ```"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ───────────── Streamlit UI ───────────── #
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
        "Select portion to analyse (minutes)",
        0, max(1, dur_min),
        (0, min(5, dur_min)), step=1
    )
    if start_min >= end_min:
        st.error("Start time must be < end time")
        st.stop()

    if st.button("Generate insights", type="primary"):
        with st.status("Downloading audio…"):
            wav_path = download_audio(url)

        with st.status("Clipping + down‑mixing audio…"):
            clip_path = clip_audio(wav_path, start_min * 60, end_min * 60)

        with st.status("Loading models…"):
            whisper_model, _ = load_whisper_fast()
            dia_pipe = load_diarization_pipeline(os.getenv("HF_TOKEN", ""))

        # --- Transcription
        st.info("Transcribing with faster‑whisper…")
        chunks = list(chunk_audio(clip_path))
        segments = transcribe_chunks(chunks, whisper_model)

        # --- Diarisation (parallel)
        with st.status("Running speaker diarisation…"):
            diar_results = diarize_chunks(chunks, dia_pipe)
        turns = assign_speakers(segments, diar_results)

        # --- GPT‑4o
        with st.spinner("Talking to GPT‑4o…"):
            insights = generate_insights(turns, os.getenv("OPENAI_API_KEY", ""))

        # --- Display
        st.subheader("Topic");            st.write(insights.get("topic"))
        st.subheader("Summary");          st.write(insights.get("summary"))
        st.subheader("Speaker opinions")
        for spk, bullets in insights.get("speaker_opinions", {}).items():
            st.markdown(f"**{spk}**")
            for b in bullets: st.markdown(f"- {b}")

        st.download_button(
            "Download raw JSON",
            data=json.dumps(insights, indent=2, ensure_ascii=False),
            file_name="insights.json",
            mime="application/json",
        )
        st.caption("Made with faster‑Whisper, PyAnnote, and GPT‑4o ✨")
