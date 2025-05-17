# Project Overview

This repository contains a lightweight Python application (entry point: `app.py`) that leverages state‑of‑the‑art BART and T5 transformer models to summarise text and AssemblyAI + OpenAI services for speech‑to‑text, language understanding and generation.

Because API credentials **must be kept private** and the transformer checkpoints are several hundred MB each, neither the environment file nor the model folders are checked into version control. Follow the instructions below to prepare your local environment.

---

## Repository Structure

```text
.
├── app.py                  # 🚀 Main application entry point
├── requirements.txt        # Python dependencies
├── README.md               # (This file)
├── utils/                  # Helper utilities (pre‑/post‑processing, logging, …)
├── venv/                   # 💡 *Ignored* – local virtual‑env
├── outputs/                # 💡 *Ignored* – generated artefacts (logs, results, etc.)
├── bart_summarization_model/   # 💡 *Ignored* – downloaded BART model weights
└── t5_summarization_model/     # 💡 *Ignored* – downloaded T5 model weights
```

### Why are some folders ignored?

The projectʼs **`.gitignore`** (or `.dockerignore`) excludes:

* `.env` – holds your private API keys. Never commit credentials to VCS.
* `venv/` – virtual‑envs are machine‑specific.
* `outputs/` – runtime artefacts that can be re‑generated.
* `bart_summarization_model/` & `t5_summarization_model/` – large model checkpoints that would bloat the repo.

---

## 1. Prerequisites

| Requirement | Version (tested) |
| ----------- | ---------------- |
| Python      | ≥ 3.9            |
| `pip`       | ≥ 23             |
| Git         | any              |

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Configure API Keys

Create a file named **`.env`** in the project root **or** export the variables in your shell. The application expects the following keys:

```bash
# .env (do NOT commit!)
ASSEMBLYAI_API_KEY=<your‑assemblyai‑key>
OPENAI_API_KEY=<your‑openai‑key>
```

> **Required** – the app will exit if either key is missing.

---

## 3. Download the Summarisation Models

Both BART and T5 checkpoints are hosted on Hugging Face Hub. Use one of the options below; the destination folders must match the names ignored above (`bart_summarization_model/` and `t5_summarization_model/`).

### Option A – Python helper script (recommended)

```python
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = {
    "bart_summarization_model": "facebook/bart-large-cnn",  # or any fine‑tuned variant
    "t5_summarization_model":   "t5-base"                  # or t5‑large, etc.
}

for folder, model_name in MODELS.items():
    path = Path(folder)
    path.mkdir(exist_ok=True)
    print(f"Downloading {model_name} → {path} …")
    AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=path)
    AutoTokenizer.from_pretrained(model_name, cache_dir=path)
```

Run:

```bash
python download_models.py  # assuming you saved the snippet as download_models.py
```

### Option B – `huggingface-cli`

```bash
# Install the CLI if you don't have it
pip install --upgrade huggingface_hub

# BART
huggingface-cli download facebook/bart-large-cnn --local-dir bart_summarization_model --local-dir-use-symlinks False
# T5
huggingface-cli download t5-base --local-dir t5_summarization_model --local-dir-use-symlinks False
```

---

## 4. Run the Application

```bash
python app.py --help  # view CLI options

```

The script will:

1. **Transcribe** audio (if an audio file is provided) via AssemblyAI.
2. **Summarise** text using BART or T5.
3. Optionally **refine / expand** the summary with OpenAI GPT.
4. Write outputs to the `outputs/` folder.

---

## 5. Troubleshooting

| Symptom                          | Fix                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------ |
| `KeyError: 'ASSEMBLYAI_API_KEY'` | Ensure `.env` is present and spelled correctly, then restart your terminal or IDE.               |
| Slow first run                   | The models need to be downloaded and cached (\~1–2 GB combined). Subsequent runs will be faster. |
| CUDA not used                    | Install a GPU‑enabled PyTorch build or set the `--device cpu` flag.                              |

---

## 6. Contributing

Feel free to open issues or PRs for bug fixes, performance tweaks, or new features! 🤗

---

## License

[MIT](LICENSE)

