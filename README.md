# 🧠 Blickshift Assistant – A Metacognitive, Truth‑Aware Conversational AI

> **One‑sentence pitch:**  *Plan* → *Check* → *Speak.*  Blickshift thinks before it talks, so you get answers that are both fluent **and** fact‑checked – in text **or voice** – all on your local machine.

---

## 📜 Table of Contents

1. [Why Blickshift?](#why-blickshift)
2. [System Overview](#system-overview)
3. [Quick Start](#quick-start)
4. [Demo Scenarios](#demo-scenarios)
5. [Project Layout](#project-layout)
6. [Extending & Research Hooks](#extending--research-hooks)
7. [FAQ](#faq)
8. [Roadmap](#roadmap)
9. [License](#license)

---

## Why Blickshift?

LLMs are eloquent but unreliable – they hallucinate, interrupt, and ramble.  **Blickshift** tackles all three by borrowing from *metacognition*:

| Human faculty                                                              | Implementation                                              | Pay‑off                                               |
| -------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| **Prediction‑error monitoring** – sense when a thought feels “off”         | Token‑level surprise hook in the decoder                    | Catches nonsensical continuations early               |
| **Epistemic feelings** – gut‑level confidence, relevance, “is it my turn?” | Fast heuristics: semantic similarity, VAD silence, KG match | Keeps chat on topic & timed like a human conversation |
| **Conceptual introspection** – explicit reasoning & correction             | Large Concept Model (LCM) plans → NLI fact verifier         | Makes sure content is logically sound & sourced       |

The result is a chatbot that **plans in concepts**, **verifies facts on the fly**, and **speaks only when the user is really done**.

---

## System Overview  <a name="system-overview"></a>

```
User (voice or text)
        │
  Whisper / tokenizer ──► Intent →  LCM Planner  ──► Concept plan
        │                               │
        ▼                               │  (confidence, relevance)
Turn‑taking (VAD)                 Factual Verifier  (E5+YAGO)
        │                               │
        └──────── “OK / Revise” ◄───────┘
                         │
                Surface Realiser (Flan‑T5‑LoRA)
                         │
                    Piper TTS  →  Speaker
```

### Core Components

| Layer                 | Open‑source model                       | Size          | Runs on         |
| --------------------- | --------------------------------------- | ------------- | --------------- |
| Planner               | **Meta Large Concept Model (base)**     | 550 MB        | CPU or GPU      |
| Embedding + Retrieval | **intfloat/e5‑small‑v2** + Faiss        | 90 MB         | CPU             |
| Fact Verifier         | **microsoft/deberta‑v3‑base‑mnli‑lora** | 360 MB        | GPU recommended |
| Realiser              | **Flan‑T5‑Small + LoRA**                | 300 MB        | CPU or GPU      |
| ASR / TTS             | **Whisper‑tiny**, **Piper**             | 70 MB / 50 MB | CPU             |

All models are Apache 2.0 or CC licensed – fully offline‑friendly.

---

## Quick Start  <a name="quick-start"></a>

### 1 Prerequisites

* Python 3.9+
* `pip install torch==2.2.0` (CUDA 11.8) or CPU build
* ≈ 2 GB VRAM (GPU) **or** 8 GB RAM (CPU‑only)

### 2 Install

```bash
# clone repo
 git clone https://github.com/yourname/blickshift.git
 cd blickshift

# install deps
 pip install -r requirements.txt

# fetch weights (~1 GB total)
 python scripts/download_weights.py  # resumable
```

### 3 Text‑only demo

```bash
python blickshift_pipeline.py "Explain why solar costs keep falling"
```

### 4 Voice chat

```bash
python audio_chat.py --mic 1 --voice
```

Hold `Space` to talk.  Release → the bot thinks, verifies, then answers aloud.

---

## Demo Scenarios  <a name="demo-scenarios"></a>

| Scenario               | What to try                         | Metacognitive moment                                                                     |
| ---------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------- |
| **Factual correction** | “How many Grammys has Madonna won?” | Planner says **20**, verifier finds source saying **7** → realiser fixes before speaking |
| **Turn‑taking**        | Keep talking without pause          | VAD shows *low turn‑score*, bot waits vs. interrupting                                   |
| **Off‑topic guard**    | Ask physics after a Python question | Relevance score < 0.3 → bot asks for clarification                                       |

---

## Project Layout  <a name="project-layout"></a>

```
├─ data/              # sample KG slices & embeddings
├─ scripts/
│   ├─ download_weights.py
│   └─ prepare_embeddings.py
├─ blickshift_pipeline.py     # main text demo
├─ audio_chat.py              # ASR ↔ TTS loop
├─ lcm_pipeline.py            # planner only
├─ plan_realiser.py           # plan→text demo
└─ notebooks/
    └─ Blickshift_Playground.ipynb
```

---

## Extending & Research Hooks  <a name="extending--research-hooks"></a>

* **Swap planner:** Plug a rule engine (`pyDatalog`) into `PlannerBase` interface.
* **Add citation chains:**  Return the verifier’s top evidence snippets next to each sentence.
* **Fine‑tune realiser style:** Train LoRA on company tone or multiple languages.
* **Integrate gesture signals:** Use webcam to boost turn‑taking heuristics.

---

## FAQ  <a name="faq"></a>

**Q Is this production‑ready?**
*A* No, it’s a research prototype – good for demos, papers, or as a teaching tool on metacognition & LLM alignment.

**Q How fast is it?**
A typical exchange (3 concept plan) takes **< 600 ms** end‑to‑end on RTX‑3060; **\~1.8 s** CPU‑only.

**Q Can I use another LLM as realiser?**
Yes – any seq‑2‑seq model (e.g. Mistral‑7B‑Instruct) that you fine‑tune on `plan → text` pairs.

---

## Roadmap  <a name="roadmap"></a>

| Quarter     | Goal                                                          |
| ----------- | ------------------------------------------------------------- |
|  Q3 ‑ 2025  | Release Docker image; live web demo w/ WebRTC audio           |
|  Q4 ‑ 2025  | Multi‑language support (ES + DE) via SONAR fine‑tune          |
|  Q1 ‑ 2026  | Open dataset of **1 M** plan↔text pairs for realiser training |

---

## License  <a name="license"></a>

Code is **Apache 2.0**.  Model weights keep original licenses (Meta LCM Research, CC‑BY‑SA for ConceptNet, etc.).  Check `LICENSES/` folder for details.

---

> **Blickshift’s credo:**  *"Reason in silence, verify with evidence, then speak with confidence."*

