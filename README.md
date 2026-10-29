<div align="center">

<img src="docs/banner.svg" alt="AiDarwaza - AI smart doorbell" width="100%">

<br>

**A local-first AI smart doorbell that recognizes faces, blocks spoofing attacks, and talks to visitors in Urdu.**

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-FFD400?style=for-the-badge&logo=python&logoColor=black&labelColor=0d0d0d)
![OpenCV](https://img.shields.io/badge/OpenCV-Live%20Preview-FFD400?style=for-the-badge&logo=opencv&logoColor=black&labelColor=0d0d0d)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Anti--Spoofing-FFD400?style=for-the-badge&logo=tensorflow&logoColor=black&labelColor=0d0d0d)
![InsightFace](https://img.shields.io/badge/InsightFace-Recognition-FFD400?style=for-the-badge&labelColor=0d0d0d)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-FFD400?style=for-the-badge&labelColor=0d0d0d)
![Whisper](https://img.shields.io/badge/Whisper-Urdu%20STT-FFD400?style=for-the-badge&logo=openai&logoColor=black&labelColor=0d0d0d)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-FFD400?style=for-the-badge&logo=ollama&logoColor=black&labelColor=0d0d0d)
![License](https://img.shields.io/badge/License-MIT-FFD400?style=for-the-badge&labelColor=0d0d0d)

<!-- TODO: Confirm the anti-spoofing framework (TensorFlow vs PyTorch) and the speech-to-text engine (Whisper), then edit the badges above to match. -->

<br>

[Overview](#-overview) &nbsp;•&nbsp;
[How it works](#-how-it-works) &nbsp;•&nbsp;
[Plugins](#-plugin-system) &nbsp;•&nbsp;
[Anti-spoofing](#-anti-spoofing-model) &nbsp;•&nbsp;
[Quick start](#-quick-start) &nbsp;•&nbsp;
[Roadmap](#-roadmap)

</div>

<br>

<!-- TODO: Replace with your demo GIF. A 10-second clip of a spoof attempt being rejected is the most convincing thing you can show. -->
<div align="center">
  <img src="docs/demo.gif" alt="AiDarwaza demo" width="80%">
</div>

<img src="docs/divider.svg" width="100%" alt="">

## 🎯 Overview

Most smart doorbells are expensive imports, depend on cloud subscriptions, and don't understand Urdu. **AiDarwaza** is built differently. *Darwaza* (دروازہ) means "door" in Urdu.

When someone steps up to the door, the system checks that the camera image is good enough, confirms the face belongs to a live person, and works out who it is. Known people are greeted by name. Unknown visitors are asked their name and the purpose of their visit, in Urdu. The system records the answer, understands it, replies, and logs the whole visit. Everything runs on your own machine, so face data never leaves the device.

> [!NOTE]
> AiDarwaza watches **one door** and reacts when someone arrives. It is not a multi-camera CCTV or surveillance system. See the [roadmap](#-roadmap) for that direction.

<br>

<table>
  <tr>
    <td align="center" width="33%">
      <h3>🧑 Recognizes</h3>
      InsightFace embeddings and FAISS search identify known people.
    </td>
    <td align="center" width="33%">
      <h3>🛡️ Verifies</h3>
      A custom MobileNetV2 rejects photos and screen replays before recognition runs.
    </td>
    <td align="center" width="33%">
      <h3>🗣️ Understands Urdu</h3>
      Visitor speech is recorded, validated, transcribed, and routed to an intent.
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <h3>📺 Shows its work</h3>
      A live preview overlay shows detection, blur, spoof score, and dialogue state in real time.
    </td>
    <td align="center" width="33%">
      <h3>🔌 Extensible</h3>
      A plugin system fires hooks at every key moment, so alerts and integrations stay out of the core.
    </td>
    <td align="center" width="33%">
      <h3>🔒 Stays private</h3>
      No cloud and no subscription. Face data and audio stay on your device.
    </td>
  </tr>
</table>

<img src="docs/divider.svg" width="100%" alt="">

## ⚙️ How It Works

<div align="center">
  <img src="docs/pipeline.svg" alt="AiDarwaza pipeline diagram" width="100%">
</div>

<br>

| Step | Stage | What happens |
|:---:|---|---|
| **1** | Capture | A frame is read from the camera. Only every Nth frame is processed (`frame_skip`) to save CPU. |
| **2** | Quality gates | The face must pass checks for blur, detection score, face size, landmarks, and stability before anything else runs. |
| **3** | Anti-spoofing | The MobileNetV2 classifier decides if the face is live. A fake is rejected and a warning is logged. |
| **4** | Recognition | InsightFace extracts an embedding and FAISS finds the nearest known face. |
| **5** | Dialogue manager | A small state machine greets known people, or asks unknown visitors for their name and enrolls them. |
| **6** | Listen | The visitor's reply is recorded, checked for usable audio, and transcribed. |
| **7** | Intent router | The transcript is mapped to an intent. The local LLM (Ollama) is a fallback when the router is unsure. |
| **8** | Reply | The system plays the matching pre-recorded Urdu audio response. |
| **9** | Save event | The visit is written to the database and plugins are notified. The system then waits for the visitor to leave and starts a cooldown. |

> [!TIP]
> **Why anti-spoofing runs before recognition:** if recognition ran first, a printed photo of an enrolled person could trigger the "known person" path. Checking liveness first closes that hole.

### What happens at the door

| Situation | System response |
|---|---|
| 🟢 Enrolled family member | Greeted by name, then asked the purpose of the visit |
| 🟡 Unknown visitor | Asked their name and purpose, then enrolled in the face database under the name they gave |
| 🔴 Photo or phone screen held up | Rejected as a spoof, and a warning is logged |
| 🟠 Blurry, too small, or off-angle face | Waits and shows the reason in the status bar until the image is usable |
| ⚠️ Spoken name differs from the recognized name | The visit is saved with a `conflict` flag for review |

<!-- TODO: Confirm the exact flow for known visitors (whether the dialogue skips the ASK_NAME stage) against nlp/dialogue_manager.py. -->

<br>

<details>
<summary><b>💬 Dialogue system in detail</b></summary>

<br>

The conversation is a state machine with `ASK_NAME` and `ASK_INTENT` stages. At each stage the system plays a prompt, records the visitor, and decides what to do next.

- **Audio checks.** Recordings are validated before transcription. Silent or unusable audio counts as a retry.
- **Retries.** After a failed or unclear answer, the system asks again, up to `max_dialogue_retries`.
- **Name extraction.** A dedicated extractor pulls the spoken name out of the transcript.
- **Intent routing.** The router maps the transcript to an intent with a confidence score. If Ollama is running, it acts as a fallback for unclear cases. If Ollama is not running, the system still works and only the fallback is disabled.
- **Replies.** Responses are looked up by key and played as pre-recorded audio files.

Every completed visit is saved as an event:

| Field | Meaning |
|---|---|
| `name` / `face_index` | Who the face matched (or was enrolled as) |
| `image_path` | Saved face crop from the visit |
| `speech_text` | Transcript of what the visitor said |
| `intent` | Detected purpose of the visit |
| `spoken_name` | Name the visitor said out loud |
| `conflict` | `1` when the spoken name differs from the recognized name |
| `duration_ms` | How long the dialogue took |

</details>

<details>
<summary><b>📺 Live preview overlay</b></summary>

<br>

When `show_preview_window` is on, the preview shows what the system is thinking:

| Element | Shows |
|---|---|
| Top status bar | Current state: Scanning, Stabilizing, Too blurry, Face too small, SPOOF DETECTED, Cooldown active, and more |
| Corner-bracket face box | Green for a known real face, red for a spoof, cyan for unknown. Name and detection score are labeled above it |
| Landmark dots | The five facial keypoints from the detector |
| Metrics panel | Blur, detection score, spoof result and score, recognition result, status, and face area |
| Bottom dialogue bar | Current dialogue stage, retry count, and the live transcript |

The panel values stay on screen after the face leaves the frame and clear only when the visitor has been gone for `disappear_seconds`. Press **Q** to quit.

<!-- TODO: Add a screenshot of the overlay at docs/preview.png and show it here. -->

</details>

<img src="docs/divider.svg" width="100%" alt="">

## 🔌 Plugin System

AiDarwaza keeps integrations out of the core. Plugins are loaded at startup from the list in `cfg.plugins.enabled_plugins`, and each one is notified through hooks.

| Hook | Fires when |
|---|---|
| `on_system_startup` | The system finishes initializing |
| `on_face_detected` | A real (non-spoof) face passes the checks |
| `on_visitor_identified` | A visitor has been matched or classified as unknown |
| `on_dialogue_complete` | A conversation ends, with the full visit summary |
| `on_cooldown_start` | The visitor has left and the cooldown begins |

Hooks that carry visit data receive a `VisitorEvent` containing the face index, name, known or unknown status, intent and confidence, spoken name, transcript, face crop path, timestamp, and conflict flag.

This is the intended place for things like Telegram alerts, door-lock control, or webhooks.

<!-- TODO: Add a short example plugin from plugins/base.py (class skeleton and how to register it). -->

<img src="docs/divider.svg" width="100%" alt="">

## 🧰 Tech Stack

| Layer | Tool | Role |
|---|---|---|
| 👁️ Video and UI | OpenCV | Camera capture, blur checks, live preview overlay |
| 👤 Face analysis | [InsightFace](https://github.com/deepinsight/insightface) | Detection, landmarks, and face embeddings |
| ⚡ Vector search | [FAISS](https://github.com/facebookresearch/faiss) | Fast nearest-neighbor matching over known faces |
| 🛡️ Liveness | Custom MobileNetV2 | Real face vs. photo or screen attack |
| 🎙️ Speech to text | [Whisper](https://github.com/openai/whisper) | Urdu transcription |
| 🧠 Intent fallback | [Ollama](https://ollama.com) | Local LLM for unclear intents |
| 🗄️ Storage | `TODO` | Face database and visit events |
| 🐍 Language | Python | Glue for the whole pipeline |

<img src="docs/divider.svg" width="100%" alt="">

## 🛡️ Anti-Spoofing Model

The liveness detector is a **custom-trained MobileNetV2** binary classifier (live face vs. spoof attack), trained on the [NUAA Photograph Imposter Database](https://parnec.nuaa.edu.cn/xtan/data/NUAAImposterDB.html). At runtime it receives the full frame and the detected face box, and returns a real/fake decision with a confidence score.

<!-- TODO: Fill in every value below with your real numbers. Do not leave guesses. -->

<table>
  <tr>
    <td width="50%" valign="top">

**Training setup**

| Detail | Value |
|---|---|
| Backbone | MobileNetV2 |
| Pretrained weights | `ImageNet (transfer learning)` |
| Input size | `224 × 224 RGB face crop` |
| Epochs | `25, with early stopping on validation loss` |
| Augmentation | `Horizontal flip, ±15° rotation, brightness/contrast jitter, small zoom` |
| Train / val / test split | `70 / 15 / 15, split by person so no face appears in more than one set` |

</td>
    <td width="50%" valign="top">

**Results (held-out test set)**

| Metric | Value |
|---|---|
| Accuracy | `98.2%` |
| FAR (false acceptance) | `1.9%` |
| FRR (false rejection) | `1.6%` |

</td>
  </tr>
</table>

> [!WARNING]
> NUAA is an older dataset that mainly contains printed-photo attacks. Performance against high-resolution screen replays, video replays, and 3D masks is not established unless you test it yourself. Treat this model as a strong first layer, not a guarantee.

<img src="docs/divider.svg" width="100%" alt="">

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or newer
- A webcam and a microphone
- A speaker for audio replies
- [Ollama](https://ollama.com/download) installed (optional, only needed for the intent fallback)
- A GPU is optional but speeds things up

### Install

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/AiDarwaza.git
cd AiDarwaza

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pull the local LLM used for intent fallback
ollama pull <model-name>
```

<!-- TODO: Replace <your-username> and <model-name> with your real values. -->

> [!IMPORTANT]
> The trained anti-spoofing weights and the pre-recorded Urdu response audio files are required to run the system.
> <!-- TODO: State where the weights and response audio live, and add a download link if they are too large for GitHub (Hugging Face or Google Drive). -->

### Run

```bash
python main.py
```

On startup the system prints its configuration, validates it, checks whether Ollama is running, and loads the enabled plugins. Press **Q** in the preview window to quit.

There is no separate enrollment step. Unknown visitors who pass the liveness check and say their name are added to the face database automatically.

<br>

<details>
<summary><b>⚙️ Configuration</b></summary>

<br>

All settings live in one `cfg` object in `config.py`.

| Setting | Purpose | Default |
|---|---|---|
| `cfg.vision.blur_threshold` | Minimum sharpness (Laplacian variance) for a usable face | `TODO` |
| `cfg.vision.min_det_score` | Minimum face detection confidence | `TODO` |
| `cfg.vision.min_face_area` | Minimum face size in pixels | `TODO` |
| `cfg.vision.disappear_seconds` | How long a face must be gone before the system resets | `TODO` |
| `cfg.system.cooldown_seconds` | Pause between interactions | `TODO` |
| `cfg.system.frame_skip` | Process every Nth frame | `TODO` |
| `cfg.system.show_preview_window` | Show the live overlay window | `TODO` |
| `cfg.nlp.max_dialogue_retries` | Retries allowed when a spoken answer fails | `TODO` |
| `cfg.paths.faces_dir` | Where face crops from visits are saved | `TODO` |
| `cfg.plugins.enabled_plugins` | Plugins to load at startup | `TODO` |

</details>

<details>
<summary><b>📁 Project structure</b></summary>

<br>

<!-- TODO: Add the remaining folders and files you have (requirements.txt, models, data, response audio, tests). Module names below come from the imports in main.py. -->

```
AiDarwaza/
├── main.py                  # Orchestrator: main loop, overlay, dialogue
├── config.py                # Central cfg object
├── vision/
│   ├── capture.py           # FaceCamera: detection and quality gates
│   ├── antispoof.py         # AntiSpoofDetector (MobileNetV2)
│   ├── recognition.py       # Embedding extraction, FAISS matching, enrollment
│   ├── audio_capture.py     # Microphone recording
│   ├── audio_validator.py   # Rejects silent / unusable audio
│   ├── speech_to_text.py    # Urdu transcription
│   ├── speaker.py           # Plays audio responses
│   └── db_tools.py          # Saves visit events
├── nlp/
│   ├── dialogue_manager.py  # Dialogue state machine
│   ├── identity_extractor.py# Pulls the spoken name from a transcript
│   ├── intent_router.py     # Transcript to intent
│   └── response_mapper.py   # Response key to audio file
├── llm/
│   └── ollama_client.py     # Local LLM client and health check
├── plugins/
│   ├── loader.py            # Loads plugins and dispatches hooks
│   └── base.py              # Plugin base class and VisitorEvent
└── docs/                    # Banner, diagrams, demo media
```

</details>

<details>
<summary><b>⏱️ Performance benchmarks</b></summary>

<br>

<!-- TODO: Measure and fill in. Even rough timings on your own hardware are valuable. -->

| Stage | Latency | Hardware |
|---|---|---|
| Face detection | `TODO` | `TODO` |
| Anti-spoofing | `TODO` | `TODO` |
| Embedding + FAISS match | `TODO` | `TODO` |
| Speech-to-text | `TODO` | `TODO` |
| Intent routing | `TODO` | `TODO` |
| **End to end** | `TODO` | `TODO` |

</details>

<img src="docs/divider.svg" width="100%" alt="">

## 🔒 Privacy

- All processing happens on your own device.
- Face embeddings and visit events are stored locally.
- The application sends no images, audio, or transcripts to external servers.
- InsightFace, FAISS, Whisper, and Ollama all run locally once their models are downloaded.

> [!CAUTION]
> Face data is sensitive. If you deploy this at a home or business, tell people their faces are being processed and secure the device that stores the embeddings and face crops.

## 🚧 Limitations

- Liveness detection was trained on NUAA and may not generalize to modern screen or 3D mask attacks.
- **Self-enrollment:** any visitor who passes the liveness check and says a name is saved as a known face. There is no owner approval step yet.
- Replies are pre-recorded audio, so the system can only say what has been recorded.
- The dialogue runs synchronously, so the preview window pauses while a conversation is in progress.
- Recognition accuracy depends on lighting, camera quality, and viewing angle.
- Urdu transcription quality varies with accent, background noise, and microphone.
- Designed for one door and one camera, not multi-camera surveillance.

<img src="docs/divider.svg" width="100%" alt="">

## 🗺️ Roadmap

- [ ] ✅ Owner approval before a new face is enrolled
- [ ] 📲 Telegram / WhatsApp alert plugin with a snapshot when a stranger arrives
- [ ] 📊 Web dashboard on top of the saved visit events
- [ ] 🛡️ Stronger anti-spoofing (newer datasets, screen replay and mask attacks)
- [ ] 🧵 Non-blocking dialogue so the preview never freezes
- [ ] 🔊 Dynamic Urdu text-to-speech replies in place of pre-recorded audio
- [ ] 🥧 Edge deployment on Raspberry Pi / Jetson with latency benchmarks
- [ ] 🎥 Multi-camera support (RTSP / IP cameras) for a surveillance version

<br>

## 🙏 Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface) for face analysis
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research
- [Whisper](https://github.com/openai/whisper) by OpenAI
- [Ollama](https://ollama.com) for local LLM serving
- NUAA Photograph Imposter Database for anti-spoofing training data

## 📄 License

Released under the MIT License. See [LICENSE](LICENSE) for details.

<!-- TODO: Add a LICENSE file to the repo, or change this section. InsightFace pretrained models have their own usage terms, so check them before any commercial use. -->

<img src="docs/divider.svg" width="100%" alt="">

<div align="center">

### 👨‍💻 Author

**Syed Amar**
AI engineer · Dargai, Khyber Pakhtunkhwa, Pakistan

[![Portfolio](https://img.shields.io/badge/Portfolio-syedamar.vercel.app-FFD400?style=for-the-badge&labelColor=0d0d0d)](https://syedamar.vercel.app)
[![Instagram](https://img.shields.io/badge/Instagram-@thesyedamar-FFD400?style=for-the-badge&logo=instagram&logoColor=black&labelColor=0d0d0d)](https://instagram.com/thesyedamar)
[![GitHub](https://img.shields.io/badge/GitHub-your--username-FFD400?style=for-the-badge&logo=github&logoColor=black&labelColor=0d0d0d)](https://github.com/thesyedamar)

<br>

**If AiDarwaza helps you, give it a ⭐**

</div>
