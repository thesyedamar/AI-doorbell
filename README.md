<div align="center"> <img src="docs/banner.svg" alt="AiDarwaza - AI smart doorbell" width="100%"> <br>

A local-first AI smart doorbell that recognizes faces, blocks spoofing attacks, and talks to visitors in Urdu.

<br>

Show Image Show Image Show Image Show Image Show Image Show Image Show Image

<!-- TODO: Confirm the anti-spoofing framework (TensorFlow vs PyTorch) and edit the badge above to match. --> <br>

Overview  •  How it works  •  Anti-spoofing  •  Quick start  •  Roadmap

</div> <br> <!-- TODO: Replace with your demo GIF. A 10-second clip of a spoof attempt being rejected is the most convincing thing you can show. --> <div align="center"> <img src="docs/demo.gif" alt="AiDarwaza demo" width="80%"> </div> <img src="docs/divider.svg" width="100%" alt="">
🎯 Overview

Most smart doorbells are expensive imports, depend on cloud subscriptions, and don't understand Urdu. AiDarwaza is built differently. Darwaza (دروازہ) means "door" in Urdu.

It identifies who is at the door and greets known people by name. It checks that the face is a real person and not a photo or a phone screen. When a stranger arrives, it listens in Urdu and replies using a local language model. Everything runs on your own machine, so face data never leaves the device.

[!NOTE] AiDarwaza watches one door and reacts when someone arrives. It is not a multi-camera CCTV or surveillance system. See the roadmap for that direction.

<br> <table> <tr> <td align="center" width="25%"> <h3>🧑 Recognizes</h3> InsightFace embeddings and FAISS search identify known people in real time. </td> <td align="center" width="25%"> <h3>🛡️ Verifies</h3> A custom MobileNetV2 rejects printed photos and screen replays before recognition runs. </td> <td align="center" width="25%"> <h3>🗣️ Understands Urdu</h3> Whisper transcribes what visitors say at the door. </td> <td align="center" width="25%"> <h3>🔒 Stays private</h3> No cloud, no subscription. Face data and audio stay on your device. </td> </tr> </table> <img src="docs/divider.svg" width="100%" alt="">
⚙️ How It Works
<div align="center"> <img src="docs/pipeline.svg" alt="AiDarwaza pipeline diagram" width="100%"> </div> <br>
Step	Stage	What happens
1	Capture	A frame is read from the camera.
2	Detect	A face is located in the frame.
3	Liveness check	The anti-spoofing model classifies the face as real or fake. Fakes are rejected and logged.
4	Embed	For real faces, InsightFace converts the face into a numeric embedding.
5	Match	FAISS finds the closest stored embedding. A similarity above the threshold means a known person.
6	Respond	Known people are greeted. Unknown visitors are heard by Whisper, and the local LLM generates a reply.

[!TIP] Why anti-spoofing runs before recognition: if recognition ran first, a printed photo of an enrolled person could trigger the "known person" path. Checking liveness first closes that hole.

What happens at the door
Situation	System response
🟢 Enrolled family member	Greets them by name
🟡 Unknown visitor	Listens in Urdu, transcribes, and replies through the LLM
🔴 Photo or phone screen held up	Rejected as a spoof and logged
<img src="docs/divider.svg" width="100%" alt="">
🧰 Tech Stack
Layer	Tool	Role
👤 Face analysis	InsightFace	Detection and face embeddings
⚡ Vector search	FAISS	Fast nearest-neighbor matching over known faces
🛡️ Liveness	Custom MobileNetV2	Real face vs. photo or screen attack
🎙️ Speech to text	Whisper	Urdu transcription
🧠 Reasoning	Ollama	Local LLM for replies and fallback
🐍 Language	Python	Glue for the whole pipeline
<!-- TODO: Add OpenCV, the TTS engine (if any), and any other libraries you actually use. --> <img src="docs/divider.svg" width="100%" alt="">
🛡️ Anti-Spoofing Model

The liveness detector is a custom-trained MobileNetV2 binary classifier (live face vs. spoof attack), trained on the NUAA Photograph Imposter Database.

<!-- TODO: Fill in every value below with your real numbers. Do not leave guesses. --> <table> <tr> <td width="50%" valign="top">

Training setup

Detail	Value
Backbone	MobileNetV2
Input size	TODO
Epochs	TODO
Augmentation	TODO
Train / val / test split	TODO
</td> <td width="50%" valign="top">

Results (held-out test set)

Metric	Value
Accuracy	TODO
FAR (false acceptance)	TODO
FRR (false rejection)	TODO
</td> </tr> </table>

[!WARNING] NUAA is an older dataset that mainly contains printed-photo attacks. Performance against high-resolution screen replays, video replays, and 3D masks is not established unless you test it yourself. Treat this model as a strong first layer, not a guarantee.

<img src="docs/divider.svg" width="100%" alt="">
🚀 Quick Start
Prerequisites
Python 3.10 or newer
A webcam and a microphone
Ollama installed
A GPU is optional but speeds things up
Install
bash
# 1. Clone the repository
git clone https://github.com/<your-username>/AiDarwaza.git
cd AiDarwaza

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the local LLM
ollama pull <model-name>
<!-- TODO: Replace <your-username> and <model-name> with your real values. -->

[!IMPORTANT] The trained anti-spoofing weights are required to run the system. Place the file at models/<your-weights-file>.

<!-- TODO: If the weights are too large for GitHub, host them on Hugging Face or Google Drive and add the download link here. -->
Run
bash
# Enroll a known person
python enroll.py --name "Ahmed" --images path/to/ahmed_photos/

# Start the doorbell
python main.py
<!-- TODO: Replace enroll.py / main.py and the flags with your real script names. --> <br> <details> <summary><b>⚙️ Configuration</b></summary> <br>
Setting	Purpose	Default
Recognition threshold	Minimum similarity to count as a known person	TODO
Spoof threshold	Minimum live probability to pass the liveness check	TODO
Whisper model size	Accuracy vs. speed trade-off for Urdu transcription	TODO
Ollama model	Which local LLM generates replies	TODO
Camera index	Which camera device to use	0
<!-- TODO: Point this to your real config file (config.py, config.yaml or .env). --> </details> <details> <summary><b>📁 Project structure</b></summary> <br> <!-- TODO: Replace with your real folder layout. This is only a template. -->
AiDarwaza/
├── main.py                # Entry point
├── enroll.py              # Add known faces
├── requirements.txt
├── models/                # Anti-spoofing weights
├── data/                  # Stored embeddings and FAISS index
├── src/
│   ├── detection.py
│   ├── antispoof.py
│   ├── recognition.py
│   ├── speech.py
│   └── llm.py
├── training/              # Anti-spoofing training notebook / scripts
├── docs/                  # Banner, diagrams, demo media
└── README.md
</details> <details> <summary><b>⏱️ Performance benchmarks</b></summary> <br> <!-- TODO: Measure and fill in. Even rough timings on your own hardware are valuable. -->
Stage	Latency	Hardware
Face detection	TODO	TODO
Anti-spoofing	TODO	TODO
Embedding + FAISS match	TODO	TODO
Whisper transcription	TODO	TODO
LLM response	TODO	TODO
End to end	TODO	TODO
</details> <img src="docs/divider.svg" width="100%" alt="">
🔒 Privacy
All processing happens on your own device.
Face embeddings are stored locally in the FAISS index.
The application sends no images, audio, or transcripts to external servers.
InsightFace, FAISS, Whisper, and Ollama all run locally once their models are downloaded.

[!CAUTION] Face data is sensitive. If you deploy this at a home or business, tell people their faces are being processed and secure the device that stores the embeddings.

🚧 Limitations
Liveness detection was trained on NUAA and may not generalize to modern screen or 3D mask attacks.
Recognition accuracy depends on lighting, camera quality, and viewing angle.
Urdu transcription quality varies with accent, background noise, and microphone.
Designed for one door and one camera, not multi-camera surveillance.
LLM replies are only as reliable as the local model you choose.
<img src="docs/divider.svg" width="100%" alt="">
🗺️ Roadmap
 📲 Telegram / WhatsApp alert with a snapshot when a stranger arrives
 📊 Visitor event log and web dashboard
 🛡️ Stronger anti-spoofing (newer datasets, screen replay and mask attacks)
 🥧 Edge deployment on Raspberry Pi / Jetson with latency benchmarks
 🎥 Multi-camera support (RTSP / IP cameras) for a surveillance version
 🗂️ Recording with searchable history by person and time
 🔊 Urdu voice replies (text-to-speech)
<br>
🙏 Acknowledgements
InsightFace for face analysis
FAISS by Meta AI Research
Whisper by OpenAI
Ollama for local LLM serving
NUAA Photograph Imposter Database for anti-spoofing training data
📄 License

Released under the MIT License. See LICENSE for details.

<!-- TODO: Add a LICENSE file to the repo, or change this section. InsightFace pretrained models have their own usage terms, so check them before any commercial use. --> <img src="docs/divider.svg" width="100%" alt=""> <div align="center">
👨‍💻 Author

Syed Amar AI engineer · Dargai, Khyber Pakhtunkhwa, Pakistan

Portfolio Instagram GitHub

<br>

If AiDarwaza helps you, give it a ⭐

</div>
