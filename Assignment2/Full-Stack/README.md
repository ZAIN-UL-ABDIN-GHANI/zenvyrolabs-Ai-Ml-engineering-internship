# 🎙️ Zenvyrolabs Advanced Voice Studio

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![CUDA](https://img.shields.io/badge/GPU-CUDA-green)
![License](https://img.shields.io/badge/License-MIT-success)
![Tests](https://img.shields.io/badge/Tests-176_Passing-success)

</p>

---

# 🚀 Production Edition

An enterprise-grade AI Voice Studio capable of:

- AI Voice Cloning
- Multi-Speaker Podcast Generation
- Voice Training
- Speaker Verification
- Intelligent Pronunciation Routing
- Production Audio Processing
- GPU Accelerated Inference

The project has been transformed from an MVP into a modular production architecture with dependency injection, health monitoring, automated testing, and Docker deployment.

---

# ✨ Features

## 🎙️ Voice Cloning

- Zero-shot voice cloning
- Multiple speaker support
- High quality speech synthesis
- Voice preview
- Reference voice management

---

## 🎧 Podcast Studio

Generate podcasts using multiple cloned voices.

Supports:

- Multiple speakers
- Automatic speaker switching
- Crossfade stitching
- Background processing
- Long-form podcast generation

---

## 🌍 Pronunciation Routing

Automatically detects:

- Urdu
- Hindi
- English
- Mixed Urdu-English
- Mixed Hindi-English

Uses Microsoft Neural Voices before applying cloned voice conversion.

---

## 🎵 Audio Pipeline

Production audio processing includes

- Noise Reduction
- Silence Removal
- Audio Normalization
- Loudness Balancing
- Crossfade
- Smart Chunk Generation
- Invalid Audio Recovery

---

## 🤖 AI Processing

- F5-TTS
- Edge TTS
- Resemblyzer
- Transformers
- Speaker Verification
- Voice Similarity Analysis

---

## 🛡 Production Engineering

✔ Dependency Injection

✔ Structured Logging

✔ GPU Detection

✔ CPU Fallback

✔ Health Checks

✔ Validation

✔ Modular Architecture

✔ Error Recovery

✔ Docker Support

✔ Docker Compose

---

# 🧱 Project Architecture

```
Presentation Layer
        │
        ▼
Application Layer
        │
        ▼
Domain Layer
        │
        ▼
Storage Layer
        │
        ▼
Infrastructure Layer
        │
        ▼
AI Processing Layer
```

---

# 📁 Project Structure

```
voice-studio/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-rvc.txt
│
├── voice_studio/
│   ├── application/
│   ├── domain/
│   ├── infrastructure/
│   ├── storage/
│   ├── ai_processing/
│   ├── configuration/
│   └── gradio/
│
├── saved_voices/
├── rvc_models/
├── training_data/
├── logs/
├── temp/
├── hf_cache/
│
└── tests/
```

---

# 🛠 Technology Stack

### AI

- F5-TTS
- Edge-TTS
- Transformers
- Resemblyzer
- PyTorch

### Backend

- Python
- Gradio

### Audio

- FFmpeg
- Pydub
- SoundFile
- Noisereduce

### Deployment

- Docker
- Docker Compose
- NVIDIA CUDA

---

# 💻 System Requirements

## Minimum

- Python 3.10
- 8 GB RAM
- Windows / Linux

## Recommended

- NVIDIA GPU
- CUDA 12+
- 16 GB RAM

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/zenvyrolabs-Ai-Ml-engineering-internship.git

cd zenvyrolabs-Ai-Ml-engineering-internship/Assignment2/Full-Stack
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Requirements

```bash
pip install -U pip

pip install -r requirements.txt
```

---

# Install FFmpeg

Windows

Download FFmpeg

https://ffmpeg.org/download.html

Add FFmpeg to PATH.

Linux

```bash
sudo apt update

sudo apt install ffmpeg
```

Verify

```bash
ffmpeg -version
```

---

# Hugging Face Setup

Create a Read Token

https://huggingface.co/settings/tokens

Create a file named

```
.env
```

Add

```env
HF_TOKEN=hf_your_token_here
```

The application automatically uses this token for downloading AI models.

Never upload your personal token to GitHub.

```

---

**Next:** **Part 2** will include:
- Local execution
- Docker
- Docker Compose
- GPU setup
- Environment variables
- Running the application
- Health checks
- Docker image build and deployment
# ▶ Running the Application

## Start the Application

Activate the virtual environment.

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Run the application.

```bash
python app.py
```

The Gradio interface will be available at

```
http://127.0.0.1:7860
```

or

```
http://localhost:7860
```

---

# 🐳 Docker Deployment

The project includes a production-ready Docker configuration with:

- NVIDIA GPU support
- Persistent storage
- Health checks
- Non-root container
- Separate Python environments for the application and RVC
- Automatic Hugging Face cache persistence

---

# Docker Prerequisites

Install:

- Docker Desktop
- Docker Compose

Linux users with NVIDIA GPU should also install:

- NVIDIA Driver
- NVIDIA Container Toolkit

Verify Docker:

```bash
docker --version
docker compose version
```

Verify GPU (Linux):

```bash
nvidia-smi
```

---

# Environment Configuration

Create a `.env` file in the project root.

Example:

```env
HF_TOKEN=hf_your_token_here

VOICE_STUDIO_F5TTS_MODEL=F5TTS_Base
VOICE_STUDIO_NARRATION_VOICE=en-US-GuyNeural
VOICE_STUDIO_AUTO_ROUTE_PRONUNCIATION=true

HF_HOME=/app/hf_cache
MPLBACKEND=Agg
```

---

# Build Docker Image

Build the image:

```bash
docker compose build
```

Or rebuild from scratch:

```bash
docker compose build --no-cache
```

---

# Start the Container

```bash
docker compose up
```

Run in detached mode:

```bash
docker compose up -d
```

---

# Stop the Container

```bash
docker compose down
```

---

# Restart the Container

```bash
docker compose restart
```

---

# View Logs

```bash
docker compose logs
```

Follow logs continuously:

```bash
docker compose logs -f
```

---

# Check Running Containers

```bash
docker ps
```

---

# Access the Application

After startup, open:

```
http://localhost:7860
```

---

# Docker Volumes

The application stores data in persistent Docker volumes.

| Volume | Purpose |
|---------|---------|
| saved_voices | Saved voice profiles |
| training_data | Voice training datasets |
| rvc_models | RVC model storage |
| hf_cache | Hugging Face model cache |
| logs | Application logs |

These volumes persist even if the container is removed.

---

# GPU Support

The Docker image supports NVIDIA GPUs.

Check GPU availability:

```bash
docker exec -it voice-studio nvidia-smi
```

If the GPU is available, PyTorch will automatically use CUDA.

---

# CPU Mode

If no NVIDIA GPU is detected, the application automatically falls back to CPU mode.

No configuration changes are required.

---

# Health Check

The container includes a health check.

View container health:

```bash
docker ps
```

Example:

```
STATUS

Up 2 minutes (healthy)
```

---

# Updating the Project

Pull the latest changes:

```bash
git pull
```

Rebuild:

```bash
docker compose build
```

Restart:

```bash
docker compose up -d
```

---

# Cleaning Docker

Remove stopped containers:

```bash
docker container prune
```

Remove unused images:

```bash
docker image prune
```

Remove unused volumes:

```bash
docker volume prune
```

Remove everything unused:

```bash
docker system prune -a
```

---

# Running Without Docker

Clone the repository.

Create a virtual environment.

Install dependencies.

Install FFmpeg.

Create the `.env` file.

Run:

```bash
python app.py
```

---

# Project Configuration

Configuration is managed through environment variables.

| Variable | Description |
|-----------|-------------|
| HF_TOKEN | Hugging Face access token |
| HF_HOME | Hugging Face cache location |
| MPLBACKEND | Matplotlib backend |
| VOICE_STUDIO_F5TTS_MODEL | F5-TTS model |
| VOICE_STUDIO_NARRATION_VOICE | Default Edge TTS voice |
| VOICE_STUDIO_AUTO_ROUTE_PRONUNCIATION | Enable pronunciation routing |

---

# Production Deployment

Recommended production setup:

- Ubuntu 22.04
- NVIDIA Driver
- Docker Engine
- Docker Compose
- NVIDIA Container Toolkit
- CUDA 12+
- 16 GB RAM or higher
- Persistent Docker volumes
- Reverse proxy (Nginx or Traefik)
- HTTPS using Let's Encrypt

---

# Supported Platforms

✅ Windows 10/11

✅ Ubuntu 22.04+

✅ Docker Desktop

✅ NVIDIA CUDA

✅ CPU-only execution
# 🎙 User Guide

The application provides multiple AI-powered voice processing modules through a single Gradio interface.

---

# 🎤 AI Voice Cloning

Create a high-quality AI clone from a reference voice.

## Steps

1. Open **Voice Cloning**
2. Enter a speaker name.
3. Upload a clean voice sample.
4. Enter the transcript of the uploaded audio.
5. Click **Clone Voice**.
6. Wait for processing.
7. The cloned voice is automatically saved inside:

```
saved_voices/
```

---

## Best Practices

- 20–60 seconds reference audio
- Minimal background noise
- One speaker only
- Normal speaking speed
- WAV format recommended

---

# 🎙 Podcast Studio

Generate multi-speaker podcasts using cloned voices.

## Steps

1. Open **Podcast Studio**
2. Enter an episode title.
3. Write the podcast script.

Example

```
HAKEEM: Assalam o Alaikum!

ZAIN: Wa Alaikum Salam!
```

4. Select speakers.
5. Click **Generate Podcast**.

The application automatically:

- Generates speech
- Switches speakers
- Applies pronunciation routing
- Merges audio
- Applies crossfade
- Produces the final podcast

---

# 🌍 Pronunciation Routing

The routing engine automatically detects language before generating speech.

Supported languages

- English
- Urdu
- Hindi
- Urdu-English
- Hindi-English

Mixed-language sentences are automatically routed to the appropriate synthesis pipeline for improved pronunciation.

---

# 🎓 Voice Training

Train your own voice model.

## Steps

1. Open **Voice Training**
2. Upload training dataset.
3. Configure training parameters.
4. Start training.
5. Monitor logs.
6. Save the trained model.

Training outputs are stored in:

```
training_data/
```

---

# 🎧 Speaker Verification

The project uses **Resemblyzer** to compare speaker similarity.

Features

- Voice similarity scoring
- Speaker matching
- Identity verification
- Embedding generation

---

# 📊 Voice Quality Analysis

Quality analysis includes:

- Loudness
- Noise level
- Speech clarity
- Duration
- Similarity score
- Signal quality

---

# 🩺 System Health Dashboard

The Health Dashboard validates:

- Python environment
- FFmpeg installation
- CUDA availability
- GPU detection
- Hugging Face connectivity
- Required models
- Disk space
- Application dependencies

---

# 🧪 Testing

The project includes **176 automated tests**.

Run all tests

```bash
pytest
```

Generate coverage

```bash
pytest --cov
```

Run a specific test

```bash
pytest tests/test_voice_generation.py
```

---

# 📜 Logs

Application logs are stored in

```
logs/
```

Docker logs

```bash
docker compose logs -f
```

---

# ⚠ Troubleshooting

## Hugging Face Token Missing

Error

```
401 Unauthorized
```

Solution

1. Create a Hugging Face Read Token.
2. Create a `.env` file.
3. Add:

```env
HF_TOKEN=hf_your_token_here
```

Restart the application.

---

## FFmpeg Not Found

Error

```
ffmpeg not found
```

Install FFmpeg.

Windows

Download from

```
https://ffmpeg.org/download.html
```

Linux

```bash
sudo apt install ffmpeg
```

Verify

```bash
ffmpeg -version
```

---

## CUDA Not Detected

Run

```bash
nvidia-smi
```

If no GPU is detected, install the latest NVIDIA Driver and NVIDIA Container Toolkit.

The application will automatically fall back to CPU mode.

---

## Docker Build Failed

Rebuild from scratch

```bash
docker compose build --no-cache
```

---

## Model Download Failed

Check

- Internet connection
- HF_TOKEN
- Hugging Face availability

Clear cache if necessary

```
hf_cache/
```

---

## Port Already in Use

Change

```
7860
```

to another port in

```
docker-compose.yml
```

Example

```yaml
ports:
  - "8080:7860"
```

---

# 📈 Performance Tips

For best performance:

- Use an NVIDIA GPU
- Keep models in `hf_cache`
- Store voices in `saved_voices`
- Use SSD storage
- Allocate at least 16 GB RAM
- Keep Docker volumes persistent

---

# 🤝 Contributing

Contributions are welcome.

Workflow

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 🙏 Acknowledgements

This project builds upon several outstanding open-source projects.

- F5-TTS
- Edge-TTS
- PyTorch
- Hugging Face
- Transformers
- Gradio
- Resemblyzer
- FFmpeg
- Pydub
- Noisereduce

Special thanks to the creators and maintainers of these projects.

---

# ✅ Assignment Completion

| Task | Status |
|------|--------|
| Production Architecture | ✅ Completed |
| Docker Deployment | ✅ Completed |
| Voice Cloning | ✅ Completed |
| AI Voice Training | ✅ Completed |
| Podcast Generation | ✅ Completed |
| Pronunciation Routing | ✅ Completed |
| Speaker Verification | ✅ Completed |
| Audio Processing Pipeline | ✅ Completed |
| Dependency Injection | ✅ Completed |
| Health Monitoring | ✅ Completed |
| GPU Support | ✅ Completed |
| Docker Compose | ✅ Completed |
| Automated Testing (176 Tests) | ✅ Completed |
| Documentation | ✅ Completed |

---

# 👨‍💻 Author

**Zain ul Abdin Ghani**

BS Computer Science Graduate

AI • Machine Learning • Generative AI • LLM Applications • Voice AI • Full Stack Development

GitHub:
```
https://github.com/ZAIN-UL-ABDIN-GHANI
```

---

# ⭐ Support

If you found this project helpful:

⭐ Star the repository

🍴 Fork the project

🐛 Report issues

💡 Suggest new features

Thank you for exploring **Zenvyrolabs Advanced Voice Studio**.