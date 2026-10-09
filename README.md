# EDITH AI

**Created by:** Vighnesh Mestri

A personal AI assistant powered by a locally running language model through Ollama.

EDITH is designed to provide a conversational assistant experience with terminal and voice interfaces, memory, reminders, and system tools.

## Features

- Local AI responses using Ollama.
- Terminal-based conversation.
- Voice interface with speech recognition and text-to-speech.
- Context and memory management.
- Reminder scheduling and automation.
- File operations and system commands.
- Web-search tooling.
- SQLite-based local data storage.
- Configurable application settings and logging.

## Technology Stack

- **Language:** Python
- **Local AI:** Ollama
- **Speech Recognition:** Faster-Whisper and supporting audio libraries
- **Text-to-Speech:** Edge-TTS with a fallback engine
- **Database:** SQLite
- **Automation:** Python scheduling and trigger modules

## Requirements

- Windows or another compatible Python environment.
- Python installed.
- Ollama installed and running.
- A language model downloaded through Ollama.
- A microphone and speakers for voice interaction.

Install additional Python dependencies from `requirements.txt`.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create your local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set `OLLAMA_MODEL` to a model name available on your computer. Check installed models with:

```powershell
ollama list
```

Make sure Ollama is running and the configured model is available before launching EDITH.

### 5. Run EDITH

Start the terminal application:

```powershell
python main.py
```

Use the voice interface entry point configured in the project when you want to interact by voice.

## Project Structure

```text
EDITH AI/
├── audio/          # Speech recognition and text-to-speech
├── automation/     # Scheduler and automation triggers
├── brain/          # Ollama client and conversation context
├── config/         # Application settings and prompts
├── core/           # Main assistant orchestration and commands
├── database/       # Database and memory handling
├── logs/           # Local runtime logs
├── tools/          # Assistant tools
├── ui/             # Terminal, voice, and related interfaces
├── utils/          # Shared utilities and logging
├── main.py         # Terminal application entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Privacy

EDITH uses Ollama for local language-model inference. Local databases, logs, environment files, and other runtime data may contain private information.

Keep `.env`, database files, logs, virtual environments, and personal recordings out of version control. Some optional tools may require an internet connection.

## Development Status

This project is developed for personal use. Features and setup requirements may change as development continues.

## License

No license has been selected yet. All rights are reserved unless a license is added to this repository.
