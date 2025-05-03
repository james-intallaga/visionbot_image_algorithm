# BookBuddy (bilbyreadsv11)

An Interactive Reading Companion for Children

---

BookBuddy is an AI-powered, nurturing reading companion designed for children ages 2-9. Taking on the persona of "Auntie Bea," BookBuddy analyzes book pages via video feed, reads stories with enthusiasm, and interacts naturally with children. Built on LiveKit's agent framework, it leverages advanced speech, vision, and language models for a magical, interactive storytime experience.

## Key Features
- **Real-time book and page detection** using computer vision
- **Natural, child-friendly speech** (OpenAI GPT-4o Mini TTS, Sage voice)
- **Character and theme extraction** from stories
- **Context tracking** across reading sessions
- **Age-appropriate, warm interactions**
- **Event-driven, asynchronous architecture** for real-time responsiveness

## How It Works
- BookBuddy watches a live video feed of a picture book
- Detects when a new page appears (using perceptual hashing)
- Reads the story aloud, asks questions, and engages the child
- Tracks characters, themes, and previous discussions for continuity

## Requirements
- Python 3.9+
- [LiveKit Agents](https://github.com/livekit/agents)
- OpenAI API access (for GPT-4o, TTS, STT)
- See `requirements.txt` for all dependencies

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/james-intallaga/bilbyreadsv11.git
   cd bilbyreadsv11
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with required API keys (see `.env.example` if provided)

## Usage
Run the main agent script:
```bash
python bilbyreadsv11.py
```

BookBuddy will connect to a LiveKit room, process video/audio, and interact as "Auntie Bea."

## File Overview
- `bilbyreadsv11.py` — Main agent code
- `requirements.txt` — Python dependencies

## Author
James Intallaga

## License
[MIT](LICENSE) (or specify your license)

---

*For more details, see the code docstring and comments in `bilbyreadsv11.py`.* 