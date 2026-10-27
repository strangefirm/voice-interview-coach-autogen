# 🎤 AI Interview Coach

A voice-enabled mock interview app built with **Microsoft AutoGen** and **Streamlit**. Pick a role and seniority level, then answer questions by typing or speaking. After each answer you get a score, specific feedback, and a model answer. At the end you get a full report with a hiring verdict and a study plan.

## Features

- **Multi-agent interviews**: three AutoGen agents, each with one job:
  - **Interviewer** asks one question at a time, building on earlier answers and raising the difficulty.
  - **Evaluator** scores the latest answer (1-10) and returns structured output: strengths, an actionable fix, and a model answer.
  - **Reporter** writes the final Markdown report: verdict, average score, strengths, areas to improve, and a 3-step study plan.
- **Voice mode**: questions are read aloud (OpenAI TTS) and you can answer with your microphone (Whisper speech-to-text).
- **Guardrails**: a local prompt-injection filter plus OpenAI's moderation endpoint on every answer. Candidate text is also wrapped in `<answer>` tags and the agents are told to treat it as data.
- **Cost and latency tracking**: live per-step latency and estimated cost shown in the sidebar.
- **Live eval score**: running average score across the session.
- **Downloadable report** as Markdown.

## How it works

```
Candidate answer
      │
      ▼
 guardrails.check_input ──(rejected)──► warning shown, nothing sent to agents
      │
      ▼
 AutoGen RoundRobinGroupChat
   evaluator ──► Evaluation (score, strengths, improvement, model_answer)
      │
      ▼
   interviewer ──► next question        (or reporter ──► final report on the last question)
```

| File | Purpose |
| --- | --- |
| [app.py](app.py) | Streamlit UI and session flow |
| [interview.py](interview.py) | AutoGen agents, prompts, and the evaluate-then-ask loop |
| [voice.py](voice.py) | Text-to-speech and speech-to-text helpers |
| [guardrails.py](guardrails.py) | Injection filter and moderation check |
| [metrics.py](metrics.py) | Latency and cost tracker |
| [tests/](tests/) | Unit tests and live evals |
| [pitch/](pitch/) | Pitch deck (HTML and PDF) |

## Getting started

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Install

```bash
git clone <your-repo-url>
cd voice-interview-coach-autogen

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...

# Optional overrides (defaults shown)
OPENAI_MODEL=gpt-4o-mini
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy
OPENAI_STT_MODEL=whisper-1
```

### Run

```bash
streamlit run app.py
```

Then choose a role, level, and number of questions (3-8) in the sidebar and click **Start interview**.

## Testing

Unit tests run offline and are free:

```bash
pytest
```

Live evals call the real OpenAI API (a few cents per run) and check things like score ranges, resistance to prompt injection, single-question output, and report structure:

```bash
pytest tests/test_evals.py -m live
```

## Cost

The sidebar shows an estimate based on the prices in [metrics.py](metrics.py) (defaults assume `gpt-4o-mini`, `tts-1`, and `whisper-1`). If OpenAI pricing changes, edit the `PRICES` dict.

## Tech stack

- [AutoGen](https://microsoft.github.io/autogen/) (`autogen-agentchat`, `autogen-ext`)
- [Streamlit](https://streamlit.io/)
- OpenAI API (chat, TTS, Whisper, moderation)
- Pydantic, python-dotenv, pytest
