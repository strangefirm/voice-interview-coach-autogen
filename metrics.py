"""Latency and cost tracking for one interview session."""
import io
import time
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field

# Estimated USD prices. Check platform.openai.com/docs/pricing and edit if they change.
PRICES = {
    "input_per_1m_tokens": 0.15,  # gpt-4o-mini
    "output_per_1m_tokens": 0.60,  # gpt-4o-mini
    "tts_per_1m_chars": 15.00,  # tts-1
    "stt_per_minute": 0.006,  # whisper-1
}


@dataclass
class Usage:
    """Filled in by interview.py while it runs."""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Call:
    name: str
    seconds: float
    cost: float


@dataclass
class Tracker:
    calls: list[Call] = field(default_factory=list)

    @contextmanager
    def track(self, name: str, usage: Usage | None = None, chars: int = 0, audio=None):
        """Time a block. Pass the Usage that the block fills, or chars (TTS) or audio (STT)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - start
            cost = 0.0
            if usage:
                cost += usage.prompt_tokens * PRICES["input_per_1m_tokens"] / 1e6
                cost += usage.completion_tokens * PRICES["output_per_1m_tokens"] / 1e6
            cost += chars * PRICES["tts_per_1m_chars"] / 1e6
            if audio is not None:
                cost += audio_minutes(audio) * PRICES["stt_per_minute"]
            self.calls.append(Call(name, seconds, cost))

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls)

    @property
    def total_seconds(self) -> float:
        return sum(c.seconds for c in self.calls)

    def summary(self) -> dict[str, tuple[int, float, float]]:
        """name -> (number of calls, average seconds, total cost)."""
        out = {}
        for name in dict.fromkeys(c.name for c in self.calls):
            group = [c for c in self.calls if c.name == name]
            out[name] = (len(group), sum(c.seconds for c in group) / len(group), sum(c.cost for c in group))
        return out


def audio_minutes(audio) -> float:
    """Length of a recorded WAV file in minutes."""
    try:
        with wave.open(io.BytesIO(audio.getvalue())) as wav:
            return wav.getnframes() / wav.getframerate() / 60
    except Exception:
        return 0.0
