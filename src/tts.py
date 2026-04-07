"""Kokoro TTS via mlx-audio (Apple Silicon GPU)."""

import numpy as np


class TTSBackend:
    """MLX-based Kokoro TTS backend for Apple Silicon."""

    sample_rate: int

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        # Warmup: triggers pipeline init (phonemizer, spacy, etc.)
        list(self._model.generate(text="Hello", voice="af_heart", speed=1.0))

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        results = list(self._model.generate(text=text, voice=voice, speed=speed))
        return np.concatenate([np.array(r.audio) for r in results])


def load() -> TTSBackend:
    """Load the TTS backend."""
    backend = TTSBackend()
    print(f"TTS: mlx-audio (Apple GPU, sample_rate={backend.sample_rate})")
    return backend
