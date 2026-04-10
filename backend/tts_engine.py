"""
Text-to-speech engine backed by gTTS with file-based caching.
"""

import hashlib
import io
import logging
import os

logger = logging.getLogger(__name__)

_CACHE_DIR = "/tmp/tts_cache"

_LANG_MAP = {
    "en": "en",
    "ta": "ta",
    "hi": "hi",
}


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_key(text: str, lang: str) -> str:
    raw = f"{lang}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_path(text: str, lang: str = "en") -> str:
    """Return the filesystem path for the cached audio file (may not exist yet)."""
    _ensure_cache_dir()
    key = _cache_key(text, lang)
    return os.path.join(_CACHE_DIR, f"{key}.mp3")


def generate(text: str, lang: str = "en") -> bytes:
    """
    Generate MP3 audio for *text* in *lang*.

    Checks the on-disk cache first; generates via gTTS on a cache miss.

    Parameters
    ----------
    text : str   The text to synthesise.
    lang : str   Language code — one of 'en', 'ta', 'hi'.

    Returns
    -------
    bytes   Raw MP3 audio data.

    Raises
    ------
    ValueError   If an unsupported language code is supplied.
    RuntimeError If gTTS fails to produce audio.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    gtts_lang = _LANG_MAP.get(lang)
    if gtts_lang is None:
        raise ValueError(
            f"Unsupported language '{lang}'. Supported: {list(_LANG_MAP.keys())}"
        )

    cache_path = get_cached_path(text, lang)

    if os.path.exists(cache_path):
        logger.debug("TTS cache hit: %s", cache_path)
        with open(cache_path, "rb") as fh:
            return fh.read()

    try:
        from gtts import gTTS
    except ImportError as exc:
        raise RuntimeError(
            "gTTS is not installed. Add 'gTTS' to requirements.txt."
        ) from exc

    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        audio_bytes = buf.getvalue()
    except Exception as exc:
        raise RuntimeError(f"gTTS synthesis failed: {exc}") from exc

    # Persist to cache
    _ensure_cache_dir()
    try:
        with open(cache_path, "wb") as fh:
            fh.write(audio_bytes)
        logger.debug("TTS cached to: %s", cache_path)
    except OSError as exc:
        logger.warning("Could not write TTS cache file: %s", exc)

    return audio_bytes
