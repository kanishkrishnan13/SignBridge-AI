import os
import hashlib
import warnings
import io


class TTSEngine:
    LANG_MAP = {
        'en': 'en',
        'ta': 'ta',
        'hi': 'hi',
    }

    def __init__(self, cache_dir='tts_cache'):
        self._cache_dir = self._resolve_path(cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)
        self._memory_cache: dict[str, bytes] = {}

    @staticmethod
    def _resolve_path(path):
        if os.path.isabs(path):
            return path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(backend_dir, path)

    @staticmethod
    def _cache_key(text: str, language: str) -> str:
        raw = f"{language}:{text}"
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def synthesize(self, text: str, language: str = 'en') -> bytes:
        """
        Convert *text* to speech audio.

        Returns the MP3 audio as raw bytes.
        Raises RuntimeError if gTTS is unavailable and no cached version exists.
        """
        return self.get_cached_or_generate(text, language)

    def get_cached_or_generate(self, text: str, language: str) -> bytes:
        """
        Return cached MP3 bytes for *text*+*language*, generating if needed.

        Cache hierarchy:
          1. In-memory dict (fastest)
          2. On-disk file in cache_dir
          3. Generate via gTTS, persist to both caches
        """
        lang_code = self.LANG_MAP.get(language, 'en')
        key = self._cache_key(text, lang_code)

        # 1 — memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]

        # 2 — disk cache
        cache_file = os.path.join(self._cache_dir, f"{key}.mp3")
        if os.path.isfile(cache_file):
            with open(cache_file, 'rb') as fh:
                data = fh.read()
            self._memory_cache[key] = data
            return data

        # 3 — generate
        data = self._generate(text, lang_code)

        # Persist to disk
        try:
            with open(cache_file, 'wb') as fh:
                fh.write(data)
        except OSError as exc:
            warnings.warn(f"Could not write TTS cache file: {exc}")

        self._memory_cache[key] = data
        return data

    @staticmethod
    def _generate(text: str, lang_code: str) -> bytes:
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise RuntimeError(
                "gTTS is not installed. Run `pip install gTTS`."
            ) from exc

        buf = io.BytesIO()
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
