import os
import hashlib
import logging

logger = logging.getLogger(__name__)

LANG_MAP = {
    'en': 'en',
    'ta': 'ta',
    'hi': 'hi',
    'english': 'en',
    'tamil': 'ta',
    'hindi': 'hi',
}


class TTSEngine:
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '..', 'tts_cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"TTS Engine initialized, cache: {self.cache_dir}")

    def synthesize(self, text, language='en'):
        """Generate speech audio for the given text and language code.

        Returns the path to an MP3 file, or None on unrecoverable failure.
        """
        lang_code = LANG_MAP.get(language.lower(), 'en')

        cache_key = hashlib.md5(f"{text}_{lang_code}".encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.mp3")

        if os.path.exists(cache_path):
            logger.info(f"TTS cache hit: {cache_key}")
            return cache_path

        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(cache_path)
            logger.info(f"TTS generated: {text[:50]!r} [{lang_code}]")
            return cache_path
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            return self._create_fallback_audio(cache_path)

    def _create_fallback_audio(self, output_path):
        """Write a minimal silent MP3 as a fallback when gTTS is unavailable."""
        try:
            # Minimal silent MPEG Layer 3 frame repeated to fill ~1 second
            silence_frame = bytes([
                0xFF, 0xFB, 0x90, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ])
            with open(output_path, 'wb') as f:
                f.write(silence_frame * 100)
            return output_path
        except Exception as e:
            logger.error(f"Fallback audio creation failed: {e}")
            return None

    def get_supported_languages(self):
        return [
            {'code': 'en', 'name': 'English', 'gtts_code': 'en'},
            {'code': 'ta', 'name': 'Tamil', 'gtts_code': 'ta'},
            {'code': 'hi', 'name': 'Hindi', 'gtts_code': 'hi'},
        ]
