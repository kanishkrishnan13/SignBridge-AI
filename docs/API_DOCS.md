# SignBridge AI — API Reference

Base URL: `http://localhost:5000`

All request and response bodies are JSON. CORS is enabled for all origins.

---

## GET /api/health

Health check and system status.

**Response 200:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "mediapipe_ready": true,
  "version": "1.0.0"
}
```

---

## POST /api/detect

Detect an ISL sign from a single camera frame.

**Request:**
```json
{
  "frame": "<base64-encoded JPEG or PNG>"
}
```

**Response 200:**
```json
{
  "label": "help",
  "confidence": 0.92,
  "landmarks": [
    [0.51, 0.72, 0.001],
    ...
  ],
  "top_predictions": [
    {"label": "help", "confidence": 0.92},
    {"label": "stop", "confidence": 0.05},
    {"label": "yes",  "confidence": 0.02}
  ]
}
```

**Response when no hand detected:**
```json
{
  "label": "unknown",
  "confidence": 0.0,
  "landmarks": [],
  "top_predictions": []
}
```

**Response 400:**
```json
{"error": "No frame provided"}
```

---

## POST /api/mapper

Map a natural language sentence to a sequence of ISL sign tokens.

**Request:**
```json
{
  "text": "I have a headache and chest pain"
}
```

**Response 200:**
```json
{
  "signs": ["headache", "chest_pain"],
  "original_text": "I have a headache and chest pain"
}
```

---

## POST /api/speech

Convert text to speech audio (gTTS).

**Request:**
```json
{
  "text": "help",
  "language": "en"
}
```

Supported `language` values: `"en"`, `"ta"`, `"hi"`

**Response 200:**
```json
{
  "audio": "<base64-encoded MP3>",
  "text": "help",
  "language": "en"
}
```

**Response on TTS error:**
```json
{
  "error": "TTS failed",
  "audio": null
}
```

---

## POST /api/calibrate

Submit calibration sample data.

**Request:**
```json
{
  "landmarks": [[x, y, z], ...],
  "label": "help",
  "session_id": "abc123"
}
```

**Response 200:**
```json
{
  "status": "calibrated",
  "samples_received": 1,
  "session_id": "abc123"
}
```

---

## Error Codes

| HTTP | Meaning |
|------|---------|
| 400 | Bad request — missing or invalid input |
| 500 | Internal server error |

All errors include an `"error"` string field in the JSON response.
