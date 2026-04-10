# SignBridge AI — API Documentation

Base URL (development): `http://localhost:5000`

All request and response bodies use `application/json` unless noted.

---

## Table of Contents

1. [POST /api/detect](#post-apidetect)
2. [POST /api/speech](#post-apispeech)
3. [POST /api/mapper](#post-apimapper)
4. [GET /api/avatar/sequence/\<sign\>](#get-apiavatarsequencesign)
5. [POST /api/calibrate/start](#post-apicalibratestart)
6. [POST /api/calibrate/frame](#post-apicalibrateframe)
7. [POST /api/calibrate/complete](#post-apicalibratecomplete)
8. [GET /api/health](#get-apihealth)
9. [Error Codes](#error-codes)

---

## POST /api/detect

Detect a hand gesture from a single video frame.

### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `frame` | `string` | ✅ | Base64-encoded JPEG or PNG image |
| `calibration_offset` | `object` | ❌ | Per-session calibration data (see below) |

```json
{
  "frame": "<base64-encoded JPEG>",
  "calibration_offset": {
    "offset": [0.02, -0.01, 0.0],
    "scale": 0.98
  }
}
```

### Response `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `label` | `string \| null` | Predicted ISL sign token, e.g. `"PAIN"` |
| `confidence` | `float` | Prediction confidence `[0.0, 1.0]` |
| `landmarks` | `array` | 21 × `[x, y, z]` raw landmark coordinates |
| `is_signing` | `bool` | `true` when confidence ≥ 0.6 and a label is present |

```json
{
  "label": "PAIN",
  "confidence": 0.91,
  "landmarks": [
    [0.51, 0.83, 0.00],
    [0.54, 0.76, -0.02],
    "... (21 entries total)"
  ],
  "is_signing": true
}
```

When no hand is detected:

```json
{
  "label": null,
  "confidence": 0.0,
  "landmarks": [],
  "is_signing": false
}
```

### Error Responses

| HTTP | `error` | Cause |
|------|---------|-------|
| `400` | `Missing 'frame' field` | `frame` key absent in request body |
| `400` | `Invalid frame data: could not decode image` | Base64 is corrupt or not a valid image |

---

## POST /api/speech

Synthesise speech from text using gTTS.

### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `string` | ✅ | Text to speak |
| `lang` | `string` | ❌ | Language code (default `"en"`). Supported: `en`, `ta`, `hi` |

```json
{
  "text": "You have a fever. Please rest.",
  "lang": "en"
}
```

### Response `200 OK`

Binary `audio/mpeg` stream (MP3 audio).

```
Content-Type: audio/mpeg
Content-Length: 18432
<binary MP3 data>
```

Play in the browser:

```javascript
const resp = await fetch('/api/speech', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Hello', lang: 'en' })
});
const blob = await resp.blob();
const url = URL.createObjectURL(blob);
new Audio(url).play();
```

### Error Responses

| HTTP | `error` | Cause |
|------|---------|-------|
| `400` | `Missing or empty 'text' field` | `text` absent or blank |
| `400` | `Unsupported language: <lang>` | Language code not in `{en, ta, hi}` |
| `500` | `Speech synthesis failed` | gTTS network error or internal failure |

---

## POST /api/mapper

Convert natural-language medical text to ISL sign tokens and retrieve keypoint sequences.

### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `string` | ✅ | Plain-text medical sentence |

```json
{
  "text": "I have chest pain and I feel dizzy"
}
```

### Response `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `signs` | `string[]` | Ordered list of ISL sign tokens |
| `sequences` | `(object \| null)[]` | Keypoint JSON for each sign, or `null` if not found |

```json
{
  "signs": ["CHEST_PAIN", "DIZZY"],
  "sequences": [
    {
      "sign": "CHEST_PAIN",
      "frames": [
        { "landmarks": [[0.5, 0.8, 0.0], "..."] },
        "..."
      ]
    },
    null
  ]
}
```

### Error Responses

| HTTP | `error` | Cause |
|------|---------|-------|
| `400` | `Missing or empty 'text' field` | `text` absent or blank |

---

## GET /api/avatar/sequence/\<sign\>

Return the keypoint animation sequence for a single ISL sign token.

### Path Parameter

| Parameter | Type | Description |
|-----------|------|-------------|
| `sign` | `string` | ISL sign token, e.g. `PAIN`, `HELP`, `FEVER` |

### Response `200 OK`

Returns a JSON object read from `frontend/assets/keypoints/<sign>.json`.

```json
{
  "sign": "PAIN",
  "fps": 30,
  "frames": [
    {
      "landmarks": [
        [0.51, 0.82, 0.00],
        [0.54, 0.76, -0.02],
        "... (21 entries)"
      ]
    },
    "..."
  ]
}
```

### Error Responses

| HTTP | `error` | Cause |
|------|---------|-------|
| `404` | `Keypoint file not found for sign: <sign>` | No JSON file exists for the requested sign |

---

## POST /api/calibrate/start

Initialise a new calibration session. Must be called before submitting calibration frames.

### Request

No body required.

### Response `200 OK`

```json
{
  "status": "started",
  "message": "Calibration session started. Submit neutral-pose frames."
}
```

---

## POST /api/calibrate/frame

Submit one landmark frame captured during the calibration neutral pose.

### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `landmarks` | `array` | ✅ | 21 × `[x, y, z]` landmark array (raw output from `/api/detect`) |

```json
{
  "landmarks": [
    [0.50, 0.81, 0.00],
    [0.53, 0.75, -0.01],
    "... (21 entries total)"
  ]
}
```

### Response `200 OK`

```json
{
  "status": "ok",
  "frames_collected": 12,
  "message": "Frame accepted."
}
```

### Error Responses

| HTTP | `error` | Cause |
|------|---------|-------|
| `400` | `Missing 'landmarks' field` | `landmarks` key absent |
| `400` | `Calibration session not started` | `/api/calibrate/start` not called first |

---

## POST /api/calibrate/complete

Finalise calibration. Computes the per-user offset and scale from collected frames.

### Request

No body required.

### Response `200 OK`

```json
{
  "status": "calibrated",
  "offset": [0.012, -0.008, 0.001],
  "scale": 0.97,
  "frames_used": 30
}
```

### Error Responses

| HTTP | `error` | Cause |
|------|---------|-------|
| `400` | `Not enough calibration frames` | Fewer than the minimum frames collected |
| `400` | `Calibration session not started` | `/api/calibrate/start` not called first |

---

## GET /api/health

Liveness and readiness probe for monitoring / load balancers.

### Response `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Always `"ok"` |
| `model_loaded` | `bool` | `true` if `hand_landmark.tflite` is loaded |

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Error Codes

All error responses share the same JSON envelope:

```json
{
  "error": "<human-readable description>"
}
```

### HTTP Status Summary

| Code | Meaning | Common causes |
|------|---------|---------------|
| `200` | Success | — |
| `400` | Bad Request | Missing or invalid fields in the request body |
| `403` | Forbidden | Path traversal attempt on static file route |
| `404` | Not Found | Requested resource (keypoint file, static asset) does not exist |
| `500` | Internal Server Error | gTTS network failure, model inference crash |

### Sign Token Reference

The NLP mapper recognises the following tokens (partial list):

| Surface text | ISL token |
|--------------|-----------|
| `pain` | `PAIN` |
| `chest pain` | `CHEST_PAIN` |
| `head` / `headache` | `HEAD` / `HEADACHE` |
| `fever` | `FEVER` |
| `dizzy` | `DIZZY` |
| `help` | `HELP` |
| `emergency` | `EMERGENCY` |
| `doctor` | `DOCTOR` |
| `water` | `WATER` |
| `yes` / `no` | `YES` / `NO` |
| `thank you` | `THANK_YOU` |

Unknown words are spelled letter-by-letter as `LETTER_A` … `LETTER_Z`.
