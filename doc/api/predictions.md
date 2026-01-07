# Predictions API

## Overview

The Predictions API provides endpoints for battery State of Health (SoH) and Remaining Useful Life (RUL) prediction using trained models.

## Endpoints

### Single Prediction

```http
POST /api/v1/predictions
```

Predict SoH and RUL for a single battery dataset.

**Request Body:**

```json
{
  "algorithm": "bilstm",
  "model_id": 7,
  "battery_id": 1
}
```

**Alternative (Direct Data Input):**

```json
{
  "algorithm": "bilstm",
  "model_id": 7,
  "cycle_data": {
    "voltage": [3.5, 3.6, 3.7, 3.8],
    "current": [1.0, 1.0, 1.0, 1.0],
    "temperature": [25.0, 25.5, 26.0, 26.5]
  }
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `algorithm` | string | Yes | Algorithm: `baseline`, `bilstm`, `deephpm` |
| `model_id` | integer | No | Specific model ID (uses latest if omitted) |
| `battery_id` | integer | No* | Reference existing battery data |
| `cycle_data` | object | No* | Or provide cycle data directly |

*Either `battery_id` or `cycle_data` must be provided.

**Response (200 OK):**

```json
{
  "id": 99,
  "algorithm": "bilstm",
  "model_id": 7,
  "result": {
    "soh": 0.87,
    "rul": 120.5,
    "confidence": 0.91
  },
  "created_at": "2025-01-07T13:00:00Z"
}
```

---

### Batch Prediction

```http
POST /api/v1/predictions/batch
```

Predict for multiple batteries in a single request.

**Request Body:**

```json
{
  "requests": [
    {
      "algorithm": "bilstm",
      "battery_id": 1
    },
    {
      "algorithm": "deephpm",
      "battery_id": 2
    }
  ]
}
```

**Response (200 OK):**

```json
{
  "results": [
    {
      "id": 100,
      "algorithm": "bilstm",
      "model_id": 7,
      "result": {
        "soh": 0.87,
        "rul": 120.5,
        "confidence": 0.91
      },
      "created_at": "2025-01-07T13:00:00Z"
    },
    {
      "id": 101,
      "algorithm": "deephpm",
      "model_id": 8,
      "result": {
        "soh": 0.85,
        "rul": 115.0,
        "confidence": 0.89
      },
      "created_at": "2025-01-07T13:00:01Z"
    }
  ]
}
```

---

### List Prediction History

```http
GET /api/v1/predictions
```

Retrieve historical predictions (optional feature).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page |

**Response (200 OK):**

```json
{
  "items": [
    {
      "id": 99,
      "algorithm": "bilstm",
      "model_id": 7,
      "result": {
        "soh": 0.87,
        "rul": 120.5,
        "confidence": 0.91
      },
      "created_at": "2025-01-07T13:00:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "page_size": 20,
    "total": 150,
    "total_pages": 8
  }
}
```

## Data Models

### Prediction Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `algorithm` | string | Yes | `baseline`, `bilstm`, or `deephpm` |
| `model_id` | integer | No | Specific model (default: latest) |
| `battery_id` | integer | Conditional | Existing battery reference |
| `cycle_data` | object | Conditional | Direct cycle data input |

### Cycle Data Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `voltage` | array[float] | Yes | Voltage measurements |
| `current` | array[float] | Yes | Current measurements |
| `temperature` | array[float] | No | Temperature measurements |

### Prediction Result

| Field | Type | Description |
|-------|------|-------------|
| `soh` | float | State of Health (0-1 scale) |
| `rul` | float | Remaining Useful Life (cycles) |
| `confidence` | float | Prediction confidence score (0-1) |

## Algorithm Comparison

| Algorithm | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| **Baseline** | Fast | Good | Quick estimates, baseline comparison |
| **BiLSTM** | Medium | Better | Time-series analysis, sequential patterns |
| **DeepHPM** | Slower | Best | Physics-informed predictions, high accuracy |

## Prediction Workflow

```
1. Select algorithm
2. Choose model (or use latest)
3. Provide battery data (ID or direct input)
4. Receive SoH/RUL prediction
5. (Optional) Store prediction for audit trail
```

## Error Handling

**Missing Data (400):**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Either battery_id or cycle_data must be provided",
    "details": [
      {
        "field": "battery_id",
        "issue": "missing required field"
      }
    ],
    "trace_id": "req_abc123"
  }
}
```

**Model Not Found (404):**

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Model with ID 999 not found",
    "trace_id": "req_xyz789"
  }
}
```

**Invalid Input Data (422):**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid cycle data format",
    "details": [
      {
        "field": "cycle_data.voltage",
        "issue": "array length must match current array"
      }
    ],
    "trace_id": "req_def456"
  }
}
```

## Performance Notes

- Single predictions: ~100-500ms (depending on algorithm)
- Batch predictions: Processed sequentially
- Maximum batch size: 50 requests
- Predictions are cached for 1 hour (same input = same result)

## Example Usage

### Python

```python
import requests

response = requests.post(
    'http://localhost:8000/api/v1/predictions',
    json={
        'algorithm': 'bilstm',
        'battery_id': 1
    }
)

result = response.json()
print(f"SoH: {result['result']['soh']}")
print(f"RUL: {result['result']['rul']} cycles")
```

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "bilstm",
    "battery_id": 1
  }'
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/api/v1/predictions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    algorithm: 'bilstm',
    battery_id: 1
  })
});

const result = await response.json();
console.log(`SoH: ${result.result.soh}`);
console.log(`RUL: ${result.result.rul} cycles`);
```

## Notes

- Predictions use the latest trained model by default
- Confidence scores are algorithm-specific
- Historical predictions are stored for audit purposes
- Real-time predictions do not require database storage
