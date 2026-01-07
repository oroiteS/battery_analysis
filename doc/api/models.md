# Models API

## Overview

The Models API manages trained model artifacts. Models are automatically created when training jobs complete successfully.

## Endpoints

### List Models

```http
GET /api/v1/models
```

Retrieve all trained models with optional filtering.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page |
| `algorithm` | string | - | Filter by algorithm: `baseline`, `bilstm`, `deephpm` |

**Response (200 OK):**

```json
{
  "items": [
    {
      "id": 7,
      "algorithm": "bilstm",
      "name": "bilstm-v1-experiment",
      "version": "1.0.0",
      "metrics": {
        "train_loss": 0.0234,
        "val_loss": 0.0289,
        "test_accuracy": 0.92
      },
      "created_at": "2025-01-07T12:00:00Z",
      "file_size": 15728640
    }
  ],
  "meta": {
    "page": 1,
    "page_size": 20,
    "total": 8,
    "total_pages": 1
  }
}
```

---

### Get Model Details

```http
GET /api/v1/models/{model_id}
```

Retrieve detailed information about a specific model.

**Response (200 OK):**

```json
{
  "id": 7,
  "algorithm": "bilstm",
  "name": "bilstm-v1-experiment",
  "version": "1.0.0",
  "metrics": {
    "train_loss": 0.0234,
    "val_loss": 0.0289,
    "test_accuracy": 0.92
  },
  "created_at": "2025-01-07T12:00:00Z",
  "file_size": 15728640
}
```

---

### Delete Model

```http
DELETE /api/v1/models/{model_id}
```

Delete a trained model and its associated files.

**Response (204 No Content)**

**Note:** Cannot delete models that are currently in use by active training jobs or predictions.

---

### Download Model

```http
GET /api/v1/models/{model_id}/download
```

Download the trained model file (.pth format).

**Response (200 OK):**

- Content-Type: `application/octet-stream`
- Content-Disposition: `attachment; filename="model_7_bilstm.pth"`

**Example:**

```bash
curl -O -J http://localhost:8000/api/v1/models/7/download
```

## Data Model

### Model Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique model identifier |
| `algorithm` | string | Algorithm type: `baseline`, `bilstm`, `deephpm` |
| `name` | string | Model name (from training job) |
| `version` | string | Model version |
| `metrics` | object | Training metrics |
| `created_at` | datetime | Creation timestamp |
| `file_size` | integer | Model file size in bytes |

### Metrics Object

| Field | Type | Description |
|-------|------|-------------|
| `train_loss` | float | Final training loss |
| `val_loss` | float | Final validation loss |
| `test_accuracy` | float | Test set accuracy (0-1) |

## Model Storage

- Models are stored as PyTorch `.pth` files
- Storage location: `backend/storage/models/`
- File naming: `model_{id}_{algorithm}.pth`
- Maximum model size: 500MB

## Model Versioning

Models use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Algorithm changes
- **MINOR**: Hyperparameter changes
- **PATCH**: Bug fixes or retraining

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `RESOURCE_NOT_FOUND` | 404 | Model not found |
| `MODEL_IN_USE` | 409 | Cannot delete model in use |
| `INTERNAL_ERROR` | 500 | Server error |

## Notes

- Models are automatically created when training jobs succeed
- Model files are stored on the server filesystem
- Download URLs are direct file downloads (no signed URLs in MVP)
- Models can be used immediately after training completes
