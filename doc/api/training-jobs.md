# Training Jobs API

## Overview

The Training Jobs API manages asynchronous model training operations for battery SoH/RUL prediction. Supports three algorithms: Baseline, BiLSTM, and DeepHPM.

## Endpoints

### List Training Jobs

```http
GET /api/v1/training-jobs
```

Retrieve all training jobs with optional filtering.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page |
| `status` | string | - | Filter by status: `queued`, `running`, `succeeded`, `failed`, `stopped` |

**Response (200 OK):**

```json
{
  "items": [
    {
      "id": 42,
      "status": "running",
      "algorithm": "bilstm",
      "progress": 0.45,
      "current_epoch": 45,
      "total_epochs": 100,
      "created_at": "2025-01-07T10:00:00Z",
      "started_at": "2025-01-07T10:01:00Z",
      "finished_at": null,
      "model_id": null
    }
  ],
  "meta": {
    "page": 1,
    "page_size": 20,
    "total": 15,
    "total_pages": 1
  }
}
```

---

### Start Training Job

```http
POST /api/v1/training-jobs
```

Initiate a new training job. Training runs asynchronously in the background.

**Request Body:**

```json
{
  "algorithm": "bilstm",
  "battery_ids": [1, 2, 3, 4, 5],
  "hyperparams": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "model_name": "bilstm-v1-experiment"
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `algorithm` | string | Yes | Algorithm: `baseline`, `bilstm`, or `deephpm` |
| `battery_ids` | array[int] | Yes | Battery datasets to train on (min 1) |
| `hyperparams` | object | No | Training hyperparameters |
| `model_name` | string | No | Custom model name |

**Response (201 Created):**

```json
{
  "id": 43,
  "status": "queued",
  "algorithm": "bilstm",
  "progress": 0.0,
  "current_epoch": 0,
  "total_epochs": 100,
  "created_at": "2025-01-07T11:00:00Z",
  "started_at": null,
  "finished_at": null,
  "model_id": null
}
```

---

### Get Training Job Details

```http
GET /api/v1/training-jobs/{job_id}
```

Retrieve detailed status and progress of a training job.

**Response (200 OK):**

```json
{
  "id": 42,
  "status": "running",
  "algorithm": "bilstm",
  "progress": 0.67,
  "current_epoch": 67,
  "total_epochs": 100,
  "created_at": "2025-01-07T10:00:00Z",
  "started_at": "2025-01-07T10:01:00Z",
  "finished_at": null,
  "model_id": null
}
```

---

### Stop Training Job

```http
POST /api/v1/training-jobs/{job_id}/stop
```

Request cancellation of a running training job.

**Response (200 OK):**

```json
{
  "id": 42,
  "status": "stopped",
  "algorithm": "bilstm",
  "progress": 0.67,
  "current_epoch": 67,
  "total_epochs": 100,
  "created_at": "2025-01-07T10:00:00Z",
  "started_at": "2025-01-07T10:01:00Z",
  "finished_at": "2025-01-07T11:30:00Z",
  "model_id": null
}
```

---

### Get Training Logs

```http
GET /api/v1/training-jobs/{job_id}/logs
```

Retrieve training logs for debugging and monitoring.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 50 | Items per page |

**Response (200 OK):**

```json
{
  "logs": [
    {
      "timestamp": "2025-01-07T10:05:00Z",
      "level": "INFO",
      "message": "Epoch 10/100 - Loss: 0.0234"
    },
    {
      "timestamp": "2025-01-07T10:10:00Z",
      "level": "WARNING",
      "message": "Validation loss increased"
    }
  ]
}
```

---

### Get Training Metrics

```http
GET /api/v1/training-jobs/{job_id}/metrics
```

Retrieve training metrics for visualization (loss curves, accuracy).

**Response (200 OK):**

```json
{
  "train_loss": [0.5, 0.4, 0.3, 0.25, 0.2],
  "val_loss": [0.52, 0.42, 0.35, 0.3, 0.28],
  "epochs": [1, 2, 3, 4, 5]
}
```

## Training Job Lifecycle

```
queued → running → succeeded
                 ↘ failed
                 ↘ stopped (manual cancellation)
```

**Status Descriptions:**

| Status | Description |
|--------|-------------|
| `queued` | Job created, waiting to start |
| `running` | Training in progress |
| `succeeded` | Training completed successfully, model saved |
| `failed` | Training failed due to error |
| `stopped` | Training manually cancelled |

## Hyperparameters

### Baseline Algorithm

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | integer | 100 | Number of training epochs |
| `batch_size` | integer | 32 | Batch size |
| `learning_rate` | float | 0.001 | Learning rate |

### BiLSTM Algorithm

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | integer | 100 | Number of training epochs |
| `batch_size` | integer | 32 | Batch size |
| `learning_rate` | float | 0.001 | Learning rate |
| `hidden_size` | integer | 64 | LSTM hidden layer size |

### DeepHPM Algorithm

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | integer | 100 | Number of training epochs |
| `batch_size` | integer | 32 | Batch size |
| `learning_rate` | float | 0.001 | Learning rate |
| `physics_weight` | float | 0.5 | Weight for physics loss term |

## Progress Monitoring

### Polling (Recommended for Simple UIs)

Poll the job status endpoint every 2-5 seconds:

```bash
while true; do
  curl http://localhost:8000/api/v1/training-jobs/42
  sleep 3
done
```

### WebSocket (Real-time Updates)

Connect to WebSocket endpoint for live progress updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/training-jobs/42/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress * 100}%`);
  console.log(`Epoch: ${data.current_epoch}/${data.total_epochs}`);
};
```

## Error Handling

**Validation Error (422):**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid training configuration",
    "details": [
      {
        "field": "battery_ids",
        "issue": "must contain at least 1 item"
      }
    ],
    "trace_id": "req_abc123"
  }
}
```

**Training Failed (200 with status=failed):**

When retrieving a failed job, check the logs endpoint for error details.

## Notes

- Training jobs run asynchronously using FastAPI BackgroundTasks
- Maximum concurrent training jobs: 3 (configurable)
- Job history is retained for 30 days
- Model artifacts are saved only for `succeeded` jobs
