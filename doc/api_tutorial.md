# Battery Life Analysis API - Complete Tutorial

## Introduction

This tutorial walks through a complete workflow: uploading battery data, training a model, and making predictions.

## Prerequisites

- API server running at `http://localhost:8000`
- Valid user credentials
- Battery data file (`.mat` format)

## Step 1: Authentication

First, obtain an access token:

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=secretpassword"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyQGV4YW1wbGUuY29tIiwiZXhwIjoxNzA0NjM2MDAwfQ.abc123",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Save the token:**
```bash
export TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Step 2: Upload Battery Data

Upload a `.mat` file containing battery cycle data:

```bash
curl -X POST http://localhost:8000/api/v1/batteries \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@battery_cell_001.mat" \
  -F "manufacturer=Severson" \
  -F "chemistry=LFP" \
  -F "nominal_capacity=1.1"
```

**Response:**
```json
{
  "id": 125,
  "filename": "battery_cell_001.mat",
  "total_cycles": 1500,
  "nominal_capacity": 1.1,
  "manufacturer": "Severson",
  "chemistry": "LFP",
  "created_at": "2025-01-07T10:00:00Z",
  "updated_at": "2025-01-07T10:00:00Z"
}
```

**Note the battery ID:** `125`

## Step 3: Verify Upload

Check that the battery was uploaded successfully:

```bash
curl http://localhost:8000/api/v1/batteries/125 \
  -H "Authorization: Bearer $TOKEN"
```

## Step 4: View Cycle Data

Retrieve cycle data for visualization:

```bash
curl "http://localhost:8000/api/v1/batteries/125/cycles?page=1&page_size=10" \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "items": [
    {
      "id": 5001,
      "battery_id": 125,
      "cycle_index": 1,
      "voltage": [3.5, 3.6, 3.7, 3.8],
      "current": [1.0, 1.0, 1.0, 1.0],
      "temperature": [25.0, 25.5, 26.0, 26.5],
      "soh": null,
      "rul": null
    }
  ],
  "meta": {
    "page": 1,
    "page_size": 10,
    "total": 1500,
    "total_pages": 150
  }
}
```

## Step 5: Start Training

Train a BiLSTM model using the uploaded battery data:

```bash
curl -X POST http://localhost:8000/api/v1/training-jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "bilstm",
    "battery_ids": [125],
    "hyperparams": {
      "epochs": 100,
      "batch_size": 32,
      "learning_rate": 0.001
    },
    "model_name": "bilstm-tutorial-model"
  }'
```

**Response:**
```json
{
  "id": 43,
  "status": "queued",
  "algorithm": "bilstm",
  "progress": 0.0,
  "current_epoch": 0,
  "total_epochs": 100,
  "created_at": "2025-01-07T10:05:00Z",
  "started_at": null,
  "finished_at": null,
  "model_id": null
}
```

**Note the job ID:** `43`

## Step 6: Monitor Training Progress

### Option A: Polling

Poll the job status every 3 seconds:

```bash
while true; do
  curl http://localhost:8000/api/v1/training-jobs/43 \
    -H "Authorization: Bearer $TOKEN"
  sleep 3
done
```

**Response (in progress):**
```json
{
  "id": 43,
  "status": "running",
  "algorithm": "bilstm",
  "progress": 0.45,
  "current_epoch": 45,
  "total_epochs": 100,
  "created_at": "2025-01-07T10:05:00Z",
  "started_at": "2025-01-07T10:06:00Z",
  "finished_at": null,
  "model_id": null
}
```

**Response (completed):**
```json
{
  "id": 43,
  "status": "succeeded",
  "algorithm": "bilstm",
  "progress": 1.0,
  "current_epoch": 100,
  "total_epochs": 100,
  "created_at": "2025-01-07T10:05:00Z",
  "started_at": "2025-01-07T10:06:00Z",
  "finished_at": "2025-01-07T11:30:00Z",
  "model_id": 8
}
```

### Option B: WebSocket (Real-time)

Connect to WebSocket for live updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/training-jobs/43/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress * 100}%`);
  console.log(`Epoch: ${data.current_epoch}/${data.total_epochs}`);
};
```

## Step 7: View Training Metrics

Retrieve training loss curves:

```bash
curl http://localhost:8000/api/v1/training-jobs/43/metrics \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "train_loss": [0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15],
  "val_loss": [0.52, 0.42, 0.35, 0.3, 0.28, 0.26, 0.24],
  "epochs": [1, 10, 20, 30, 40, 50, 60]
}
```

## Step 8: Verify Model

Check that the model was created:

```bash
curl http://localhost:8000/api/v1/models/8 \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "id": 8,
  "algorithm": "bilstm",
  "name": "bilstm-tutorial-model",
  "version": "1.0.0",
  "metrics": {
    "train_loss": 0.15,
    "val_loss": 0.24,
    "test_accuracy": 0.92
  },
  "created_at": "2025-01-07T11:30:00Z",
  "file_size": 15728640
}
```

## Step 9: Make Prediction

Use the trained model to predict SoH and RUL:

```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "bilstm",
    "model_id": 8,
    "battery_id": 125
  }'
```

**Response:**
```json
{
  "id": 101,
  "algorithm": "bilstm",
  "model_id": 8,
  "result": {
    "soh": 0.87,
    "rul": 120.5,
    "confidence": 0.91
  },
  "created_at": "2025-01-07T12:00:00Z"
}
```

**Interpretation:**
- **SoH**: 0.87 (87% health remaining)
- **RUL**: 120.5 cycles remaining
- **Confidence**: 0.91 (91% confidence)

## Step 10: Batch Predictions

Predict for multiple batteries:

```bash
curl -X POST http://localhost:8000/api/v1/predictions/batch \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"algorithm": "bilstm", "battery_id": 125},
      {"algorithm": "bilstm", "battery_id": 126}
    ]
  }'
```

## Complete Python Example

```python
import requests
import time

BASE_URL = "http://localhost:8000/api/v1"

# 1. Login
response = requests.post(
    f"{BASE_URL}/auth/token",
    data={"username": "user@example.com", "password": "password"}
)
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload battery data
with open("battery_cell_001.mat", "rb") as f:
    files = {"file": f}
    data = {"manufacturer": "Severson", "chemistry": "LFP"}
    response = requests.post(
        f"{BASE_URL}/batteries",
        headers=headers,
        files=files,
        data=data
    )
battery_id = response.json()["id"]
print(f"Uploaded battery ID: {battery_id}")

# 3. Start training
response = requests.post(
    f"{BASE_URL}/training-jobs",
    headers=headers,
    json={
        "algorithm": "bilstm",
        "battery_ids": [battery_id],
        "hyperparams": {"epochs": 100}
    }
)
job_id = response.json()["id"]
print(f"Training job ID: {job_id}")

# 4. Monitor training
while True:
    response = requests.get(
        f"{BASE_URL}/training-jobs/{job_id}",
        headers=headers
    )
    job = response.json()
    print(f"Progress: {job['progress'] * 100:.1f}%")

    if job["status"] in ["succeeded", "failed", "stopped"]:
        break

    time.sleep(5)

# 5. Make prediction
if job["status"] == "succeeded":
    model_id = job["model_id"]
    response = requests.post(
        f"{BASE_URL}/predictions",
        headers=headers,
        json={
            "algorithm": "bilstm",
            "model_id": model_id,
            "battery_id": battery_id
        }
    )
    result = response.json()["result"]
    print(f"SoH: {result['soh']:.2f}")
    print(f"RUL: {result['rul']:.1f} cycles")
```

## Complete JavaScript Example

```javascript
const BASE_URL = 'http://localhost:8000/api/v1';

async function runWorkflow() {
  // 1. Login
  const loginResponse = await fetch(`${BASE_URL}/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      username: 'user@example.com',
      password: 'password'
    })
  });
  const { access_token } = await loginResponse.json();
  const headers = { 'Authorization': `Bearer ${access_token}` };

  // 2. Upload battery data
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('manufacturer', 'Severson');

  const uploadResponse = await fetch(`${BASE_URL}/batteries`, {
    method: 'POST',
    headers,
    body: formData
  });
  const { id: batteryId } = await uploadResponse.json();
  console.log(`Uploaded battery ID: ${batteryId}`);

  // 3. Start training
  const trainingResponse = await fetch(`${BASE_URL}/training-jobs`, {
    method: 'POST',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      algorithm: 'bilstm',
      battery_ids: [batteryId],
      hyperparams: { epochs: 100 }
    })
  });
  const { id: jobId } = await trainingResponse.json();

  // 4. Monitor via WebSocket
  const ws = new WebSocket(`ws://localhost:8000/api/v1/training-jobs/${jobId}/ws`);
  ws.onmessage = async (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'complete') {
      // 5. Make prediction
      const predictionResponse = await fetch(`${BASE_URL}/predictions`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          algorithm: 'bilstm',
          model_id: data.model_id,
          battery_id: batteryId
        })
      });
      const { result } = await predictionResponse.json();
      console.log(`SoH: ${result.soh}`);
      console.log(`RUL: ${result.rul} cycles`);
    }
  };
}
```

## Troubleshooting

### Authentication Failed

```json
{
  "error": {
    "code": "INVALID_CREDENTIALS",
    "message": "Incorrect username or password"
  }
}
```

**Solution**: Verify credentials are correct.

### File Upload Failed

```json
{
  "error": {
    "code": "INVALID_FILE_FORMAT",
    "message": "Unable to parse .mat file"
  }
}
```

**Solution**: Ensure file is valid MATLAB .mat format (v7.0 or v7.3).

### Training Failed

Check training logs:

```bash
curl http://localhost:8000/api/v1/training-jobs/43/logs \
  -H "Authorization: Bearer $TOKEN"
```

## Next Steps

- Explore [API Reference](./api/README.md)
- Learn about [Error Handling](./api/errors.md)
- Try different [algorithms](./api/training-jobs.md#hyperparameters)
- Implement [WebSocket monitoring](./api/websocket.md)

## Summary

You've completed a full workflow:
1. ✅ Authenticated with the API
2. ✅ Uploaded battery data
3. ✅ Trained a BiLSTM model
4. ✅ Monitored training progress
5. ✅ Made SoH/RUL predictions

The API is now ready for integration into your application!
