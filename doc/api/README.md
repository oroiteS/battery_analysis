# Battery Life Analysis Platform API

## Overview

RESTful API for battery State of Health (SoH) and Remaining Useful Life (RUL) prediction. Supports three deep learning algorithms: Baseline, BiLSTM, and DeepHPM.

## Base URL

```
http://localhost:8000/api/v1
```

## Quick Start

### 1. Authentication

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=password"
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 2. Upload Battery Data

```bash
curl -X POST http://localhost:8000/api/v1/batteries \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@battery_data.mat" \
  -F "manufacturer=Severson"
```

### 3. Start Training

```bash
curl -X POST http://localhost:8000/api/v1/training-jobs \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "bilstm",
    "battery_ids": [1, 2, 3],
    "hyperparams": {"epochs": 100}
  }'
```

### 4. Make Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "bilstm",
    "battery_id": 1
  }'
```

## API Resources

| Resource          | Description                 | Documentation                          |
| ----------------- | --------------------------- | -------------------------------------- |
| **Batteries**     | Battery data management     | [batteries.md](./batteries.md)         |
| **Training Jobs** | Model training operations   | [training-jobs.md](./training-jobs.md) |
| **Models**        | Trained model management    | [models.md](./models.md)               |
| **Predictions**   | SoH/RUL prediction          | [predictions.md](./predictions.md)     |
| **System**        | Health check & version info | `/health`, `/version`                  |

## Core Concepts

### Algorithms

| Algorithm    | Type             | Use Case                     |
| ------------ | ---------------- | ---------------------------- |
| **Baseline** | Data-driven NN   | Quick baseline predictions   |
| **BiLSTM**   | Recurrent NN     | Time-series pattern analysis |
| **DeepHPM**  | Physics-informed | High-accuracy predictions    |

### Data Flow

```
Upload .mat file → Extract features → Train model → Predict SoH/RUL
```

### Asynchronous Operations

Training jobs run asynchronously. Monitor progress via:

- **Polling**: `GET /training-jobs/{id}` every 3-5 seconds
- **WebSocket**: Real-time updates via `WS /training-jobs/{id}/ws`

## Documentation

### API Reference

- [OpenAPI Specification](./openapi.yaml) - Complete API spec
- [Batteries API](./batteries.md) - Battery data management
- [Training Jobs API](./training-jobs.md) - Model training
- [Models API](./models.md) - Model management
- [Predictions API](./predictions.md) - SoH/RUL prediction

### Guides

- [Authentication](./authentication.md) - OAuth2 authentication
- [Error Handling](./errors.md) - Error codes and formats
- [Pagination](./pagination.md) - List pagination
- [WebSocket](./websocket.md) - Real-time updates

### Tutorials

- [Complete Tutorial](../api_tutorial.md) - End-to-end workflow

## Response Format

### Success Response

```json
{
  "id": 1,
  "field": "value",
  "created_at": "2025-01-07T10:00:00Z"
}
```

### Error Response

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": [...],
    "trace_id": "req_abc123"
  }
}
```

## HTTP Status Codes

| Code | Meaning          |
| ---- | ---------------- |
| 200  | Success          |
| 201  | Created          |
| 204  | No Content       |
| 400  | Bad Request      |
| 401  | Unauthorized     |
| 404  | Not Found        |
| 422  | Validation Error |
| 500  | Server Error     |

## Rate Limits

- **Login**: 10 requests/minute
- **API calls**: 100 requests/minute (authenticated)
- **File uploads**: 5 requests/minute

## Supported File Formats

- **Battery data**: MATLAB .mat files (v7.0, v7.3)
- **Model files**: PyTorch .pth files

## Client Libraries

### Python

```python
import requests

class BatteryAPI:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}

    def get_batteries(self):
        return requests.get(
            f'{self.base_url}/batteries',
            headers=self.headers
        ).json()
```

### JavaScript

```javascript
class BatteryAPI {
  constructor(baseURL, token) {
    this.baseURL = baseURL;
    this.token = token;
  }

  async getBatteries() {
    const response = await fetch(`${this.baseURL}/batteries`, {
      headers: { 'Authorization': `Bearer ${this.token}` }
    });
    return response.json();
  }
}
```

## Development

### Local Setup

```bash
# Start backend
cd backend
uv sync
source .venv/bin/activate
uvicorn main:app --reload

# Start database
docker run -d --name battery-mysql \
  -p 13306:3306 \
  -e MYSQL_ROOT_PASSWORD=root \
  mysql:8.0
```

### Testing

```bash
# Run tests
pytest tests/

# Test API endpoint
curl http://localhost:8000/api/v1/health
```

## Support

- **Issues**: Report bugs at [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: Full docs at `/docs` (Swagger UI)
- **Email**: support@battery-analysis.com

## Version

Current version: **1.0.0**

Last updated: 2025-01-07
