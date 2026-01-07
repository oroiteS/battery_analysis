# WebSocket API

## Overview

WebSocket endpoints provide real-time updates for long-running operations, specifically training job progress monitoring.

## Endpoint

```
WS /api/v1/training-jobs/{job_id}/ws
```

Connect to this endpoint to receive real-time training progress updates.

## Connection

### JavaScript

```javascript
const jobId = 42;
const ws = new WebSocket(`ws://localhost:8000/api/v1/training-jobs/${jobId}/stream`);

ws.onopen = () => {
  console.log('Connected to training job stream');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress update:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Connection closed');
};
```

### Python

```python
import asyncio
import websockets
import json

async def monitor_training(job_id):
    uri = f"ws://localhost:8000/api/v1/training-jobs/{job_id}/ws"

    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            data = json.loads(message)
            print(f"Progress: {data['progress'] * 100}%")
            print(f"Epoch: {data['current_epoch']}/{data['total_epochs']}")

asyncio.run(monitor_training(42))
```

## Message Format

### Progress Update

```json
{
  "type": "progress",
  "job_id": 42,
  "status": "running",
  "progress": 0.45,
  "current_epoch": 45,
  "total_epochs": 100,
  "current_loss": 0.0234,
  "timestamp": "2025-01-07T10:15:00Z"
}
```

### Log Message

```json
{
  "type": "log",
  "job_id": 42,
  "level": "INFO",
  "message": "Epoch 45/100 - Loss: 0.0234",
  "timestamp": "2025-01-07T10:15:00Z"
}
```

### Completion

```json
{
  "type": "complete",
  "job_id": 42,
  "status": "succeeded",
  "model_id": 7,
  "final_metrics": {
    "train_loss": 0.0234,
    "val_loss": 0.0289
  },
  "timestamp": "2025-01-07T12:00:00Z"
}
```

### Error

```json
{
  "type": "error",
  "job_id": 42,
  "status": "failed",
  "error_message": "CUDA out of memory",
  "timestamp": "2025-01-07T11:30:00Z"
}
```

## Message Types

| Type | Description | When Sent |
|------|-------------|-----------|
| `progress` | Training progress update | Every epoch |
| `log` | Training log message | As logs are generated |
| `complete` | Training completed | Job finishes successfully |
| `error` | Training failed | Job encounters error |

## Connection Lifecycle

```
1. Client connects to WebSocket endpoint
2. Server validates job_id exists
3. Server sends initial status message
4. Server streams updates as training progresses
5. Server sends completion/error message
6. Connection closes automatically
```

## Authentication

WebSocket connections require authentication:

```javascript
const token = 'your_jwt_token';
const ws = new WebSocket(
  `ws://localhost:8000/api/v1/training-jobs/${jobId}/stream`,
  ['Bearer', token]
);
```

Or via query parameter:

```javascript
const ws = new WebSocket(
  `ws://localhost:8000/api/v1/training-jobs/${jobId}/stream?token=${token}`
);
```

## Error Handling

### Job Not Found

If the job_id doesn't exist, the connection closes immediately with code 1008:

```javascript
ws.onclose = (event) => {
  if (event.code === 1008) {
    console.error('Training job not found');
  }
};
```

### Connection Lost

Handle reconnection logic:

```javascript
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

function connect(jobId) {
  const ws = new WebSocket(`ws://localhost:8000/api/v1/training-jobs/${jobId}/stream`);

  ws.onclose = () => {
    if (reconnectAttempts < maxReconnectAttempts) {
      reconnectAttempts++;
      console.log(`Reconnecting... (${reconnectAttempts}/${maxReconnectAttempts})`);
      setTimeout(() => connect(jobId), 2000);
    }
  };

  return ws;
}
```

## Complete Example

### React Component

```javascript
import { useEffect, useState } from 'react';

function TrainingMonitor({ jobId }) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('connecting');

  useEffect(() => {
    const ws = new WebSocket(
      `ws://localhost:8000/api/v1/training-jobs/${jobId}/stream`
    );

    ws.onopen = () => setStatus('connected');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'progress') {
        setProgress(data.progress);
        setStatus(data.status);
      } else if (data.type === 'complete') {
        setStatus('completed');
        setProgress(1.0);
      } else if (data.type === 'error') {
        setStatus('failed');
      }
    };

    ws.onerror = () => setStatus('error');
    ws.onclose = () => setStatus('disconnected');

    return () => ws.close();
  }, [jobId]);

  return (
    <div>
      <p>Status: {status}</p>
      <progress value={progress} max="1" />
      <p>{Math.round(progress * 100)}%</p>
    </div>
  );
}
```

## Polling Alternative

If WebSocket is not available, use polling:

```javascript
async function pollTrainingStatus(jobId) {
  const interval = setInterval(async () => {
    const response = await fetch(
      `http://localhost:8000/api/v1/training-jobs/${jobId}`
    );
    const data = await response.json();

    console.log(`Progress: ${data.progress * 100}%`);

    if (['succeeded', 'failed', 'stopped'].includes(data.status)) {
      clearInterval(interval);
    }
  }, 3000); // Poll every 3 seconds
}
```

## Performance Notes

- WebSocket connections are lightweight
- Updates sent every epoch (not every batch)
- Maximum concurrent connections per job: 10
- Idle connections timeout after 5 minutes
- Automatic reconnection recommended

## Browser Compatibility

WebSocket is supported in all modern browsers:
- Chrome 16+
- Firefox 11+
- Safari 7+
- Edge (all versions)

## Notes

- WebSocket endpoint is optional; polling is always available
- Connection closes automatically when job completes
- Multiple clients can connect to the same job stream
- Messages are broadcast to all connected clients
- No message acknowledgment required (fire-and-forget)
