# Error Handling

## Overview

All API errors follow a consistent JSON format with detailed error information, field-level validation details, and trace IDs for debugging.

## Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error description",
    "details": [
      {
        "field": "field_name",
        "issue": "specific issue description"
      }
    ],
    "trace_id": "req_abc123xyz"
  }
}
```

## HTTP Status Codes

| Status Code | Meaning | When to Use |
|-------------|---------|-------------|
| **400** | Bad Request | Invalid request format, malformed JSON |
| **401** | Unauthorized | Missing or invalid authentication token |
| **403** | Forbidden | Valid token but insufficient permissions |
| **404** | Not Found | Resource does not exist |
| **409** | Conflict | Resource conflict (e.g., duplicate, in use) |
| **422** | Unprocessable Entity | Validation failed (Pydantic errors) |
| **500** | Internal Server Error | Unexpected server error |
| **503** | Service Unavailable | Service temporarily unavailable |

## Error Codes

### Authentication Errors (401)

| Code | Description |
|------|-------------|
| `INVALID_TOKEN` | JWT token is invalid or expired |
| `MISSING_TOKEN` | Authorization header not provided |
| `TOKEN_EXPIRED` | Token has expired |

**Example:**

```json
{
  "error": {
    "code": "INVALID_TOKEN",
    "message": "Authentication token is invalid",
    "trace_id": "req_abc123"
  }
}
```

### Validation Errors (400, 422)

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `INVALID_FILE_FORMAT` | Uploaded file format is invalid |
| `MISSING_REQUIRED_FIELD` | Required field not provided |

**Example:**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "battery_ids",
        "issue": "must contain at least 1 item"
      },
      {
        "field": "learning_rate",
        "issue": "must be greater than 0"
      }
    ],
    "trace_id": "req_def456"
  }
}
```

### Resource Errors (404, 409)

| Code | Description |
|------|-------------|
| `RESOURCE_NOT_FOUND` | Requested resource does not exist |
| `RESOURCE_CONFLICT` | Resource conflict (duplicate, in use) |
| `MODEL_IN_USE` | Cannot delete model currently in use |

**Example:**

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Battery with ID 999 not found",
    "trace_id": "req_ghi789"
  }
}
```

### Server Errors (500, 503)

| Code | Description |
|------|-------------|
| `INTERNAL_ERROR` | Unexpected server error |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable |
| `DATABASE_ERROR` | Database connection or query error |

**Example:**

```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "trace_id": "req_jkl012"
  }
}
```

## Field-Level Validation

Pydantic validation errors (422) include detailed field-level information:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "algorithm",
        "issue": "value is not a valid enumeration member; permitted: 'baseline', 'bilstm', 'deephpm'"
      },
      {
        "field": "epochs",
        "issue": "ensure this value is greater than 0"
      }
    ],
    "trace_id": "req_mno345"
  }
}
```

## Trace IDs

Every error response includes a `trace_id` for debugging:

- Format: `req_<random_string>`
- Used to correlate errors with server logs
- Include trace ID when reporting issues

## Error Handling Best Practices

### Client-Side

```javascript
try {
  const response = await fetch('/api/v1/batteries', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    console.error(`Error ${error.error.code}: ${error.error.message}`);
    console.error(`Trace ID: ${error.error.trace_id}`);

    // Handle field-level errors
    if (error.error.details) {
      error.error.details.forEach(detail => {
        console.error(`Field ${detail.field}: ${detail.issue}`);
      });
    }
  }
} catch (err) {
  console.error('Network error:', err);
}
```

### Python

```python
import requests

response = requests.post('http://localhost:8000/api/v1/batteries', files=files)

if response.status_code != 201:
    error = response.json()['error']
    print(f"Error {error['code']}: {error['message']}")
    print(f"Trace ID: {error['trace_id']}")

    if 'details' in error:
        for detail in error['details']:
            print(f"Field {detail['field']}: {detail['issue']}")
```

## Common Error Scenarios

### Invalid File Upload

```http
POST /api/v1/batteries
Content-Type: multipart/form-data

file: corrupted_file.mat
```

**Response (400):**

```json
{
  "error": {
    "code": "INVALID_FILE_FORMAT",
    "message": "Unable to parse .mat file",
    "details": [
      {
        "field": "file",
        "issue": "File format not recognized or corrupted"
      }
    ],
    "trace_id": "req_pqr678"
  }
}
```

### Missing Authentication

```http
GET /api/v1/batteries
```

**Response (401):**

```json
{
  "error": {
    "code": "MISSING_TOKEN",
    "message": "Authentication required",
    "trace_id": "req_stu901"
  }
}
```

### Resource Not Found

```http
GET /api/v1/batteries/999
```

**Response (404):**

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Battery with ID 999 not found",
    "trace_id": "req_vwx234"
  }
}
```

### Training Job Validation Error

```http
POST /api/v1/training-jobs
Content-Type: application/json

{
  "algorithm": "invalid_algo",
  "battery_ids": []
}
```

**Response (422):**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "algorithm",
        "issue": "value is not a valid enumeration member; permitted: 'baseline', 'bilstm', 'deephpm'"
      },
      {
        "field": "battery_ids",
        "issue": "must contain at least 1 item"
      }
    ],
    "trace_id": "req_yza567"
  }
}
```

## Rate Limiting (Future)

When rate limiting is implemented, responses will include:

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1609459200
```

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests. Please try again later.",
    "trace_id": "req_bcd890"
  }
}
```

## Notes

- All errors use consistent JSON structure
- Trace IDs are logged server-side for debugging
- Field-level validation details help identify specific issues
- HTTP status codes follow REST conventions
