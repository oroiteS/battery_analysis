# Battery Data Management API

## Overview

The Batteries API provides endpoints for managing battery datasets, including uploading `.mat` files, retrieving metadata, and accessing cycle data for visualization.

## Endpoints

### List Batteries

```http
GET /api/v1/batteries
```

Retrieve a paginated list of all battery datasets.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page (max 100) |
| `manufacturer` | string | - | Filter by manufacturer |

**Response (200 OK):**

```json
{
  "items": [
    {
      "id": 1,
      "filename": "battery_cell_001.mat",
      "total_cycles": 1200,
      "nominal_capacity": 1.1,
      "manufacturer": "Severson",
      "chemistry": "LFP",
      "created_at": "2025-01-01T00:00:00Z",
      "updated_at": "2025-01-01T00:00:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "page_size": 20,
    "total": 124,
    "total_pages": 7
  }
}
```

---

### Upload Battery Data

```http
POST /api/v1/batteries
```

Upload a `.mat` file containing battery cycle data. The file will be parsed automatically to extract metadata.

**Request (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | binary | Yes | .mat file with battery data |
| `manufacturer` | string | No | Battery manufacturer |
| `chemistry` | string | No | Battery chemistry (e.g., LFP, NMC) |
| `nominal_capacity` | float | No | Nominal capacity in Ah |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/batteries \
  -F "file=@battery_data.mat" \
  -F "manufacturer=Severson" \
  -F "chemistry=LFP" \
  -F "nominal_capacity=1.1"
```

**Response (201 Created):**

```json
{
  "id": 125,
  "filename": "battery_data.mat",
  "total_cycles": 1500,
  "nominal_capacity": 1.1,
  "manufacturer": "Severson",
  "chemistry": "LFP",
  "created_at": "2025-01-07T10:00:00Z",
  "updated_at": "2025-01-07T10:00:00Z"
}
```

**Error Response (400 Bad Request):**

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
    "trace_id": "req_abc123"
  }
}
```

---

### Get Battery Details

```http
GET /api/v1/batteries/{battery_id}
```

Retrieve detailed information about a specific battery.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `battery_id` | integer | Battery ID |

**Response (200 OK):**

```json
{
  "id": 1,
  "filename": "battery_cell_001.mat",
  "total_cycles": 1200,
  "nominal_capacity": 1.1,
  "manufacturer": "Severson",
  "chemistry": "LFP",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

**Error Response (404 Not Found):**

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Battery with ID 999 not found",
    "trace_id": "req_xyz789"
  }
}
```

---

### Update Battery Metadata

```http
PATCH /api/v1/batteries/{battery_id}
```

Update battery metadata (partial update).

**Request Body:**

```json
{
  "manufacturer": "Updated Manufacturer",
  "chemistry": "NMC",
  "nominal_capacity": 1.2
}
```

**Response (200 OK):**

```json
{
  "id": 1,
  "filename": "battery_cell_001.mat",
  "total_cycles": 1200,
  "nominal_capacity": 1.2,
  "manufacturer": "Updated Manufacturer",
  "chemistry": "NMC",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-07T10:30:00Z"
}
```

---

### Delete Battery

```http
DELETE /api/v1/batteries/{battery_id}
```

Delete a battery dataset and all associated cycle data.

**Response (204 No Content)**

---

### Get Battery Cycle Data

```http
GET /api/v1/batteries/{battery_id}/cycles
```

Retrieve cycle data for a specific battery. Useful for visualization and analysis.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page |
| `cycle_index` | integer | - | Filter by specific cycle index |

**Response (200 OK):**

```json
{
  "items": [
    {
      "id": 1001,
      "battery_id": 1,
      "cycle_index": 100,
      "voltage": [3.5, 3.6, 3.7, 3.8],
      "current": [1.0, 1.0, 1.0, 1.0],
      "temperature": [25.0, 25.5, 26.0, 26.5],
      "soh": 0.95,
      "rul": 1100
    }
  ],
  "meta": {
    "page": 1,
    "page_size": 20,
    "total": 1200,
    "total_pages": 60
  }
}
```

## Data Model

### Battery Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique battery identifier |
| `filename` | string | Original .mat filename |
| `total_cycles` | integer | Total number of charge/discharge cycles |
| `nominal_capacity` | float | Nominal capacity in Ah |
| `manufacturer` | string | Battery manufacturer |
| `chemistry` | string | Battery chemistry type |
| `created_at` | datetime | Creation timestamp |
| `updated_at` | datetime | Last update timestamp |

### Cycle Data Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique cycle record identifier |
| `battery_id` | integer | Associated battery ID |
| `cycle_index` | integer | Cycle number |
| `voltage` | array[float] | Voltage measurements |
| `current` | array[float] | Current measurements |
| `temperature` | array[float] | Temperature measurements |
| `soh` | float | State of Health (0-1) |
| `rul` | float | Remaining Useful Life (cycles) |

## Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_FILE_FORMAT` | 400 | .mat file cannot be parsed |
| `VALIDATION_ERROR` | 422 | Request validation failed |
| `RESOURCE_NOT_FOUND` | 404 | Battery not found |
| `INTERNAL_ERROR` | 500 | Server error |

## Notes

- Maximum file upload size: 100MB
- Supported .mat file versions: MATLAB v7.0 and v7.3
- Cycle data arrays are stored as JSON arrays for API responses
- Large datasets are automatically paginated
