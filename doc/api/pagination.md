# Pagination

## Overview

All list endpoints support pagination to handle large datasets efficiently. The API uses offset-based pagination with page numbers.

## Query Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `page` | integer | 1 | 1-∞ | Page number (1-indexed) |
| `page_size` | integer | 20 | 1-100 | Items per page |

## Response Format

All paginated responses include:

```json
{
  "items": [...],
  "meta": {
    "page": 1,
    "page_size": 20,
    "total": 124,
    "total_pages": 7
  }
}
```

### Meta Object

| Field | Type | Description |
|-------|------|-------------|
| `page` | integer | Current page number |
| `page_size` | integer | Items per page |
| `total` | integer | Total number of items |
| `total_pages` | integer | Total number of pages |

## Examples

### First Page

```http
GET /api/v1/batteries?page=1&page_size=20
```

**Response:**

```json
{
  "items": [
    {"id": 1, "filename": "battery_001.mat"},
    {"id": 2, "filename": "battery_002.mat"}
  ],
  "meta": {
    "page": 1,
    "page_size": 20,
    "total": 124,
    "total_pages": 7
  }
}
```

### Last Page

```http
GET /api/v1/batteries?page=7&page_size=20
```

**Response:**

```json
{
  "items": [
    {"id": 121, "filename": "battery_121.mat"},
    {"id": 122, "filename": "battery_122.mat"},
    {"id": 123, "filename": "battery_123.mat"},
    {"id": 124, "filename": "battery_124.mat"}
  ],
  "meta": {
    "page": 7,
    "page_size": 20,
    "total": 124,
    "total_pages": 7
  }
}
```

### Custom Page Size

```http
GET /api/v1/batteries?page=1&page_size=50
```

## Paginated Endpoints

| Endpoint | Default Page Size | Max Page Size |
|----------|-------------------|---------------|
| `GET /api/v1/batteries` | 20 | 100 |
| `GET /api/v1/batteries/{id}/cycles` | 20 | 100 |
| `GET /api/v1/training-jobs` | 20 | 100 |
| `GET /api/v1/models` | 20 | 100 |
| `GET /api/v1/predictions` | 20 | 100 |

## Client Implementation

### JavaScript

```javascript
async function fetchAllBatteries() {
  let page = 1;
  let allBatteries = [];

  while (true) {
    const response = await fetch(
      `http://localhost:8000/api/v1/batteries?page=${page}&page_size=50`
    );
    const data = await response.json();

    allBatteries.push(...data.items);

    if (page >= data.meta.total_pages) break;
    page++;
  }

  return allBatteries;
}
```

### Python

```python
def fetch_all_batteries():
    batteries = []
    page = 1

    while True:
        response = requests.get(
            'http://localhost:8000/api/v1/batteries',
            params={'page': page, 'page_size': 50}
        )
        data = response.json()

        batteries.extend(data['items'])

        if page >= data['meta']['total_pages']:
            break
        page += 1

    return batteries
```

## Error Handling

### Invalid Page Number

```http
GET /api/v1/batteries?page=0
```

**Response (422):**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid pagination parameters",
    "details": [
      {
        "field": "page",
        "issue": "ensure this value is greater than or equal to 1"
      }
    ],
    "trace_id": "req_abc123"
  }
}
```

### Page Size Exceeds Maximum

```http
GET /api/v1/batteries?page_size=200
```

**Response (422):**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid pagination parameters",
    "details": [
      {
        "field": "page_size",
        "issue": "ensure this value is less than or equal to 100"
      }
    ],
    "trace_id": "req_def456"
  }
}
```

### Page Beyond Total Pages

```http
GET /api/v1/batteries?page=999
```

**Response (200 OK):**

```json
{
  "items": [],
  "meta": {
    "page": 999,
    "page_size": 20,
    "total": 124,
    "total_pages": 7
  }
}
```

*Note: Returns empty items array, not an error.*

## Performance Considerations

- **Small page sizes** (1-20): Faster response, more requests
- **Large page sizes** (50-100): Fewer requests, slower response
- **Recommended**: Use `page_size=50` for bulk operations
- **Database**: Pagination uses `LIMIT` and `OFFSET` SQL clauses

## Filtering with Pagination

Combine pagination with filters:

```http
GET /api/v1/batteries?manufacturer=Severson&page=1&page_size=20
```

```http
GET /api/v1/training-jobs?status=running&page=1&page_size=10
```

## Notes

- Page numbers are 1-indexed (first page is `page=1`)
- Empty result sets return `items: []` with `total: 0`
- Total count is calculated on every request
- Pagination parameters are validated by Pydantic
- Maximum page size is enforced server-side
