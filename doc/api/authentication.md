# Authentication

## Overview

The API uses OAuth2 with Password Flow for authentication. All protected endpoints require a valid JWT Bearer token.

## Authentication Flow

```
1. Client sends username/password to /auth/token
2. Server validates credentials
3. Server returns JWT access token
4. Client includes token in Authorization header for subsequent requests
```

## Obtaining a Token

### Request

```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=secretpassword
```

### Response (200 OK)

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Example (cURL)

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=secretpassword"
```

### Example (JavaScript)

```javascript
const response = await fetch('http://localhost:8000/api/v1/auth/token', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded',
  },
  body: new URLSearchParams({
    username: 'user@example.com',
    password: 'secretpassword'
  })
});

const data = await response.json();
const token = data.access_token;
```

### Example (Python)

```python
import requests

response = requests.post(
    'http://localhost:8000/api/v1/auth/token',
    data={
        'username': 'user@example.com',
        'password': 'secretpassword'
    }
)

token = response.json()['access_token']
```

## Using the Token

Include the token in the `Authorization` header with the `Bearer` scheme:

```http
GET /api/v1/batteries
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Example (cURL)

```bash
curl http://localhost:8000/api/v1/batteries \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Example (JavaScript)

```javascript
const response = await fetch('http://localhost:8000/api/v1/batteries', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
```

### Example (Python)

```python
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('http://localhost:8000/api/v1/batteries', headers=headers)
```

## Token Expiration

- Default expiration: **1 hour** (3600 seconds)
- Tokens cannot be refreshed (obtain a new token when expired)
- Check `expires_in` field in token response

## Logout

```http
POST /api/v1/auth/logout
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (204 No Content)**

Note: Logout invalidates the token server-side.

## Get Current User

```http
GET /api/v1/auth/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (200 OK):**

```json
{
  "id": 1,
  "username": "user@example.com",
  "email": "user@example.com",
  "created_at": "2025-01-01T00:00:00Z"
}
```

## Error Responses

### Invalid Credentials (401)

```json
{
  "error": {
    "code": "INVALID_CREDENTIALS",
    "message": "Incorrect username or password",
    "trace_id": "req_abc123"
  }
}
```

### Missing Token (401)

```json
{
  "error": {
    "code": "MISSING_TOKEN",
    "message": "Authentication required",
    "trace_id": "req_def456"
  }
}
```

### Invalid Token (401)

```json
{
  "error": {
    "code": "INVALID_TOKEN",
    "message": "Authentication token is invalid",
    "trace_id": "req_ghi789"
  }
}
```

### Expired Token (401)

```json
{
  "error": {
    "code": "TOKEN_EXPIRED",
    "message": "Authentication token has expired",
    "trace_id": "req_jkl012"
  }
}
```

## Public Endpoints

The following endpoints do NOT require authentication:

- `POST /api/v1/auth/token` - Login
- `GET /api/v1/health` - Health check
- `GET /api/v1/version` - Version info

## Protected Endpoints

All other endpoints require authentication:

- Battery management
- Training jobs
- Models
- Predictions

## Token Storage

### Browser (localStorage)

```javascript
// Store token
localStorage.setItem('access_token', token);

// Retrieve token
const token = localStorage.getItem('access_token');

// Remove token
localStorage.removeItem('access_token');
```

### Browser (sessionStorage)

```javascript
// More secure: cleared when tab closes
sessionStorage.setItem('access_token', token);
```

### Python

```python
# Store in memory
class APIClient:
    def __init__(self):
        self.token = None

    def login(self, username, password):
        response = requests.post(
            'http://localhost:8000/api/v1/auth/token',
            data={'username': username, 'password': password}
        )
        self.token = response.json()['access_token']

    def get_batteries(self):
        headers = {'Authorization': f'Bearer {self.token}'}
        return requests.get(
            'http://localhost:8000/api/v1/batteries',
            headers=headers
        ).json()
```

## Security Best Practices

1. **HTTPS Only**: Always use HTTPS in production
2. **Secure Storage**: Never store tokens in cookies without `httpOnly` flag
3. **Token Rotation**: Obtain new tokens before expiration
4. **Logout**: Always logout when done
5. **No URL Parameters**: Never pass tokens in URL query parameters

## Complete Authentication Example

### JavaScript

```javascript
class BatteryAPI {
  constructor(baseURL) {
    this.baseURL = baseURL;
    this.token = null;
  }

  async login(username, password) {
    const response = await fetch(`${this.baseURL}/auth/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({ username, password })
    });

    if (!response.ok) {
      throw new Error('Login failed');
    }

    const data = await response.json();
    this.token = data.access_token;
    localStorage.setItem('access_token', this.token);
  }

  async request(endpoint, options = {}) {
    if (!this.token) {
      this.token = localStorage.getItem('access_token');
    }

    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers: {
        ...options.headers,
        'Authorization': `Bearer ${this.token}`
      }
    });

    if (response.status === 401) {
      // Token expired, need to re-login
      this.token = null;
      localStorage.removeItem('access_token');
      throw new Error('Authentication required');
    }

    return response.json();
  }

  async getBatteries() {
    return this.request('/batteries');
  }

  async logout() {
    await this.request('/auth/logout', { method: 'POST' });
    this.token = null;
    localStorage.removeItem('access_token');
  }
}

// Usage
const api = new BatteryAPI('http://localhost:8000/api/v1');
await api.login('user@example.com', 'password');
const batteries = await api.getBatteries();
```

## Notes

- JWT tokens are stateless (server doesn't store them)
- Token payload includes user ID and expiration time
- Tokens are signed with server secret key
- No refresh tokens in MVP (re-login when expired)
- Rate limiting may apply to login endpoint (future)
