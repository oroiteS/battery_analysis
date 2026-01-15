import service from './index'
import type { LoginData, RegisterData, TokenResponse, UserResponse } from './types'

export const login = (data: LoginData) => {
  // Convert to URLSearchParams for application/x-www-form-urlencoded
  const params = new URLSearchParams()
  params.append('username', data.username)
  params.append('password', data.password)

  // Axios response interceptor unwraps response.data, so we return Promise<TokenResponse>
  return service.post<TokenResponse, TokenResponse>('/v1/auth/login', params, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
  })
}

export const register = (data: RegisterData) => {
  return service.post<UserResponse, UserResponse>('/v1/auth/register', data)
}

export const getMe = () => {
  return service.get<UserResponse, UserResponse>('/v1/auth/me')
}

export const logout = () => {
  return service.post('/v1/auth/logout')
}
