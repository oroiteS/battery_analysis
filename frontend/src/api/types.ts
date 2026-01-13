export interface LoginData {
  username: string
  password: string
}

export interface RegisterData {
  user_name: string
  email: string
  password: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface UserResponse {
  id: number
  user_name: string
  email: string
}
