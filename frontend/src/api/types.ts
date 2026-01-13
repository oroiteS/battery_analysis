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

// --- Analysis Types ---

export interface FeatureStat {
  feature_name: string
  mean: number
  variance: number
  min_val: number
  max_val: number
  corr_rul: number | null
  corr_pcl: number | null
}

export interface TrendData {
  cycles: number[]
  values: number[]
}

export interface ScatterData {
  points: number[][] // [[x, y], [x, y]]
}

export interface PCLDistribution {
  pcl_values: number[]
}

export interface CorrelationMatrix {
  features: string[]
  matrix: number[][]
}

// --- Data Management Types (Needed for Battery Selection) ---
export interface BatteryUnit {
  id: number
  battery_code: string
  group_tag: string | null
  total_cycles: number
  nominal_capacity: number | null
}

export interface Dataset {
  id: number
  name: string
  source_type: string
  battery_count: number
  created_at: string
}
