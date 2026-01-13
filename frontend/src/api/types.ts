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

// --- Training Types ---

export interface BatterySelection {
  battery_id: number
  split_role: 'train' | 'val' | 'test'
}

export interface CreateTrainingJobRequest {
  dataset_id: number
  target: 'RUL' | 'PCL' | 'BOTH'
  algorithms: string[]
  batteries: BatterySelection[]
  // Hyperparameters
  seq_len?: number
  perc_val?: number
  num_layers?: number[]
  num_neurons?: number[]
  num_epoch?: number
  batch_size?: number
  lr?: number
  dropout_rate?: number
  weight_decay?: number
  step_size?: number
  gamma?: number
  lr_scheduler?: string
  min_lr?: number
  grad_clip?: number
  early_stopping_patience?: number
  monitor_metric?: string
  num_rounds?: number
  random_seed?: number
  inputs_dynamical?: string
  inputs_dim_dynamical?: string
  loss_mode?: string
  loss_weights?: number[]
}

export interface TrainingJobResponse {
  id: number
  user_id: number
  dataset_id: number
  target: string
  hyperparams: Record<string, any>
  status: string
  progress: number
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface TrainingRunResponse {
  id: number
  job_id: number
  algorithm: string
  status: string
  current_epoch: number
  total_epochs: number
  started_at: string | null
  finished_at: string | null
}

export interface TrainingJobDetailResponse {
  job: TrainingJobResponse
  runs: TrainingRunResponse[]
  batteries: {
    battery_id: number
    battery_code: string
    split_role: string
    total_cycles: number
  }[]
}

export interface TrainingMetric {
  epoch: number
  train_loss: number
  val_loss: number
  metrics: Record<string, number>
}

export interface TrainingMetricsResponse {
  run_id: number
  algorithm: string
  metrics: TrainingMetric[]
}

export interface TrainingLog {
  timestamp: string
  level: string
  message: string
}

export interface TrainingLogsResponse {
  run_id: number
  log_file_path: string | null
  total_lines?: number
  returned_lines?: number
  logs: TrainingLog[]
  message?: string
}

// --- Models Types ---

export interface Algorithm {
  code: string
  name: string
  description: string
  supported: boolean
}

export interface AlgorithmListResponse {
  algorithms: Algorithm[]
}
