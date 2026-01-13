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
  // 基础超参数
  seq_len?: number
  perc_val?: number
  num_layers?: number[]
  num_neurons?: number[]
  num_epoch?: number
  batch_size?: number
  lr?: number
  dropout_rate?: number
  weight_decay?: number
  // 学习率调度器参数
  lr_scheduler?: 'StepLR' | 'CosineAnnealing' | 'ReduceLROnPlateau'
  step_size?: number
  gamma?: number
  min_lr?: number
  // 训练优化参数
  grad_clip?: number
  early_stopping_patience?: number
  monitor_metric?: 'val_loss' | 'RMSPE'
  // 训练轮次参数
  num_rounds?: number
  random_seed?: number
  // DeepHPM特定参数
  inputs_dynamical?: string
  inputs_dim_dynamical?: string
  loss_mode?: 'Sum' | 'AdpBal' | 'Baseline'
  loss_weights?: number[]
  // BiLSTM特定参数
  hidden_dim?: number
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

export interface ModelVersion {
  id: number
  user_id: number
  run_id: number
  algorithm: string
  name: string
  version: string
  config: Record<string, any>
  metrics: Record<string, any>
  checkpoint_path: string
  created_at: string
}

export interface ModelVersionDetail {
  model: ModelVersion
  training_info: {
    run_id: number
    job_id: number
    algorithm: string
    status: string
    started_at: string | null
    finished_at: string | null
  } | null
}

// --- Testing Types ---

export interface CreateTestJobRequest {
  model_version_id: number
  dataset_id: number
  target: 'RUL' | 'PCL' | 'BOTH'
  battery_ids: number[]
  horizon: number
}

export interface TestJobResponse {
  id: number
  user_id: number
  model_version_id: number
  dataset_id: number
  target: string
  horizon: number
  status: string
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface TestJobDetail {
  job: TestJobResponse
  model_info: {
    id: number
    name: string
    version: string
    algorithm: string
  }
  batteries: {
    battery_id: number
    battery_code: string
    total_cycles: number
  }[]
}

export interface TestMetrics {
  job_id: number
  overall_metrics: {
    target: string
    metrics: Record<string, number>
  }[]
  battery_metrics: {
    battery_id: number
    target: string
    metrics: Record<string, number>
  }[]
}

export interface TestPrediction {
  battery_id: number
  cycle_num: number
  target: string
  y_true: number
  y_pred: number
}

export interface TestPredictionsResponse {
  job_id: number
  predictions: TestPrediction[]
}

export interface TestLog {
  timestamp: string
  level: string
  message: string
}

export interface TestLogsResponse {
  job_id: number
  log_file_path: string | null
  total_lines?: number
  returned_lines?: number
  logs: TestLog[]
  message?: string
}

export interface TestExportResponse {
  export_id: number
  status: string
  message: string
}
