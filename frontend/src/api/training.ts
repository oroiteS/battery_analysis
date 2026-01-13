import service from './index'
import type {
  CreateTrainingJobRequest,
  TrainingJobResponse,
  TrainingJobDetailResponse,
  TrainingMetricsResponse,
  TrainingLogsResponse
} from './types'

export const createTrainingJob = (data: CreateTrainingJobRequest) => {
  return service.post<any, TrainingJobResponse>('/v1/training/jobs', data)
}

export const listTrainingJobs = (params?: { status_filter?: string; limit?: number; offset?: number }) => {
  return service.get<any, TrainingJobResponse[]>('/v1/training/jobs', { params })
}

export const getTrainingJob = (jobId: number) => {
  return service.get<any, TrainingJobDetailResponse>(`/v1/training/jobs/${jobId}`)
}

export const deleteTrainingJob = (jobId: number) => {
  return service.delete(`/v1/training/jobs/${jobId}`)
}

export const getTrainingMetrics = (jobId: number, runId: number) => {
  return service.get<any, TrainingMetricsResponse>(`/v1/training/jobs/${jobId}/runs/${runId}/metrics`)
}

export const getTrainingLogs = (jobId: number, runId: number, params?: { level?: string; limit?: number }) => {
  return service.get<any, TrainingLogsResponse>(`/v1/training/jobs/${jobId}/runs/${runId}/logs`, { params })
}
