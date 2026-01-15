import service from './index'
import type {
  CreateTrainingJobRequest,
  TrainingJobResponse,
  TrainingJobDetailResponse,
  TrainingMetricsResponse,
  TrainingLogsResponse,
} from './types'

/**
 * 创建训练任务
 * @param data 训练任务配置
 * @returns 创建的训练任务信息
 */
export const createTrainingJob = (data: CreateTrainingJobRequest): Promise<TrainingJobResponse> => {
  return service.post<TrainingJobResponse, TrainingJobResponse>('/v1/training/jobs', data)
}

/**
 * 获取训练任务列表
 * @param params 查询参数（状态过滤、分页）
 * @returns 训练任务列表
 */
export const listTrainingJobs = (params?: {
  status_filter?: string
  limit?: number
  offset?: number
}): Promise<TrainingJobResponse[]> => {
  return service.get<TrainingJobResponse[], TrainingJobResponse[]>('/v1/training/jobs', {
    params,
  })
}

/**
 * 获取训练任务详情
 * @param jobId 任务ID
 * @returns 训练任务详细信息（包含runs和batteries）
 */
export const getTrainingJob = (jobId: number): Promise<TrainingJobDetailResponse> => {
  return service.get<TrainingJobDetailResponse, TrainingJobDetailResponse>(
    `/v1/training/jobs/${jobId}`,
  )
}

/**
 * 删除训练任务
 * @param jobId 任务ID
 */
export const deleteTrainingJob = (jobId: number): Promise<void> => {
  return service.delete<void, void>(`/v1/training/jobs/${jobId}`)
}

/**
 * 获取训练指标
 * @param jobId 任务ID
 * @param runId 运行ID
 * @returns 训练指标数据
 */
export const getTrainingMetrics = (
  jobId: number,
  runId: number,
): Promise<TrainingMetricsResponse> => {
  return service.get<TrainingMetricsResponse, TrainingMetricsResponse>(
    `/v1/training/jobs/${jobId}/runs/${runId}/metrics`,
  )
}

/**
 * 获取训练日志
 * @param jobId 任务ID
 * @param runId 运行ID
 * @param params 查询参数（日志级别、数量限制）
 * @returns 训练日志数据
 */
export const getTrainingLogs = (
  jobId: number,
  runId: number,
  params?: { level?: string; limit?: number },
): Promise<TrainingLogsResponse> => {
  return service.get<TrainingLogsResponse, TrainingLogsResponse>(
    `/v1/training/jobs/${jobId}/runs/${runId}/logs`,
    {
      params,
    },
  )
}

/**
 * 下载训练日志文件
 * @param jobId 任务ID
 * @param runId 运行ID
 * @returns 日志文件Blob
 */
export const downloadTrainingJobLogs = (jobId: number, runId: number): Promise<Blob> => {
  return service.get<Blob, Blob>(`/v1/training/jobs/${jobId}/runs/${runId}/logs/download`, {
    responseType: 'blob',
    timeout: 60000, // 60秒超时，适用于日志文件下载
  })
}

/**
 * 获取WebSocket连接URL
 * @param jobId 任务ID
 * @returns WebSocket URL
 */
export const getTrainingWebSocketUrl = (jobId: number): string => {
  const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'
  const wsUrl = baseUrl.replace(/^http/, 'ws').replace(/\/api$/, '')
  return `${wsUrl}/api/v1/training/ws/jobs/${jobId}`
}
