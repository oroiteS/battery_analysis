import service from './index'
import type {
  CreateTestJobRequest,
  TestJobResponse,
  TestJobDetail,
  TestMetrics,
  TestPredictionsResponse,
  TestLogsResponse,
} from './types'

/**
 * 创建测试任务
 * @param request 测试任务请求
 * @returns 测试任务响应
 */
export const createTestJob = (request: CreateTestJobRequest): Promise<TestJobResponse> => {
  return service.post<TestJobResponse, TestJobResponse>('/v1/testing/jobs', request)
}

/**
 * 获取测试任务详情
 * @param jobId 任务ID
 * @returns 测试任务详情
 */
export const getTestJob = (jobId: number): Promise<TestJobDetail> => {
  return service.get<TestJobDetail, TestJobDetail>(`/v1/testing/jobs/${jobId}`)
}

/**
 * 获取测试任务列表
 * @param params 查询参数
 * @returns 测试任务列表
 */
export const listTestJobs = (params?: {
  status_filter?: string
  limit?: number
  offset?: number
}): Promise<TestJobResponse[]> => {
  return service.get<TestJobResponse[], TestJobResponse[]>('/v1/testing/jobs', { params })
}

/**
 * 删除测试任务
 * @param jobId 任务ID
 * @returns 删除结果
 */
export const deleteTestJob = (jobId: number): Promise<void> => {
  return service.delete<void, void>(`/v1/testing/jobs/${jobId}`)
}

/**
 * 获取测试指标
 * @param jobId 任务ID
 * @returns 测试指标
 */
export const getTestMetrics = (jobId: number): Promise<TestMetrics> => {
  return service.get<TestMetrics, TestMetrics>(`/v1/testing/jobs/${jobId}/metrics`)
}

/**
 * 获取测试预测结果
 * @param jobId 任务ID
 * @param batteryId 可选的电池ID筛选
 * @returns 测试预测结果
 */
export const getTestPredictions = (
  jobId: number,
  batteryId?: number,
): Promise<TestPredictionsResponse> => {
  return service.get<TestPredictionsResponse, TestPredictionsResponse>(
    `/v1/testing/jobs/${jobId}/predictions`,
    {
      params: batteryId ? { battery_id: batteryId } : undefined,
    },
  )
}

/**
 * 获取测试日志
 * @param jobId 任务ID
 * @param params 查询参数
 * @returns 测试日志
 */
export const getTestLogs = (
  jobId: number,
  params?: {
    level?: string
    limit?: number
  },
): Promise<TestLogsResponse> => {
  return service.get<TestLogsResponse, TestLogsResponse>(`/v1/testing/jobs/${jobId}/logs`, {
    params,
  })
}

/**
 * 导出测试结果
 * @param jobId 任务ID
 * @param format 导出格式 (CSV/XLSX)
 */
export const exportTestResults = async (
  jobId: number,
  format: 'CSV' | 'XLSX' = 'CSV',
): Promise<void> => {
  const response = await service.post(`/v1/testing/jobs/${jobId}/export`, null, {
    params: { format },
    responseType: 'blob',
  })

  // 创建下载链接
  const url = window.URL.createObjectURL(new Blob([response as unknown as BlobPart]))
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', `test_job_${jobId}_results.${format.toLowerCase()}`)
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
}

/**
 * 创建WebSocket连接以接收测试进度
 * @param jobId 任务ID
 * @returns WebSocket实例
 */
export const connectTestWebSocket = (jobId: number): WebSocket => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = import.meta.env.VITE_API_BASE_URL?.replace(/^https?:\/\//, '') || 'localhost:8000'
  const token = localStorage.getItem('token')
  const wsUrl = `${protocol}//${host}/api/v1/testing/ws/jobs/${jobId}?token=${token}`
  return new WebSocket(wsUrl)
}
