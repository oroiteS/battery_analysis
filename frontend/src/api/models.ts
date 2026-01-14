import service from './index'
import type { AlgorithmListResponse, ModelVersion, ModelVersionDetail } from './types'

/**
 * 获取支持的算法列表
 * @returns 算法列表
 */
export const getAlgorithms = (): Promise<AlgorithmListResponse> => {
  return service.get<AlgorithmListResponse, AlgorithmListResponse>('/v1/models/algorithms')
}

/**
 * 获取模型版本列表
 * @param params 查询参数
 * @returns 模型版本列表
 */
export const getModelVersions = (params?: {
  algorithm?: string
  limit?: number
  offset?: number
}): Promise<ModelVersion[]> => {
  return service.get<ModelVersion[], ModelVersion[]>('/v1/models/versions', { params })
}

/**
 * 获取模型版本详情
 * @param versionId 模型版本ID
 * @returns 模型版本详情
 */
export const getModelVersion = (versionId: number): Promise<ModelVersionDetail> => {
  return service.get<ModelVersionDetail, ModelVersionDetail>(`/v1/models/versions/${versionId}`)
}

/**
 * 下载模型checkpoint
 * @param versionId 模型版本ID
 * @returns 下载URL
 */
export const downloadModelCheckpoint = (versionId: number): string => {
  return `/api/v1/models/versions/${versionId}/download`
}

/**
 * 删除模型版本
 * @param versionId 模型版本ID
 */
export const deleteModelVersion = (versionId: number): Promise<void> => {
  return service.delete<void, void>(`/v1/models/versions/${versionId}`)
}
