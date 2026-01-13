import service from './index'
import type {
  FeatureStat,
  TrendData,
  ScatterData,
  PCLDistribution,
  CorrelationMatrix,
} from './types'

export const getBatteryStats = (batteryId: number) => {
  return service.get<any, FeatureStat[]>(`/v1/data/batteries/${batteryId}/stats`)
}

export const getTrendData = (batteryId: number, featureName: string) => {
  return service.get<any, TrendData>(`/v1/data/batteries/${batteryId}/trend`, {
    params: { feature_name: featureName },
  })
}

export const getScatterData = (batteryId: number, featureName: string) => {
  return service.get<any, ScatterData>(`/v1/data/batteries/${batteryId}/scatter`, {
    params: { feature_name: featureName },
  })
}

export const getPCLDistribution = (batteryId: number) => {
  return service.get<any, PCLDistribution>(`/v1/data/batteries/${batteryId}/pcl-distribution`)
}

export const getCorrelationMatrix = (batteryId: number) => {
  return service.get<any, CorrelationMatrix>(`/v1/data/batteries/${batteryId}/correlation-matrix`)
}

/**
 * 导出分析报告
 * @param datasetId 数据集ID
 * @param batteryId 电池ID
 * @param format 导出格式 (XLSX/PDF)
 */
export const exportAnalysisReport = async (
  datasetId: number,
  batteryId: number,
  format: 'XLSX' | 'PDF' = 'XLSX',
): Promise<void> => {
  const response = await service.get('/v1/data/batteries/export', {
    params: { dataset_id: datasetId, battery_id: batteryId, format },
    responseType: 'blob',
  })

  // 创建下载链接
  const url = window.URL.createObjectURL(new Blob([response as unknown as BlobPart]))
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', `analysis_report_${batteryId}.${format.toLowerCase()}`)
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
}
