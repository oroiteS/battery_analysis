import service from './index'
import type {
  FeatureStat,
  TrendData,
  ScatterData,
  PCLDistribution,
  CorrelationMatrix
} from './types'

export const getBatteryStats = (batteryId: number) => {
  return service.get<any, FeatureStat[]>(`/v1/data/batteries/${batteryId}/stats`)
}

export const getTrendData = (batteryId: number, featureName: string) => {
  return service.get<any, TrendData>(`/v1/data/batteries/${batteryId}/trend`, {
    params: { feature_name: featureName }
  })
}

export const getScatterData = (batteryId: number, featureName: string) => {
  return service.get<any, ScatterData>(`/v1/data/batteries/${batteryId}/scatter`, {
    params: { feature_name: featureName }
  })
}

export const getPCLDistribution = (batteryId: number) => {
  return service.get<any, PCLDistribution>(`/v1/data/batteries/${batteryId}/pcl-distribution`)
}

export const getCorrelationMatrix = (batteryId: number) => {
  return service.get<any, CorrelationMatrix>(`/v1/data/batteries/${batteryId}/correlation-matrix`)
}

export const exportAnalysisReport = (batteryId: number) => {
  return service.get(`/v1/data/batteries/${batteryId}/export-report`, {
    responseType: 'blob'
  })
}
