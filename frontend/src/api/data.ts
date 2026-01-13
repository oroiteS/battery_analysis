import service from './index'
import type { Dataset, BatteryUnit } from './types'

export const getDatasets = () => {
  return service.get<any, Dataset[]>('/v1/data/datasets')
}

export const getBatteries = (datasetId: number) => {
  return service.get<any, BatteryUnit[]>(`/v1/data/datasets/${datasetId}/batteries`)
}
