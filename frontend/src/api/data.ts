import service from './index'
import type { Dataset, BatteryUnit } from './types'

/**
 * 获取数据集列表
 * @returns 数据集列表
 */
export const getDatasets = (): Promise<Dataset[]> => {
  return service.get<Dataset[], Dataset[]>('/v1/data/datasets')
}

/**
 * 获取指定数据集的电池列表
 * @param datasetId 数据集ID
 * @returns 电池列表
 */
export const getBatteries = (datasetId: number): Promise<BatteryUnit[]> => {
  return service.get<BatteryUnit[], BatteryUnit[]>(`/v1/data/datasets/${datasetId}/batteries`)
}
