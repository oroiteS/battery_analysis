import service from './index'
import type { AlgorithmListResponse } from './types'

export const getAlgorithms = () => {
  return service.get<any, AlgorithmListResponse>('/v1/models/algorithms')
}
