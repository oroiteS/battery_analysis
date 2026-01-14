<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, nextTick, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import type { EChartsOption } from 'echarts'
import { getDatasets, getBatteries } from '@/api/data'
import { getModelVersions } from '@/api/models'
import {
  createTestJob,
  getTestJob,
  getTestMetrics,
  getTestPredictions,
  exportTestResults,
  connectTestWebSocket,
  listTestJobs,
  deleteTestJob as deleteTestJobAPI,
} from '@/api/testing'
import type {
  Dataset,
  BatteryUnit,
  ModelVersion,
  TestJobResponse,
  TestMetrics,
  TestPrediction,
} from '@/api/types'

defineOptions({
  name: 'PredictionView',
})

// 主标签页
const activeTab = ref('config') // 'config' | 'monitor'
const route = useRoute()
const lastOpenedJobId = ref<number | null>(null)

// 测试任务列表
const testJobs = ref<TestJobResponse[]>([])
const jobsLoading = ref(false)

// 详情弹窗
const detailVisible = ref(false)
const detailJobId = ref<number | null>(null)

// 详情数据（独立于新建任务表单）
const detailJob = ref<TestJobResponse | null>(null)
const detailMetrics = ref<TestMetrics | null>(null)
const detailPredictions = ref<TestPrediction[]>([])
const detailTarget = ref<'RUL' | 'PCL' | 'BOTH'>('BOTH')

// 表单数据
const testForm = ref({
  datasetId: undefined as number | undefined,
  batteryIds: [] as number[],
  modelVersionId: undefined as number | undefined,
  target: 'BOTH' as 'RUL' | 'PCL' | 'BOTH',
  horizon: 1,
})

// 选项数据
const datasets = ref<Dataset[]>([])
const batteries = ref<BatteryUnit[]>([])
const modelVersions = ref<ModelVersion[]>([])

// 状态
const isTesting = ref(false)
const currentJobId = ref<number | null>(null)
const testJob = ref<TestJobResponse | null>(null)
const testProgress = ref(0)
const testLogs = ref<string[]>([])

// 结果数据
const hasResult = ref(false)
const metrics = ref<TestMetrics | null>(null)
const predictions = ref<TestPrediction[]>([])

// WebSocket
let ws: WebSocket | null = null

// ECharts实例
const rulChartRef = ref<HTMLDivElement>()
const pclChartRef = ref<HTMLDivElement>()
let rulChart: echarts.ECharts | null = null
let pclChart: echarts.ECharts | null = null
const detailRulChartRef = ref<HTMLDivElement>()
const detailPclChartRef = ref<HTMLDivElement>()
let detailRulChart: echarts.ECharts | null = null
let detailPclChart: echarts.ECharts | null = null

// 计算属性
const selectedBatteries = computed(() => {
  return batteries.value.filter((b) => testForm.value.batteryIds.includes(b.id))
})

const modelOptions = computed(() => {
  return modelVersions.value.map((m) => {
    // 格式化创建时间 - 转换为上海时区（UTC+8）
    const date = new Date(m.created_at)
    const dateStr = date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      timeZone: 'Asia/Shanghai',
    })

    // 算法名称映射
    const algoNames: Record<string, string> = {
      BASELINE: 'Baseline',
      BILSTM: 'BiLSTM',
      DEEPHPM: 'DeepHPM',
    }
    const algoName = algoNames[m.algorithm] || m.algorithm

    // 训练目标显示
    const targetStr = m.target

    // 提取关键指标
    const metrics = m.metrics
    let metricsStr = ''
    if (metrics.RMSPE !== undefined) {
      metricsStr = `RMSPE: ${metrics.RMSPE.toFixed(2)}%`
    } else if (metrics.val_loss !== undefined) {
      metricsStr = `Loss: ${metrics.val_loss.toFixed(4)}`
    }

    return {
      value: m.id,
      label: `${algoName} | ${dateStr} | ${targetStr}`,
    }
  })
})

// 加载数据集
const loadDatasets = async () => {
  try {
    datasets.value = await getDatasets()
    if (datasets.value.length > 0 && datasets.value[0]) {
      testForm.value.datasetId = datasets.value[0].id
      await loadBatteries()
    }
  } catch {
    ElMessage.error('加载数据集失败')
  }
}

// 加载电池列表
const loadBatteries = async () => {
  if (!testForm.value.datasetId) return
  try {
    batteries.value = await getBatteries(testForm.value.datasetId)
    // 默认选择test组的电池
    testForm.value.batteryIds = batteries.value
      .filter((b) => b.group_tag === 'test')
      .map((b) => b.id)
      .slice(0, 5) // 默认选择前5个
  } catch {
    ElMessage.error('加载电池列表失败')
  }
}

// 加载模型版本
const loadModelVersions = async () => {
  try {
    modelVersions.value = await getModelVersions({ limit: 100 })
  } catch {
    ElMessage.error('加载模型版本失败')
  }
}

// 加载测试任务列表
const loadTestJobs = async () => {
  try {
    jobsLoading.value = true
    testJobs.value = await listTestJobs({ limit: 10 })
  } catch {
    ElMessage.error('加载测试任务列表失败')
  } finally {
    jobsLoading.value = false
  }
}

// 查看测试任务详情
const viewTestJobDetail = async (jobId: number) => {
  try {
    const job = await getTestJob(jobId)
    detailJobId.value = jobId
    detailJob.value = job.job
    detailTarget.value = job.job.target as 'RUL' | 'PCL' | 'BOTH'

    // 如果任务已完成，加载结果
    if (job.job.status === 'SUCCEEDED') {
      // 打开弹窗
      detailVisible.value = true

      // 等待DOM更新
      await nextTick()

      // 加载详情结果数据
      await loadDetailResults(jobId)
    } else {
      ElMessage.warning('任务尚未完成')
    }
  } catch {
    ElMessage.error('加载任务详情失败')
  }
}

// 加载详情结果数据
const loadDetailResults = async (jobId: number) => {
  try {
    // 加载指标
    detailMetrics.value = await getTestMetrics(jobId)

    // 加载预测结果
    const predRes = await getTestPredictions(jobId)
    detailPredictions.value = predRes.predictions

    // 渲染图表
    await nextTick()
    renderDetailCharts()
  } catch {
    ElMessage.error('加载测试结果失败')
  }
}

// 渲染详情图表
const renderDetailCharts = () => {
  // 使用setTimeout确保DOM完全渲染
  setTimeout(() => {
    if (detailTarget.value === 'RUL' || detailTarget.value === 'BOTH') {
      renderDetailRULChart()
    }
    if (detailTarget.value === 'PCL' || detailTarget.value === 'BOTH') {
      renderDetailPCLChart()
    }

    requestAnimationFrame(() => {
      detailRulChart?.resize()
      detailPclChart?.resize()
    })
  }, 100)
}

// 渲染详情RUL图表
const renderDetailRULChart = () => {
  if (!detailRulChartRef.value) {
    console.warn('RUL chart container not found')
    return
  }

  if (!detailRulChart) {
    detailRulChart = echarts.init(detailRulChartRef.value)
  } else {
    detailRulChart.clear()
  }

  const rulPredictions = detailPredictions.value.filter((p) => p.target === 'RUL')

  // 按电池分组
  const batteryGroups = new Map<number, TestPrediction[]>()
  rulPredictions.forEach((p) => {
    if (!batteryGroups.has(p.battery_id)) {
      batteryGroups.set(p.battery_id, [])
    }
    batteryGroups.get(p.battery_id)!.push(p)
  })

  const series: echarts.SeriesOption[] = []

  // 为每个电池创建真实值和预测值系列
  batteryGroups.forEach((preds, batteryId) => {
    const batteryName = `Battery ${batteryId}`

    preds.sort((a, b) => a.cycle_num - b.cycle_num)

    series.push({
      name: `${batteryName} (真实)`,
      type: 'line',
      data: preds.map((p) => [p.cycle_num, p.y_true]),
      lineStyle: { type: 'solid' },
    })

    series.push({
      name: `${batteryName} (预测)`,
      type: 'line',
      data: preds.map((p) => [p.cycle_num, p.y_pred]),
      lineStyle: { type: 'dashed' },
    })
  })

  const option: EChartsOption = {
    title: { text: 'RUL 预测结果' },
    tooltip: { trigger: 'axis' },
    legend: { type: 'scroll', bottom: 0 },
    xAxis: { type: 'value', name: 'Cycle' },
    yAxis: { type: 'value', name: 'RUL' },
    series,
  }

  detailRulChart.setOption(option)
}

// 渲染详情PCL图表
const renderDetailPCLChart = () => {
  if (!detailPclChartRef.value) {
    console.warn('PCL chart container not found')
    return
  }

  if (!detailPclChart) {
    detailPclChart = echarts.init(detailPclChartRef.value)
  } else {
    detailPclChart.clear()
  }

  const pclPredictions = detailPredictions.value.filter((p) => p.target === 'PCL')

  // 按电池分组
  const batteryGroups = new Map<number, TestPrediction[]>()
  pclPredictions.forEach((p) => {
    if (!batteryGroups.has(p.battery_id)) {
      batteryGroups.set(p.battery_id, [])
    }
    batteryGroups.get(p.battery_id)!.push(p)
  })

  const series: echarts.SeriesOption[] = []

  // 为每个电池创建真实值和预测值系列
  batteryGroups.forEach((preds, batteryId) => {
    const batteryName = `Battery ${batteryId}`

    preds.sort((a, b) => a.cycle_num - b.cycle_num)

    series.push({
      name: `${batteryName} (真实)`,
      type: 'line',
      data: preds.map((p) => [p.cycle_num, p.y_true]),
      lineStyle: { type: 'solid' },
    })

    series.push({
      name: `${batteryName} (预测)`,
      type: 'line',
      data: preds.map((p) => [p.cycle_num, p.y_pred]),
      lineStyle: { type: 'dashed' },
    })
  })

  const option: EChartsOption = {
    title: { text: 'PCL 预测结果' },
    tooltip: { trigger: 'axis' },
    legend: { type: 'scroll', bottom: 0 },
    xAxis: { type: 'value', name: 'Cycle' },
    yAxis: { type: 'value', name: 'PCL' },
    series,
  }

  detailPclChart.setOption(option)
}

// 删除测试任务
const deleteTestJob = async (jobId: number) => {
  try {
    await deleteTestJobAPI(jobId)
    // 从列表中移除
    testJobs.value = testJobs.value.filter((j) => j.id !== jobId)
    ElMessage.success('删除成功')
  } catch {
    ElMessage.error('删除失败')
  }
}

// 格式化日期
const formatDate = (dateStr: string) => {
  return new Date(dateStr).toLocaleString('zh-CN')
}

// 标签页切换处理
const handleTabChange = (tabName: string | number) => {
  if (tabName === 'monitor') {
    // 切换到任务监控时刷新列表
    loadTestJobs()
  }
}

// 开始测试
const handleStartTest = async () => {
  if (!testForm.value.datasetId) {
    ElMessage.warning('请选择数据集')
    return
  }
  if (testForm.value.batteryIds.length === 0) {
    ElMessage.warning('请至少选择一个电池')
    return
  }
  if (!testForm.value.modelVersionId) {
    ElMessage.warning('请选择模型版本')
    return
  }

  try {
    if (rulChart) {
      rulChart.dispose()
      rulChart = null
    }
    if (pclChart) {
      pclChart.dispose()
      pclChart = null
    }
    isTesting.value = true
    hasResult.value = false
    testLogs.value = []
    testProgress.value = 0

    // 创建测试任务
    const job = await createTestJob({
      model_version_id: testForm.value.modelVersionId,
      dataset_id: testForm.value.datasetId,
      target: testForm.value.target,
      battery_ids: testForm.value.batteryIds,
      horizon: testForm.value.horizon,
    })

    currentJobId.value = job.id
    testJob.value = job

    ElMessage.success('测试任务已创建，正在执行...')

    // 连接WebSocket接收实时进度
    connectWebSocket(job.id)

    // 轮询任务状态
    pollJobStatus(job.id)
  } catch (error: unknown) {
    const err = error as { response?: { data?: { detail?: string } } }
    ElMessage.error(err.response?.data?.detail || '创建测试任务失败')
    isTesting.value = false
  }
}

// 连接WebSocket
const connectWebSocket = (jobId: number) => {
  try {
    ws = connectTestWebSocket(jobId)

    ws.onopen = () => {
      console.log('WebSocket connected')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      handleWebSocketMessage(data)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onclose = () => {
      console.log('WebSocket closed')
    }
  } catch (error) {
    console.error('Failed to connect WebSocket:', error)
  }
}

// 处理WebSocket消息
const handleWebSocketMessage = (data: Record<string, unknown>) => {
  console.log('收到WebSocket消息:', data)

  const messageType = typeof data.type === 'string' ? data.type : ''
  const payload =
    data.data && typeof data.data === 'object' ? (data.data as Record<string, unknown>) : data

  if (messageType === 'log') {
    const level = payload.level ?? data.level
    const message = payload.message ?? data.message
    if (level && message) {
      testLogs.value.push(`[${level}] ${message}`)
    }
  } else if (messageType === 'progress') {
    const rawProgress = payload.progress ?? data.progress
    const progressValue = typeof rawProgress === 'number' ? rawProgress : Number(rawProgress)
    const normalized =
      Number.isFinite(progressValue) && progressValue <= 1 ? progressValue * 100 : progressValue
    const clamped = Number.isFinite(normalized) ? Math.min(100, Math.max(0, normalized)) : 0
    console.log('更新进度:', clamped)
    testProgress.value = clamped
  } else if (messageType === 'status_change' || messageType === 'status') {
    const status = payload.new_status ?? payload.status ?? data.status
    console.log('任务状态变更:', status)
    if (status === 'SUCCEEDED') {
      ElMessage.success('测试完成')
      const jobId = payload.test_job_id ?? data.test_job_id ?? currentJobId.value
      void finalizeTest(jobId)
    } else if (status === 'FAILED') {
      isTesting.value = false
      ElMessage.error('测试失败')
    }
  }
}

// 轮询任务状态
const pollJobStatus = async (jobId: number) => {
  const maxAttempts = 300 // 最多轮询5分钟
  let attempts = 0

  const poll = async () => {
    if (attempts >= maxAttempts) {
      ElMessage.error('测试超时')
      isTesting.value = false
      return
    }

    try {
      const job = await getTestJob(jobId)
      testJob.value = job.job

      if (job.job.status === 'SUCCEEDED') {
        await finalizeTest(jobId)
      } else if (job.job.status === 'FAILED') {
        isTesting.value = false
        ElMessage.error('测试失败')
      } else {
        attempts++
        setTimeout(poll, 1000)
      }
    } catch (error) {
      console.error('Failed to poll job status:', error)
      attempts++
      setTimeout(poll, 1000)
    }
  }

  poll()
}

const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

const finalizeTest = async (jobId: number | null | undefined) => {
  if (!jobId) {
    isTesting.value = false
    return
  }

  currentJobId.value = jobId
  testProgress.value = 100
  try {
    await loadTestResults(jobId)
  } finally {
    isTesting.value = false
  }
}

// 加载测试结果
const loadTestResults = async (jobId?: number) => {
  const resolvedJobId = jobId ?? currentJobId.value
  if (!resolvedJobId) return

  try {
    // 加载指标
    metrics.value = await getTestMetrics(resolvedJobId)
    console.log('测试结果指标:', metrics.value?.overall_metrics)

    // 加载预测结果
    let predRes = await getTestPredictions(resolvedJobId)
    let attempts = 0
    while (predRes.predictions.length === 0 && attempts < 2) {
      await wait(500)
      predRes = await getTestPredictions(resolvedJobId)
      attempts++
    }
    predictions.value = predRes.predictions
    const targetCounts = predRes.predictions.reduce(
      (acc, pred) => {
        const key = pred.target
        acc[key] = (acc[key] ?? 0) + 1
        return acc
      },
      {} as Record<string, number>,
    )
    console.log('测试结果预测数量:', {
      total: predRes.predictions.length,
      targets: targetCounts,
      sample: predRes.predictions[0],
    })

    hasResult.value = true

    // 渲染图表
    await nextTick()
    renderCharts()
  } catch {
    ElMessage.error('加载测试结果失败')
  }
}

// 渲染图表
const renderCharts = () => {
  // 使用setTimeout确保DOM完全渲染，并增加重试机制
  let retryCount = 0
  const maxRetries = 3

  const tryRender = () => {
    if (retryCount >= maxRetries) {
      console.warn('图表渲染失败：超过最大重试次数')
      return
    }

    const isRulTarget = testForm.value.target === 'RUL' || testForm.value.target === 'BOTH'
    const isPclTarget = testForm.value.target === 'PCL' || testForm.value.target === 'BOTH'
    const rulReady =
      !!rulChartRef.value && rulChartRef.value.clientWidth > 0 && rulChartRef.value.clientHeight > 0
    const pclReady =
      !!pclChartRef.value && pclChartRef.value.clientWidth > 0 && pclChartRef.value.clientHeight > 0

    if (isRulTarget && !rulReady) {
      console.warn('RUL图表容器未准备好，重试中...', retryCount + 1)
      retryCount++
      setTimeout(tryRender, 200)
      return
    }

    if (isPclTarget && !pclReady) {
      console.warn('PCL图表容器未准备好，重试中...', retryCount + 1)
      retryCount++
      setTimeout(tryRender, 200)
      return
    }

    if (isRulTarget) {
      renderRULChart()
    }
    if (isPclTarget) {
      renderPCLChart()
    }

    console.log('图表容器尺寸:', {
      rul: rulChartRef.value
        ? [rulChartRef.value.clientWidth, rulChartRef.value.clientHeight]
        : null,
      pcl: pclChartRef.value
        ? [pclChartRef.value.clientWidth, pclChartRef.value.clientHeight]
        : null,
    })

    requestAnimationFrame(() => {
      rulChart?.resize()
      pclChart?.resize()
    })
  }

  setTimeout(tryRender, 100)
}

// 渲染RUL图表
const renderRULChart = () => {
  if (!rulChartRef.value) {
    console.warn('RUL chart container not found')
    return
  }

  if (!rulChart) {
    rulChart = echarts.init(rulChartRef.value)
  } else {
    // 清空旧图表
    rulChart.clear()
  }

  const rulPredictions = predictions.value.filter((p) => p.target === 'RUL')

  // 按电池分组
  const batteryGroups = new Map<number, TestPrediction[]>()
  rulPredictions.forEach((p) => {
    if (!batteryGroups.has(p.battery_id)) {
      batteryGroups.set(p.battery_id, [])
    }
    batteryGroups.get(p.battery_id)!.push(p)
  })

  const series: echarts.SeriesOption[] = []

  // 为每个电池创建真实值和预测值系列
  batteryGroups.forEach((preds, batteryId) => {
    const battery = batteries.value.find((b) => b.id === batteryId)
    const batteryName = battery?.battery_code || `Battery ${batteryId}`

    preds.sort((a, b) => a.cycle_num - b.cycle_num)

    series.push({
      name: `${batteryName} (真实)`,
      type: 'line',
      data: preds.map((p) => [p.cycle_num, p.y_true]),
      lineStyle: { type: 'solid' },
    })

    series.push({
      name: `${batteryName} (预测)`,
      type: 'line',
      data: preds.map((p) => [p.cycle_num, p.y_pred]),
      lineStyle: { type: 'dashed' },
    })
  })

  const option: EChartsOption = {
    title: { text: 'RUL 预测结果' },
    tooltip: { trigger: 'axis' },
    legend: { type: 'scroll', bottom: 0 },
    xAxis: { type: 'value', name: 'Cycle' },
    yAxis: { type: 'value', name: 'RUL' },
    series,
  }

  rulChart.setOption(option)
}

// 渲染PCL图表
const renderPCLChart = () => {
  if (!pclChartRef.value) {
    console.warn('PCL chart container not found')
    return
  }

  if (!pclChart) {
    pclChart = echarts.init(pclChartRef.value)
  } else {
    // 清空旧图表
    pclChart.clear()
  }

  const pclPredictions = predictions.value.filter((p) => p.target === 'PCL')

  // 按电池分组
  const batteryGroups = new Map<number, TestPrediction[]>()
  pclPredictions.forEach((p) => {
    if (!batteryGroups.has(p.battery_id)) {
      batteryGroups.set(p.battery_id, [])
    }
    batteryGroups.get(p.battery_id)!.push(p)
  })

  const series: echarts.SeriesOption[] = []

  // 为每个电池创建真实值和预测值系列
  batteryGroups.forEach((preds, batteryId) => {
    const battery = batteries.value.find((b) => b.id === batteryId)
    const batteryName = battery?.battery_code || `Battery ${batteryId}`

    preds.sort((a, b) => a.cycle_num - b.cycle_num)

    series.push({
      name: `${batteryName} (真实)`,
      type: 'line',
      data: preds.map((p) => [p.cycle_num, p.y_true]),
      lineStyle: { type: 'solid' },
    })

    series.push({
      name: `${batteryName} (预测)`,
      type: 'line',
      data: preds.map((p) => [p.cycle_num, p.y_pred]),
      lineStyle: { type: 'dashed' },
    })
  })

  const option: EChartsOption = {
    title: { text: 'PCL 预测结果' },
    tooltip: { trigger: 'axis' },
    legend: { type: 'scroll', bottom: 0 },
    xAxis: { type: 'value', name: 'Cycle' },
    yAxis: { type: 'value', name: 'PCL' },
    series,
  }

  pclChart.setOption(option)
}

// 导出结果
const handleExport = async (format: 'CSV' | 'XLSX') => {
  if (!currentJobId.value) return

  try {
    await exportTestResults(currentJobId.value, format)
    ElMessage.success(`${format}文件下载成功`)
  } catch {
    ElMessage.error('导出失败')
  }
}

const downloadChartImage = (chart: echarts.ECharts | null, filename: string, pixelRatio = 2) => {
  if (!chart) {
    ElMessage.warning('图表尚未生成')
    return
  }
  const dataUrl = chart.getDataURL({
    type: 'png',
    pixelRatio,
    backgroundColor: '#ffffff',
  })
  const link = document.createElement('a')
  link.href = dataUrl
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

const buildChartFilename = (prefix: string, jobId: number | null) => {
  return jobId ? `${prefix}_job_${jobId}.png` : `${prefix}.png`
}

const downloadResultChart = (target: 'RUL' | 'PCL') => {
  const chart = target === 'RUL' ? rulChart : pclChart
  const filename = buildChartFilename(`test_${target.toLowerCase()}_chart`, currentJobId.value)
  downloadChartImage(chart, filename)
}

const downloadDetailChart = (target: 'RUL' | 'PCL') => {
  const chart = target === 'RUL' ? detailRulChart : detailPclChart
  const filename = buildChartFilename(`detail_${target.toLowerCase()}_chart`, detailJobId.value)
  downloadChartImage(chart, filename)
}

// 生命周期
onMounted(() => {
  const init = async () => {
    await Promise.all([loadDatasets(), loadModelVersions(), loadTestJobs()])
    await openJobFromRoute()
  }

  void init()
})

onUnmounted(() => {
  if (ws) {
    ws.close()
  }
  if (rulChart) {
    rulChart.dispose()
    rulChart = null
  }
  if (pclChart) {
    pclChart.dispose()
    pclChart = null
  }
  if (detailRulChart) {
    detailRulChart.dispose()
  }
  if (detailPclChart) {
    detailPclChart.dispose()
  }
})

const openJobFromRoute = async () => {
  const jobIdParam = route.query.jobId
  const jobIdValue = Array.isArray(jobIdParam) ? jobIdParam[0] : jobIdParam
  const jobId = jobIdValue ? Number(jobIdValue) : NaN
  if (!Number.isFinite(jobId) || jobId === lastOpenedJobId.value) {
    return
  }

  lastOpenedJobId.value = jobId
  activeTab.value = 'monitor'
  await viewTestJobDetail(jobId)
}

watch(
  () => route.query.jobId,
  () => {
    void openJobFromRoute()
  },
)
</script>

<template>
  <div class="prediction-container">
    <!-- 标签页 -->
    <el-tabs v-model="activeTab" class="main-tabs" @tab-change="handleTabChange">
      <!-- 新建测试任务 -->
      <el-tab-pane label="新建测试任务" name="config">
        <!-- 配置表单 -->
        <el-card shadow="hover" class="config-card">
          <template #header>
            <span>测试配置</span>
          </template>
          <el-form :model="testForm" label-width="120px">
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="数据集">
                  <el-select
                    v-model="testForm.datasetId"
                    placeholder="选择数据集"
                    @change="loadBatteries"
                    style="width: 100%"
                  >
                    <el-option
                      v-for="ds in datasets"
                      :key="ds.id"
                      :label="ds.name"
                      :value="ds.id"
                    />
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="预测模型">
                  <el-select
                    v-model="testForm.modelVersionId"
                    placeholder="选择模型版本"
                    filterable
                    style="width: 100%"
                  >
                    <el-option
                      v-for="item in modelOptions"
                      :key="item.value"
                      :label="item.label"
                      :value="item.value"
                    />
                  </el-select>
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item label="待测试电池">
              <el-select
                v-model="testForm.batteryIds"
                placeholder="选择电池（可多选）"
                multiple
                filterable
                style="width: 100%"
              >
                <el-option
                  v-for="battery in batteries"
                  :key="battery.id"
                  :label="`${battery.battery_code} (${battery.group_tag || 'N/A'})`"
                  :value="battery.id"
                />
              </el-select>
            </el-form-item>

            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="预测目标">
                  <el-radio-group v-model="testForm.target">
                    <el-radio label="RUL">RUL</el-radio>
                    <el-radio label="PCL">PCL</el-radio>
                    <el-radio label="BOTH">RUL+PCL</el-radio>
                  </el-radio-group>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="预测步长">
                  <el-input-number v-model="testForm.horizon" :min="1" :max="100" :step="1" />
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item>
              <el-button
                type="primary"
                icon="VideoPlay"
                :loading="isTesting"
                @click="handleStartTest"
              >
                开始测试
              </el-button>
              <el-text v-if="selectedBatteries.length > 0" type="info" style="margin-left: 10px">
                已选择 {{ selectedBatteries.length }} 个电池
              </el-text>
            </el-form-item>
          </el-form>
        </el-card>

        <!-- 测试进度 -->
        <el-card v-if="isTesting" shadow="hover" class="mt-20">
          <template #header>
            <span>测试进度</span>
          </template>
          <el-progress :percentage="testProgress" :status="testProgress === 100 ? 'success' : ''" />
          <div v-if="testLogs.length > 0" class="log-container">
            <div v-for="(log, index) in testLogs.slice(-10)" :key="index" class="log-line">
              {{ log }}
            </div>
          </div>
        </el-card>

        <!-- 测试结果 -->
        <div v-if="hasResult" class="result-area mt-20">
          <!-- 指标卡片 -->
          <el-row :gutter="20" class="mb-20">
            <el-col
              v-for="(metric, index) in metrics?.overall_metrics"
              :key="index"
              :span="metrics!.overall_metrics.length === 1 ? 24 : 12"
            >
              <el-card shadow="hover">
                <template #header>
                  <span>{{ metric.target }} 整体指标</span>
                </template>
                <el-descriptions :column="2" border>
                  <el-descriptions-item
                    v-for="(value, key) in metric.metrics"
                    :key="key"
                    :label="key"
                  >
                    {{ typeof value === 'number' ? value.toFixed(4) : value }}
                  </el-descriptions-item>
                </el-descriptions>
              </el-card>
            </el-col>
          </el-row>

          <!-- 预测曲线 -->
          <el-row :gutter="20">
            <el-col
              v-if="testForm.target === 'RUL' || testForm.target === 'BOTH'"
              :span="testForm.target === 'BOTH' ? 12 : 24"
            >
              <el-card shadow="hover">
                <template #header>
                  <span>RUL 预测曲线</span>
                  <el-button type="primary" link @click="downloadResultChart('RUL')">
                    下载 PNG
                  </el-button>
                </template>
                <div ref="rulChartRef" style="height: 400px"></div>
              </el-card>
            </el-col>
            <el-col
              v-if="testForm.target === 'PCL' || testForm.target === 'BOTH'"
              :span="testForm.target === 'BOTH' ? 12 : 24"
            >
              <el-card shadow="hover">
                <template #header>
                  <span>PCL 预测曲线</span>
                  <el-button type="primary" link @click="downloadResultChart('PCL')">
                    下载 PNG
                  </el-button>
                </template>
                <div ref="pclChartRef" style="height: 400px"></div>
              </el-card>
            </el-col>
          </el-row>

          <!-- 导出按钮 -->
          <el-row class="mt-20">
            <el-col :span="24" style="text-align: right">
              <el-button type="success" icon="Download" @click="handleExport('CSV')">
                导出 CSV
              </el-button>
              <el-button type="success" icon="Download" @click="handleExport('XLSX')">
                导出 XLSX
              </el-button>
            </el-col>
          </el-row>
        </div>

        <!-- 空状态 -->
        <el-empty v-else-if="!isTesting" description="请配置测试参数并开始测试" class="mt-20" />
      </el-tab-pane>

      <!-- 任务监控 -->
      <el-tab-pane label="任务监控" name="monitor">
        <el-table :data="testJobs" v-loading="jobsLoading" style="width: 100%">
          <el-table-column prop="id" label="任务ID" width="80" />
          <el-table-column prop="created_at" label="创建时间" width="180">
            <template #default="scope">
              {{ formatDate(scope.row.created_at) }}
            </template>
          </el-table-column>
          <el-table-column prop="model_name" label="模型" width="250">
            <template #default="scope">
              <el-tag type="success">{{ scope.row.model_name || 'N/A' }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="target" label="目标" width="100">
            <template #default="scope">
              <el-tag>{{ scope.row.target }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="status" label="状态" width="120">
            <template #default="scope">
              <el-tag
                :type="
                  scope.row.status === 'SUCCEEDED'
                    ? 'success'
                    : scope.row.status === 'RUNNING'
                      ? 'primary'
                      : scope.row.status === 'FAILED'
                        ? 'danger'
                        : 'info'
                "
              >
                {{ scope.row.status }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="180">
            <template #default="scope">
              <el-button type="primary" link @click="viewTestJobDetail(scope.row.id)">
                查看结果
              </el-button>
              <el-button type="danger" link @click="deleteTestJob(scope.row.id)"> 删除 </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
    </el-tabs>

    <!-- 测试结果详情弹窗 -->
    <el-dialog
      v-model="detailVisible"
      title="测试结果详情"
      width="90%"
      :close-on-click-modal="false"
    >
      <div v-if="detailMetrics" class="result-area">
        <!-- 指标卡片 -->
        <el-row :gutter="20" class="mb-20">
          <el-col
            v-for="(metric, index) in detailMetrics?.overall_metrics"
            :key="index"
            :span="detailMetrics!.overall_metrics.length === 1 ? 24 : 12"
          >
            <el-card shadow="hover">
              <template #header>
                <span>{{ metric.target }} 整体指标</span>
              </template>
              <el-descriptions :column="2" border>
                <el-descriptions-item
                  v-for="(value, key) in metric.metrics"
                  :key="key"
                  :label="key"
                >
                  {{ typeof value === 'number' ? value.toFixed(4) : value }}
                </el-descriptions-item>
              </el-descriptions>
            </el-card>
          </el-col>
        </el-row>

        <!-- 预测曲线 -->
        <el-row :gutter="20">
          <el-col
            v-if="detailTarget === 'RUL' || detailTarget === 'BOTH'"
            :span="detailTarget === 'BOTH' ? 12 : 24"
          >
            <el-card shadow="hover">
              <template #header>
                <span>RUL 预测曲线</span>
                <el-button type="primary" link @click="downloadDetailChart('RUL')">
                  下载 PNG
                </el-button>
              </template>
              <div ref="detailRulChartRef" style="height: 400px"></div>
            </el-card>
          </el-col>
          <el-col
            v-if="detailTarget === 'PCL' || detailTarget === 'BOTH'"
            :span="detailTarget === 'BOTH' ? 12 : 24"
          >
            <el-card shadow="hover">
              <template #header>
                <span>PCL 预测曲线</span>
                <el-button type="primary" link @click="downloadDetailChart('PCL')">
                  下载 PNG
                </el-button>
              </template>
              <div ref="detailPclChartRef" style="height: 400px"></div>
            </el-card>
          </el-col>
        </el-row>

        <!-- 导出按钮 -->
        <el-row class="mt-20">
          <el-col :span="24" style="text-align: right">
            <el-button type="success" icon="Download" @click="handleExport('CSV')">
              导出 CSV
            </el-button>
            <el-button type="success" icon="Download" @click="handleExport('XLSX')">
              导出 XLSX
            </el-button>
          </el-col>
        </el-row>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.prediction-container {
  padding: 20px;
}

.mt-20 {
  margin-top: 20px;
}

.mb-20 {
  margin-bottom: 20px;
}

.log-container {
  margin-top: 15px;
  padding: 10px;
  background: #f5f7fa;
  border-radius: 4px;
  max-height: 200px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 12px;
}

.log-line {
  padding: 2px 0;
  color: #606266;
}
</style>
