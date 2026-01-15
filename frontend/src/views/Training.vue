<script setup lang="ts">
import { ref, reactive, onMounted, watch, onUnmounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'
import { getDatasets, getBatteries } from '../api/data'
import { getAlgorithms } from '../api/models'
import {
  createTrainingJob,
  listTrainingJobs,
  getTrainingJob,
  deleteTrainingJob,
  getTrainingWebSocketUrl,
  getTrainingMetrics,
  getTrainingLogs,
  downloadTrainingJobLogs,
} from '../api/training'
import type {
  Dataset,
  BatteryUnit,
  Algorithm,
  CreateTrainingJobRequest,
  BatterySelection,
  TrainingJobResponse,
  TrainingJobDetailResponse,
  TrainingLog,
  TrainingRunResponse,
} from '../api/types'

defineOptions({
  name: 'TrainingView',
})

// --- State ---
const datasets = ref<Dataset[]>([])
const batteries = ref<BatteryUnit[]>([])
const algorithms = ref<Algorithm[]>([])
const trainingJobs = ref<TrainingJobResponse[]>([])

const form = reactive({
  datasetId: null as number | null,
  target: 'RUL',
  selectedAlgorithms: [] as string[],
  batteries: [] as BatterySelection[],
  // 基础超参数
  num_epoch: 2000,
  batch_size: 1024,
  lr: 0.001,
  dropout_rate: 0.2,
  seq_len: 1,
  perc_val: 0.2,
  splitRatio: 0.8, // Train ratio (Train / (Train + Val))
  num_layers: [2],
  num_neurons: [128],
  // 学习率调度器参数
  lr_scheduler: 'StepLR' as 'StepLR' | 'CosineAnnealing' | 'ReduceLROnPlateau',
  step_size: 50000,
  gamma: 0.1,
  min_lr: 0.000001,
  // 训练优化参数
  weight_decay: 0.0,
  grad_clip: 0.0,
  early_stopping_patience: 0,
  monitor_metric: 'val_loss' as 'val_loss' | 'RMSPE',
  // 训练轮次参数
  num_rounds: 5,
  random_seed: 1234,
  // DeepHPM特定参数
  inputs_dynamical: 's_norm, t_norm',
  inputs_dim_dynamical: 'inputs_dim',
  loss_mode: 'Sum' as 'Sum' | 'AdpBal' | 'Baseline',
  loss_weights: [1.0, 1.0, 1.0],
  // BiLSTM特定参数
  hidden_dim: 128,
})

// Battery Selection Table Data
interface BatteryRow extends BatteryUnit {
  role: 'train' | 'val' | 'test' | 'ignore'
}
const batteryTableData = ref<BatteryRow[]>([])

const isSubmitting = ref(false)
const activeTab = ref('config') // 'config' | 'monitor'
const activeCollapse = ref(['basic']) // 默认展开基础训练参数

// 数组参数的字符串输入
const numLayersInput = ref('[2]')
const numNeuronsInput = ref('[128]')
const lossWeightsInput = ref('[1.0, 1.0, 1.0]')

// 监听字符串输入，解析为数组
watch(numLayersInput, (val) => {
  try {
    const parsed = JSON.parse(val)
    if (Array.isArray(parsed)) {
      form.num_layers = parsed
    }
  } catch {
    // 忽略解析错误
  }
})

watch(numNeuronsInput, (val) => {
  try {
    const parsed = JSON.parse(val)
    if (Array.isArray(parsed)) {
      form.num_neurons = parsed
    }
  } catch {
    // 忽略解析错误
  }
})

watch(lossWeightsInput, (val) => {
  try {
    const parsed = JSON.parse(val)
    if (Array.isArray(parsed)) {
      form.loss_weights = parsed
    }
  } catch {
    // 忽略解析错误
  }
})

// --- Detail Dialog ---
const detailVisible = ref(false)
const currentJob = ref<TrainingJobDetailResponse | null>(null)
const detailActiveTab = ref('overview')
const logs = ref<TrainingLog[]>([])
const chartInstance = ref<echarts.ECharts | null>(null)
const chartRef = ref<HTMLElement | null>(null)
const isLogScale = ref(true) // 默认开启对数坐标，避免Loss差异过大导致挤压
const selectedRunId = ref<number | null>(null) // 当前选中的算法run ID

// --- WebSocket ---
const wsMap = new Map<number, WebSocket>()
const route = useRoute()
const lastOpenedJobId = ref<number | null>(null)

const connectWebSocket = (jobId: number) => {
  const existing = wsMap.get(jobId)
  if (existing) {
    if (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING) {
      return
    }
    wsMap.delete(jobId)
  }

  const wsUrl = getTrainingWebSocketUrl(jobId)
  const ws = new WebSocket(wsUrl)
  wsMap.set(jobId, ws)

  ws.onopen = () => {
    console.log(`WebSocket connected for job ${jobId}`)
  }

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data)

    // Update list view
    const jobIndex = trainingJobs.value.findIndex((j) => j.id === jobId)
    if (jobIndex !== -1) {
      const job = trainingJobs.value[jobIndex]
      if (job && message.type === 'job_progress') {
        // Force reactivity update - 修复：使用message.data.progress
        trainingJobs.value[jobIndex] = {
          ...job,
          progress: message.data?.progress ?? job.progress,
          status: message.data?.status ?? job.status,
        }
      }
      if (job && message.type === 'job_status_change') {
        const newStatus = message.data?.new_status ?? message.status
        trainingJobs.value[jobIndex] = {
          ...job,
          status: newStatus,
          // 任务完成时强制进度为1.0 (表示100%)
          progress: newStatus === 'SUCCEEDED' || newStatus === 'FAILED' ? 1.0 : job.progress,
        }
        if (newStatus === 'SUCCEEDED' || newStatus === 'FAILED') {
          ws.close()
        }
      }
    }

    // Update detail view if open
    if (detailVisible.value && currentJob.value?.job.id === jobId) {
      handleDetailMessage(message)
    }
  }

  ws.onerror = (error) => {
    console.error(`WebSocket error for job ${jobId}:`, error)
    ElMessage.error('WebSocket连接错误')
  }

  ws.onclose = () => {
    console.log(`WebSocket disconnected for job ${jobId}`)
    wsMap.delete(jobId)
  }
}

const connectRunningJobs = (jobs: TrainingJobResponse[]) => {
  jobs.forEach((job) => {
    if (job.status === 'RUNNING') {
      connectWebSocket(job.id)
    }
  })
}

const closeWebSocket = (jobId: number) => {
  const ws = wsMap.get(jobId)
  if (ws) {
    ws.close()
    wsMap.delete(jobId)
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const handleDetailMessage = (message: any) => {
  switch (message.type) {
    case 'log':
      {
        const newLog = {
          timestamp: message.data?.timestamp || new Date().toISOString(),
          level: message.data?.level || 'INFO',
          message: message.data?.message || message.message,
        }

        // 检查是否已存在（通过时间戳+消息去重）
        const logKey = `${newLog.timestamp}_${newLog.message}`
        const exists = logs.value.some((l) => `${l.timestamp}_${l.message}` === logKey)

        if (!exists) {
          logs.value.push(newLog)
          // Auto scroll to bottom
          nextTick(() => {
            const container = document.querySelector('.log-container')
            if (container) container.scrollTop = container.scrollHeight
          })
        }
      }
      break
    case 'epoch_progress':
      // Update chart with multiple loss metrics
      if (chartInstance.value && message.data) {
        const data = message.data

        // 仅更新选中算法的数据
        if (data.run_id !== selectedRunId.value) {
          break
        }

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const option = chartInstance.value.getOption() as any

        // 检查该epoch是否已存在（避免重复添加）
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const existingEpoch = option.series[0]?.data?.find((point: any) => point[0] === data.epoch)
        if (existingEpoch) {
          // 数据已存在，跳过添加
          break
        }

        // 基础损失（总是存在）
        if (data.train_loss !== undefined) {
          option.series[0].data.push([data.epoch, data.train_loss])
        }
        if (data.val_loss !== undefined) {
          option.series[1].data.push([data.epoch, data.val_loss])
        }

        // 额外的损失分量（如果存在）
        if (data.loss_U !== undefined && option.series[2]) {
          option.series[2].data.push([data.epoch, data.loss_U])
        }
        if (data.loss_F !== undefined && option.series[3]) {
          option.series[3].data.push([data.epoch, data.loss_F])
        }
        if (data.loss_F_t !== undefined && option.series[4]) {
          option.series[4].data.push([data.epoch, data.loss_F_t])
        }

        chartInstance.value.setOption(option)
      }

      // Update run status in overview
      if (currentJob.value && currentJob.value.runs.length > 0 && message.data) {
        // Find the run by algorithm
        const runningRun = currentJob.value.runs.find((r) => r.status === 'RUNNING')
        if (runningRun) {
          runningRun.current_epoch = message.data.epoch
        }
      }
      break
    case 'training_detail':
      // Period级别的详细损失信息（可选：显示在日志中）
      if (message.data) {
        const detail = message.data
        logs.value.push({
          timestamp: new Date().toISOString(),
          level: 'DEBUG',
          message: `Epoch ${detail.epoch}, Period ${detail.period}: Loss=${detail.loss?.toFixed(5)}, Loss_U=${detail.loss_U?.toFixed(5)}, Loss_F=${detail.loss_F?.toFixed(5)}, Loss_F_t=${detail.loss_F_t?.toFixed(5)}`,
        })
      }
      break
    case 'run_status_change':
      // 更新run状态
      if (currentJob.value && message.data) {
        const run = currentJob.value.runs.find((r) => r.id === message.data.run_id)
        if (run) {
          run.status = message.data.new_status
          if (message.data.new_status === 'SUCCEEDED' || message.data.new_status === 'FAILED') {
            run.finished_at = new Date().toISOString()
          }
        }
      }
      break
    case 'job_status_change':
      // 更新job状态
      if (currentJob.value && message.data) {
        currentJob.value.job.status = message.data.new_status
        if (message.data.new_status === 'SUCCEEDED' || message.data.new_status === 'FAILED') {
          currentJob.value.job.finished_at = new Date().toISOString()
        }
      }
      break
    case 'job_progress':
      // 更新job进度
      if (currentJob.value && message.data) {
        currentJob.value.job.progress = message.data.progress
        if (message.data.status) {
          currentJob.value.job.status = message.data.status
        }
      }
      break
    default:
      console.warn('Unknown message type:', message.type)
  }
}

// --- Methods ---

const initData = async () => {
  try {
    const [datasetsRes, algosRes, jobsRes] = await Promise.all([
      getDatasets(),
      getAlgorithms(),
      listTrainingJobs({ limit: 10 }),
    ])
    datasets.value = datasetsRes
    algorithms.value = algosRes.algorithms

    // 修正已完成任务的进度显示
    trainingJobs.value = jobsRes.map((job) => ({
      ...job,
      progress: job.status === 'SUCCEEDED' || job.status === 'FAILED' ? 1.0 : job.progress,
    }))

    if (datasets.value.length > 0 && datasets.value[0]) {
      form.datasetId = datasets.value[0].id
      await handleDatasetChange()
    }

    // Connect to all running jobs
    connectRunningJobs(trainingJobs.value)
  } catch (error) {
    console.error('Init failed:', error)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ElMessage.error((error as any)?.response?.data?.detail || '初始化失败')
  }
}

const handleDatasetChange = async () => {
  if (!form.datasetId) return
  try {
    const res = await getBatteries(form.datasetId)
    batteries.value = res
    updateBatteryRoles()
  } catch (error) {
    console.error('Fetch batteries failed:', error)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ElMessage.error((error as any)?.response?.data?.detail || '获取电池列表失败')
  }
}

const updateBatteryRoles = () => {
  const total = batteries.value.length
  if (total === 0) return

  const testCount = Math.floor(total * 0.2)
  const remaining = total - testCount
  const trainCount = Math.floor(remaining * form.splitRatio)

  batteryTableData.value = batteries.value.map((b, index) => {
    let role: 'train' | 'val' | 'test' = 'test'
    if (index < trainCount) role = 'train'
    else if (index < trainCount + (remaining - trainCount)) role = 'val'
    else role = 'test'
    return { ...b, role }
  })
}

watch(
  () => form.splitRatio,
  () => {
    updateBatteryRoles()
  },
)

const handleSubmit = async () => {
  if (!form.datasetId) {
    ElMessage.warning('请选择数据集')
    return
  }
  if (form.selectedAlgorithms.length === 0) {
    ElMessage.warning('请至少选择一个算法')
    return
  }

  const selectedBatteries = batteryTableData.value
    .filter((b) => b.role !== 'ignore')
    .map((b) => ({ battery_id: b.id, split_role: b.role as 'train' | 'val' | 'test' }))

  if (selectedBatteries.length === 0) {
    ElMessage.warning('请至少选择一个电池用于训练')
    return
  }

  isSubmitting.value = true
  try {
    const payload: CreateTrainingJobRequest = {
      dataset_id: form.datasetId,
      target: form.target as 'RUL' | 'PCL' | 'BOTH',
      algorithms: form.selectedAlgorithms,
      batteries: selectedBatteries,
      // 基础超参数
      num_epoch: form.num_epoch,
      batch_size: form.batch_size,
      lr: form.lr,
      dropout_rate: form.dropout_rate,
      seq_len: form.seq_len,
      perc_val: 1 - form.splitRatio,
      num_layers: form.num_layers,
      num_neurons: form.num_neurons,
      // 学习率调度器参数
      lr_scheduler: form.lr_scheduler,
      step_size: form.step_size,
      gamma: form.gamma,
      min_lr: form.min_lr,
      // 训练优化参数
      weight_decay: form.weight_decay,
      grad_clip: form.grad_clip,
      early_stopping_patience: form.early_stopping_patience,
      monitor_metric: form.monitor_metric,
      // 训练轮次参数
      num_rounds: form.num_rounds,
      random_seed: form.random_seed,
      // DeepHPM特定参数
      inputs_dynamical: form.inputs_dynamical,
      inputs_dim_dynamical: form.inputs_dim_dynamical,
      loss_mode: form.loss_mode,
      loss_weights: form.loss_weights,
      // BiLSTM特定参数
      hidden_dim: form.hidden_dim,
    }

    const newJob = await createTrainingJob(payload)
    ElMessage.success(`训练任务 #${newJob.id} 创建成功`)

    const jobsRes = await listTrainingJobs({ limit: 10 })
    trainingJobs.value = jobsRes
    activeTab.value = 'monitor'

    connectWebSocket(newJob.id)
    connectRunningJobs(trainingJobs.value)
  } catch (error) {
    console.error('Create job failed:', error)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ElMessage.error((error as any)?.response?.data?.detail || '创建任务失败')
  } finally {
    isSubmitting.value = false
  }
}

const showDetail = async (jobId: number) => {
  try {
    const res = await getTrainingJob(jobId)
    currentJob.value = res
    detailVisible.value = true
    logs.value = [] // Clear logs

    // 设置默认选中第一个算法
    if (res.runs && res.runs.length > 0) {
      selectedRunId.value = res.runs[0]?.id ?? null
    }

    // If job is running, ensure WS is connected
    if (res.job.status === 'RUNNING') {
      connectWebSocket(jobId)
    }

    // Init Chart after dialog opens
    await nextTick()
    initChart()

    // Load historical metrics for all runs
    await loadHistoricalMetrics(jobId, res.runs)

    // Load historical logs for all runs
    await loadHistoricalLogs(jobId, res.runs)
  } catch (error) {
    console.error('Get job detail failed:', error)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ElMessage.error((error as any)?.response?.data?.detail || '获取任务详情失败')
  }
}

// 加载历史指标数据
const loadHistoricalMetrics = async (jobId: number, runs: TrainingRunResponse[]) => {
  if (!chartInstance.value || !selectedRunId.value) return

  // 仅加载选中算法的数据
  const selectedRun = runs.find((r) => r.id === selectedRunId.value)
  if (!selectedRun) return

  try {
    const metricsRes = await getTrainingMetrics(jobId, selectedRun.id)
    const metrics = metricsRes.metrics

    if (metrics && metrics.length > 0) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const option = chartInstance.value.getOption() as any

      // 清空旧数据
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      option.series.forEach((s: any) => {
        s.data = []
      })

      // 批量添加历史数据到图表
      metrics.forEach((m) => {
        const epoch = m.epoch + 1 // epoch从0开始，显示时+1

        // 添加基础损失
        if (m.train_loss !== undefined && option.series[0]) {
          option.series[0].data.push([epoch, m.train_loss])
        }
        if (m.val_loss !== undefined && option.series[1]) {
          option.series[1].data.push([epoch, m.val_loss])
        }

        // 添加额外的损失分量（如果存在）
        if (m.metrics) {
          if (m.metrics.loss_U !== undefined && option.series[2]) {
            option.series[2].data.push([epoch, m.metrics.loss_U])
          }
          if (m.metrics.loss_F !== undefined && option.series[3]) {
            option.series[3].data.push([epoch, m.metrics.loss_F])
          }
          if (m.metrics.loss_F_t !== undefined && option.series[4]) {
            option.series[4].data.push([epoch, m.metrics.loss_F_t])
          }
        }
      })

      chartInstance.value.setOption(option)
    }
  } catch (error) {
    console.error(`Failed to load metrics for run ${selectedRun.id}:`, error)
  }
}

// 加载历史日志数据
const loadHistoricalLogs = async (jobId: number, runs: TrainingRunResponse[]) => {
  // 按运行顺序加载日志（按创建时间排序）
  const sortedRuns = [...runs].sort((a, b) => {
    const timeA = a.started_at ? new Date(a.started_at).getTime() : 0
    const timeB = b.started_at ? new Date(b.started_at).getTime() : 0
    return timeA - timeB || a.id - b.id
  })

  for (const run of sortedRuns) {
    try {
      const logsRes = await getTrainingLogs(jobId, run.id, { limit: 1000 })

      if (logsRes.logs && logsRes.logs.length > 0) {
        // 批量添加历史日志
        logsRes.logs.forEach((log) => {
          // 检查是否已存在（通过时间戳+消息去重）
          const logKey = `${log.timestamp}_${log.message}`
          const exists = logs.value.some((l) => `${l.timestamp}_${l.message}` === logKey)

          if (!exists) {
            logs.value.push({
              timestamp: log.timestamp || new Date().toISOString(),
              level: log.level || 'INFO',
              message: log.message || '',
            })
          }
        })

        // 自动滚动到底部
        await nextTick()
        const container = document.querySelector('.log-container')
        if (container) {
          container.scrollTop = container.scrollHeight
        }
      }
    } catch (error) {
      console.error(`Failed to load logs for run ${run.id}:`, error)
    }
  }
}

const handleDelete = async (jobId: number) => {
  try {
    await ElMessageBox.confirm(`确定要删除训练任务 #${jobId} 吗？`, '警告', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
    })

    await deleteTrainingJob(jobId)
    ElMessage.success('删除成功')

    // Refresh list
    trainingJobs.value = trainingJobs.value.filter((j) => j.id !== jobId)

    // Close WebSocket if connected to this job
    closeWebSocket(jobId)
  } catch (err) {
    // If user clicks cancel, error will be 'cancel'
    if (err !== 'cancel') {
      console.error('Delete job failed:', err)
      const error = err as { response?: { data?: { detail?: string } } }
      ElMessage.error(error?.response?.data?.detail || '删除失败')
    }
  }
}

const handleDownloadLogs = async () => {
  if (!currentJob.value) return

  const jobId = currentJob.value.job.id
  const runId = selectedRunId.value
  if (!runId) {
    ElMessage.warning('请先选择算法')
    return
  }

  try {
    const response = await downloadTrainingJobLogs(jobId, runId)
    const url = window.URL.createObjectURL(new Blob([response]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', `training_job_${jobId}_run_${runId}.log`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  } catch {
    ElMessage.error('下载日志失败')
  }
}

const handleDialogOpened = () => {
  // Dialog 动画结束后初始化图表，确保容器尺寸正确
  if (detailActiveTab.value === 'metrics') {
    initChart()
  }
}

const handleDialogClosed = () => {
  // Dialog 关闭时销毁图表实例，防止持有无效的 DOM 引用
  if (chartInstance.value) {
    chartInstance.value.dispose()
    chartInstance.value = null
  }
  window.removeEventListener('resize', resizeChart)
}

const resizeChart = () => {
  chartInstance.value?.resize()
}

const initChart = () => {
  if (!chartRef.value) return

  // 销毁旧实例
  if (chartInstance.value) {
    chartInstance.value.dispose()
  }

  chartInstance.value = echarts.init(chartRef.value)
  chartInstance.value.setOption({
    title: {
      text: '训练损失曲线 (Training Loss)',
      left: 'center',
      textStyle: {
        fontSize: 16,
        fontWeight: 'normal',
      },
    },
    toolbox: {
      feature: {
        saveAsImage: { title: '下载为图片' },
      },
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
      },
      formatter: (arg: unknown) => {
        const params = arg as { value: number[]; marker: string; seriesName: string }[]
        if (!params || params.length === 0 || !params[0]?.value) return ''
        let result = `Epoch ${params[0].value[0]}<br/>`
        params.forEach((param) => {
          if (param.value && param.value[1] !== undefined) {
            result += `${param.marker} ${param.seriesName}: ${param.value[1].toFixed(6)}<br/>`
          }
        })
        return result
      },
    },
    legend: {
      data: ['Train Loss', 'Val Loss', 'Loss_U', 'Loss_F', 'Loss_F_t'],
      top: 30,
      selected: {
        'Train Loss': true,
        'Val Loss': true,
        Loss_U: false,
        Loss_F: false,
        Loss_F_t: false,
      },
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '10%',
      top: '15%',
      containLabel: true,
    },
    xAxis: {
      type: 'value',
      name: 'Epoch',
      nameLocation: 'middle',
      nameGap: 25,
      minInterval: 1,
    },
    yAxis: {
      type: isLogScale.value ? 'log' : 'value',
      name: 'Loss',
      nameLocation: 'end',
      scale: true,
      min: isLogScale.value
        ? (value: { min: number }) => (value.min > 0 ? value.min * 0.9 : null)
        : null,
      axisLabel: {
        formatter: (value: number) => value.toExponential(2),
      },
    },
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 100,
        xAxisIndex: 0,
      },
      {
        start: 0,
        end: 100,
        xAxisIndex: 0,
        bottom: 10,
      },
    ],
    series: [
      {
        name: 'Train Loss',
        type: 'line',
        data: [],
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2 },
        emphasis: { focus: 'series' },
      },
      {
        name: 'Val Loss',
        type: 'line',
        data: [],
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2 },
        emphasis: { focus: 'series' },
      },
      {
        name: 'Loss_U',
        type: 'line',
        data: [],
        smooth: true,
        symbol: 'none',
        lineStyle: { type: 'dashed', width: 1.5 },
        emphasis: { focus: 'series' },
      },
      {
        name: 'Loss_F',
        type: 'line',
        data: [],
        smooth: true,
        symbol: 'none',
        lineStyle: { type: 'dashed', width: 1.5 },
        emphasis: { focus: 'series' },
      },
      {
        name: 'Loss_F_t',
        type: 'line',
        data: [],
        smooth: true,
        symbol: 'none',
        lineStyle: { type: 'dashed', width: 1.5 },
        emphasis: { focus: 'series' },
      },
    ],
  })

  // 监听窗口大小变化
  window.removeEventListener('resize', resizeChart)
  window.addEventListener('resize', resizeChart)
}

watch(isLogScale, (newVal) => {
  if (chartInstance.value) {
    chartInstance.value.setOption({
      yAxis: {
        type: newVal ? 'log' : 'value',
        min: newVal ? (value: { min: number }) => (value.min > 0 ? value.min * 0.9 : null) : null,
      },
    })
  }
})

// 监听算法选择变化，重新加载数据
watch(selectedRunId, async (newRunId) => {
  if (newRunId && currentJob.value) {
    await loadHistoricalMetrics(currentJob.value.job.id, currentJob.value.runs)
  }
})

// 监听Tab切换，确保图表正确resize
watch(detailActiveTab, (newTab) => {
  if (newTab === 'metrics') {
    nextTick(() => {
      if (!chartInstance.value) {
        initChart()
      } else {
        chartInstance.value.resize()
      }
    })
  }
})

const getStatusType = (status: string) => {
  switch (status) {
    case 'SUCCEEDED':
      return 'success'
    case 'FAILED':
      return 'danger'
    case 'RUNNING':
      return 'primary'
    default:
      return 'info'
  }
}

const formatLogTime = (timestamp: string) => {
  if (!timestamp) return ''
  try {
    const date = new Date(timestamp)
    const hours = String(date.getHours()).padStart(2, '0')
    const minutes = String(date.getMinutes()).padStart(2, '0')
    const seconds = String(date.getSeconds()).padStart(2, '0')
    const ms = String(date.getMilliseconds()).padStart(3, '0')
    return `${hours}:${minutes}:${seconds}.${ms}`
  } catch {
    return timestamp
  }
}

onMounted(() => {
  const init = async () => {
    await initData()
    await openJobFromRoute()
  }

  void init()
})

onUnmounted(() => {
  window.removeEventListener('resize', resizeChart)
  wsMap.forEach((socket) => socket.close())
  wsMap.clear()
  if (chartInstance.value) {
    chartInstance.value.dispose()
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
  await showDetail(jobId)
}

watch(
  () => route.query.jobId,
  () => {
    void openJobFromRoute()
  },
)
</script>

<template>
  <div class="training-container">
    <el-tabs v-model="activeTab" type="border-card">
      <!-- Tab 1: Configuration -->
      <el-tab-pane label="新建训练任务" name="config">
        <el-row :gutter="20">
          <!-- Left: Basic Config -->
          <el-col :span="12">
            <el-card shadow="never" header="基础配置">
              <el-form :model="form" label-width="100px">
                <el-form-item label="数据集">
                  <el-select
                    v-model="form.datasetId"
                    style="width: 100%"
                    @change="handleDatasetChange"
                  >
                    <el-option v-for="d in datasets" :key="d.id" :label="d.name" :value="d.id" />
                  </el-select>
                </el-form-item>

                <el-form-item label="预测目标">
                  <el-radio-group v-model="form.target">
                    <el-radio label="RUL">RUL (剩余寿命)</el-radio>
                    <el-radio label="PCL">PCL (预测容量)</el-radio>
                  </el-radio-group>
                </el-form-item>

                <el-form-item label="选择算法">
                  <el-checkbox-group v-model="form.selectedAlgorithms">
                    <el-checkbox v-for="algo in algorithms" :key="algo.code" :label="algo.code">
                      {{ algo.name }}
                    </el-checkbox>
                  </el-checkbox-group>
                </el-form-item>

                <el-form-item label="数据集划分">
                  <div class="slider-container">
                    <span class="slider-label"
                      >训练集占比 ({{ (form.splitRatio * 100).toFixed(0) }}%)</span
                    >
                    <el-slider
                      v-model="form.splitRatio"
                      :step="0.05"
                      :min="0.5"
                      :max="0.9"
                      show-input
                      :format-tooltip="(val: number) => `${(val * 100).toFixed(0)}%`"
                    />
                  </div>
                  <div class="slider-hint">剩余部分将自动分配给验证集。测试集固定保留 20%。</div>
                </el-form-item>
              </el-form>
            </el-card>

            <el-card shadow="never" header="超参数设置" class="mt-20">
              <el-form :model="form" label-width="140px">
                <el-collapse v-model="activeCollapse">
                  <!-- 基础训练参数 -->
                  <el-collapse-item title="基础训练参数" name="basic">
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Epochs">
                          <el-input-number v-model="form.num_epoch" :min="1" :max="10000" />
                        </el-form-item>
                      </el-col>
                      <el-col :span="12">
                        <el-form-item label="Batch Size">
                          <el-select v-model="form.batch_size" style="width: 100%">
                            <el-option :value="32" label="32" />
                            <el-option :value="64" label="64" />
                            <el-option :value="128" label="128" />
                            <el-option :value="256" label="256" />
                            <el-option :value="1024" label="1024" />
                          </el-select>
                        </el-form-item>
                      </el-col>
                    </el-row>
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Learning Rate">
                          <el-input-number
                            v-model="form.lr"
                            :step="0.0001"
                            :min="0.00001"
                            :max="1"
                          />
                        </el-form-item>
                      </el-col>
                      <el-col :span="12">
                        <el-form-item label="Dropout Rate">
                          <el-input-number
                            v-model="form.dropout_rate"
                            :step="0.05"
                            :min="0"
                            :max="0.9"
                          />
                        </el-form-item>
                      </el-col>
                    </el-row>
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Weight Decay">
                          <el-input-number
                            v-model="form.weight_decay"
                            :step="0.0001"
                            :min="0"
                            :max="1"
                          />
                        </el-form-item>
                      </el-col>
                      <el-col :span="12">
                        <el-form-item label="Num Rounds">
                          <el-input-number v-model="form.num_rounds" :min="1" :max="10" />
                        </el-form-item>
                      </el-col>
                    </el-row>
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Random Seed">
                          <el-input-number v-model="form.random_seed" :min="0" />
                        </el-form-item>
                      </el-col>
                    </el-row>
                  </el-collapse-item>

                  <!-- 模型架构参数 -->
                  <el-collapse-item title="模型架构参数" name="architecture">
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Num Layers">
                          <el-input v-model="numLayersInput" placeholder="如: [2] 或 [2,3]" />
                        </el-form-item>
                      </el-col>
                      <el-col :span="12">
                        <el-form-item label="Num Neurons">
                          <el-input v-model="numNeuronsInput" placeholder="如: [128] 或 [64,128]" />
                        </el-form-item>
                      </el-col>
                    </el-row>
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Hidden Dim (BiLSTM)">
                          <el-input-number v-model="form.hidden_dim" :min="16" :max="512" />
                        </el-form-item>
                      </el-col>
                      <el-col :span="12">
                        <el-form-item label="Sequence Length">
                          <el-input-number v-model="form.seq_len" :min="1" :max="100" />
                        </el-form-item>
                      </el-col>
                    </el-row>
                  </el-collapse-item>

                  <!-- 学习率调度器参数 -->
                  <el-collapse-item title="学习率调度器参数" name="scheduler">
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="LR Scheduler">
                          <el-select v-model="form.lr_scheduler" style="width: 100%">
                            <el-option value="StepLR" label="StepLR" />
                            <el-option value="CosineAnnealing" label="CosineAnnealing" />
                            <el-option value="ReduceLROnPlateau" label="ReduceLROnPlateau" />
                          </el-select>
                        </el-form-item>
                      </el-col>
                      <el-col :span="12">
                        <el-form-item label="Step Size">
                          <el-input-number v-model="form.step_size" :min="1" :max="100000" />
                        </el-form-item>
                      </el-col>
                    </el-row>
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Gamma">
                          <el-input-number v-model="form.gamma" :step="0.1" :min="0.01" :max="1" />
                        </el-form-item>
                      </el-col>
                      <el-col :span="12">
                        <el-form-item label="Min LR">
                          <el-input-number
                            v-model="form.min_lr"
                            :step="0.000001"
                            :min="0"
                            :max="0.01"
                          />
                        </el-form-item>
                      </el-col>
                    </el-row>
                  </el-collapse-item>

                  <!-- 训练优化参数 -->
                  <el-collapse-item title="训练优化参数" name="optimization">
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Gradient Clip">
                          <el-input-number
                            v-model="form.grad_clip"
                            :step="0.1"
                            :min="0"
                            :max="10"
                          />
                        </el-form-item>
                      </el-col>
                      <el-col :span="12">
                        <el-form-item label="Early Stop Patience">
                          <el-input-number
                            v-model="form.early_stopping_patience"
                            :min="0"
                            :max="100"
                          />
                        </el-form-item>
                      </el-col>
                    </el-row>
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Monitor Metric">
                          <el-select v-model="form.monitor_metric" style="width: 100%">
                            <el-option value="val_loss" label="Validation Loss" />
                            <el-option value="RMSPE" label="RMSPE" />
                          </el-select>
                        </el-form-item>
                      </el-col>
                    </el-row>
                  </el-collapse-item>

                  <!-- DeepHPM特定参数 -->
                  <el-collapse-item title="DeepHPM特定参数" name="deephpm">
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Inputs Dynamical">
                          <el-input v-model="form.inputs_dynamical" placeholder="s_norm, t_norm" />
                        </el-form-item>
                      </el-col>
                      <el-col :span="12">
                        <el-form-item label="Inputs Dim Dynamical">
                          <el-input v-model="form.inputs_dim_dynamical" placeholder="inputs_dim" />
                        </el-form-item>
                      </el-col>
                    </el-row>
                    <el-row :gutter="20">
                      <el-col :span="12">
                        <el-form-item label="Loss Mode">
                          <el-select v-model="form.loss_mode" style="width: 100%">
                            <el-option value="Sum" label="Sum" />
                            <el-option value="AdpBal" label="AdpBal" />
                            <el-option value="Baseline" label="Baseline" />
                          </el-select>
                        </el-form-item>
                      </el-col>
                      <el-col :span="12">
                        <el-form-item label="Loss Weights">
                          <el-input v-model="lossWeightsInput" placeholder="[1.0, 1.0, 1.0]" />
                        </el-form-item>
                      </el-col>
                    </el-row>
                  </el-collapse-item>
                </el-collapse>
              </el-form>
            </el-card>
          </el-col>

          <!-- Right: Battery Selection -->
          <el-col :span="12">
            <el-card shadow="never" header="电池数据集划分">
              <template #header>
                <div class="card-header">
                  <span>电池数据集划分</span>
                  <el-tag type="info" size="small">共 {{ batteryTableData.length }} 组</el-tag>
                </div>
              </template>
              <el-table :data="batteryTableData" height="500" border stripe>
                <el-table-column prop="battery_code" label="电池编号" width="120" />
                <el-table-column prop="total_cycles" label="循环次数" width="100" />
                <el-table-column label="用途划分">
                  <template #default="scope">
                    <el-select v-model="scope.row.role" size="small">
                      <el-option label="训练集 (Train)" value="train" />
                      <el-option label="验证集 (Val)" value="val" />
                      <el-option label="测试集 (Test)" value="test" />
                      <el-option label="忽略 (Ignore)" value="ignore" />
                    </el-select>
                  </template>
                </el-table-column>
              </el-table>
            </el-card>
          </el-col>
        </el-row>

        <div class="action-bar mt-20">
          <el-button type="primary" size="large" :loading="isSubmitting" @click="handleSubmit">
            开始训练任务
          </el-button>
        </div>
      </el-tab-pane>

      <!-- Tab 2: Monitor -->
      <el-tab-pane label="任务监控" name="monitor">
        <el-table :data="trainingJobs" border stripe>
          <el-table-column prop="id" label="ID" width="80" />
          <el-table-column prop="created_at" label="创建时间" width="180">
            <template #default="scope">
              {{ new Date(scope.row.created_at).toLocaleString() }}
            </template>
          </el-table-column>
          <el-table-column prop="target" label="目标" width="80" />
          <el-table-column label="状态" width="120">
            <template #default="scope">
              <el-tag :type="getStatusType(scope.row.status)">{{ scope.row.status }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="进度">
            <template #default="scope">
              <el-progress :percentage="Math.round(scope.row.progress * 100)" />
            </template>
          </el-table-column>
          <el-table-column label="操作" width="150">
            <template #default="scope">
              <el-button size="small" type="primary" link @click="showDetail(scope.row.id)"
                >查看详情</el-button
              >
              <el-button size="small" type="danger" link @click="handleDelete(scope.row.id)"
                >删除</el-button
              >
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
    </el-tabs>

    <!-- Detail Dialog -->
    <el-dialog
      v-model="detailVisible"
      title="任务详情"
      width="90%"
      destroy-on-close
      :close-on-click-modal="false"
      @opened="handleDialogOpened"
      @closed="handleDialogClosed"
    >
      <div v-if="currentJob" class="job-detail-container">
        <el-descriptions border :column="4" size="large">
          <el-descriptions-item label="任务ID" label-align="right">
            {{ currentJob.job.id }}
          </el-descriptions-item>
          <el-descriptions-item label="目标" label-align="right">
            {{ currentJob.job.target }}
          </el-descriptions-item>
          <el-descriptions-item label="状态" label-align="right">
            <el-tag :type="getStatusType(currentJob.job.status)">
              {{ currentJob.job.status }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="进度" label-align="right">
            {{ (currentJob.job.progress * 100).toFixed(1) }}%
          </el-descriptions-item>
        </el-descriptions>

        <el-tabs v-model="detailActiveTab" class="mt-20">
          <el-tab-pane label="概览" name="overview">
            <el-table :data="currentJob.runs" border stripe style="width: 100%">
              <el-table-column prop="algorithm" label="算法" width="150" />
              <el-table-column prop="status" label="状态" width="120">
                <template #default="scope">
                  <el-tag :type="getStatusType(scope.row.status)">{{ scope.row.status }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column label="训练进度" width="200">
                <template #default="scope">
                  {{ scope.row.current_epoch }} / {{ scope.row.total_epochs }}
                </template>
              </el-table-column>
              <el-table-column label="进度百分比" width="150">
                <template #default="scope">
                  <el-progress
                    :percentage="
                      Math.round((scope.row.current_epoch / scope.row.total_epochs) * 100)
                    "
                    :status="scope.row.status === 'SUCCEEDED' ? 'success' : undefined"
                  />
                </template>
              </el-table-column>
              <el-table-column label="开始时间" width="180">
                <template #default="scope">
                  {{ scope.row.started_at ? new Date(scope.row.started_at).toLocaleString() : '-' }}
                </template>
              </el-table-column>
              <el-table-column label="完成时间" width="180">
                <template #default="scope">
                  {{
                    scope.row.finished_at ? new Date(scope.row.finished_at).toLocaleString() : '-'
                  }}
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>

          <el-tab-pane label="实时指标" name="metrics">
            <div class="chart-controls">
              <el-select
                v-model="selectedRunId"
                placeholder="选择算法"
                style="width: 200px; margin-right: 20px"
              >
                <el-option
                  v-for="run in currentJob?.runs"
                  :key="run.id"
                  :label="`${run.algorithm} (${run.status})`"
                  :value="run.id"
                />
              </el-select>
              <el-switch
                v-model="isLogScale"
                active-text="对数坐标 (Log Scale)"
                inactive-text="线性坐标 (Linear)"
              />
            </div>
            <div ref="chartRef" class="chart-container"></div>
          </el-tab-pane>

          <el-tab-pane label="日志" name="logs">
            <div class="log-toolbar">
              <el-button type="primary" link icon="Download" @click="handleDownloadLogs">
                下载完整日志
              </el-button>
            </div>
            <div class="log-container">
              <div v-if="logs.length === 0" class="log-empty">
                <el-empty description="暂无日志数据" :image-size="80" />
              </div>
              <div v-else>
                <div v-for="(log, index) in logs" :key="index" class="log-item">
                  <span class="log-time">{{ formatLogTime(log.timestamp) }}</span>
                  <span :class="['log-level', `log-${log.level.toLowerCase()}`]">
                    [{{ log.level }}]
                  </span>
                  <span class="log-msg">{{ log.message }}</span>
                </div>
              </div>
            </div>
          </el-tab-pane>
        </el-tabs>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.training-container {
  padding: 20px;
}
.mt-20 {
  margin-top: 20px;
}
.action-bar {
  display: flex;
  justify-content: center;
  padding: 20px 0;
}
.job-detail-container {
  min-height: 500px;
}
.slider-container {
  display: flex;
  align-items: center;
  gap: 10px;
}
.slider-label {
  font-size: 12px;
  color: #606266;
  width: 100px;
}
.slider-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 5px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.chart-controls {
  display: flex;
  justify-content: flex-end;
  padding: 10px 20px 0;
}
.chart-container {
  width: 100%;
  height: 600px;
}
.log-toolbar {
  display: flex;
  justify-content: flex-end;
  padding-bottom: 10px;
}
.log-container {
  height: 500px;
  overflow-y: auto;
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 15px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 13px;
  border-radius: 4px;
  border: 1px solid #333;
}
.log-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  background: #1e1e1e;
}
.log-item {
  margin-bottom: 4px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  display: flex;
  align-items: baseline;
}
.log-time {
  color: #858585;
  margin-right: 8px;
  font-size: 11px;
  flex-shrink: 0;
  min-width: 90px;
}
.log-level {
  margin-right: 8px;
  font-weight: bold;
  flex-shrink: 0;
  min-width: 60px;
}
.log-msg {
  flex: 1;
  color: #d4d4d4;
}
.log-info {
  color: #4ec9b0;
}
.log-debug {
  color: #858585;
}
.log-warning {
  color: #dcdcaa;
}
.log-error {
  color: #f48771;
}
</style>
