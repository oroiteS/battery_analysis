<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, nextTick } from 'vue'
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
      label: `${algoName} | ${dateStr} | ${metricsStr}`,
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
  if (data.type === 'log') {
    testLogs.value.push(`[${data.level}] ${data.message}`)
  } else if (data.type === 'progress') {
    testProgress.value = typeof data.progress === 'number' ? data.progress : 0
  } else if (data.type === 'status') {
    if (data.status === 'SUCCEEDED') {
      ElMessage.success('测试完成')
      loadTestResults()
    } else if (data.status === 'FAILED') {
      ElMessage.error('测试失败')
      isTesting.value = false
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
        isTesting.value = false
        await loadTestResults()
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

// 加载测试结果
const loadTestResults = async () => {
  if (!currentJobId.value) return

  try {
    // 加载指标
    metrics.value = await getTestMetrics(currentJobId.value)

    // 加载预测结果
    const predRes = await getTestPredictions(currentJobId.value)
    predictions.value = predRes.predictions

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
  if (testForm.value.target === 'RUL' || testForm.value.target === 'BOTH') {
    renderRULChart()
  }
  if (testForm.value.target === 'PCL' || testForm.value.target === 'BOTH') {
    renderPCLChart()
  }
}

// 渲染RUL图表
const renderRULChart = () => {
  if (!rulChartRef.value) return

  if (!rulChart) {
    rulChart = echarts.init(rulChartRef.value)
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
  if (!pclChartRef.value) return

  if (!pclChart) {
    pclChart = echarts.init(pclChartRef.value)
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

// 生命周期
onMounted(() => {
  loadDatasets()
  loadModelVersions()
})

onUnmounted(() => {
  if (ws) {
    ws.close()
  }
  if (rulChart) {
    rulChart.dispose()
  }
  if (pclChart) {
    pclChart.dispose()
  }
})
</script>

<template>
  <div class="prediction-container">
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
                <el-option v-for="ds in datasets" :key="ds.id" :label="ds.name" :value="ds.id" />
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
          <el-button type="primary" icon="VideoPlay" :loading="isTesting" @click="handleStartTest">
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
              <el-descriptions-item v-for="(value, key) in metric.metrics" :key="key" :label="key">
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
            <div ref="rulChartRef" style="height: 400px"></div>
          </el-card>
        </el-col>
        <el-col
          v-if="testForm.target === 'PCL' || testForm.target === 'BOTH'"
          :span="testForm.target === 'BOTH' ? 12 : 24"
        >
          <el-card shadow="hover">
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
