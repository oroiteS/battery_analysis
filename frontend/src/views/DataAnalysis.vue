<script setup lang="ts">
import { ref, onMounted } from 'vue'
import LineChart from '../components/charts/LineChart.vue'
import ScatterChart from '../components/charts/ScatterChart.vue'
import HistogramChart from '../components/charts/HistogramChart.vue'
import HeatmapChart from '../components/charts/HeatmapChart.vue'
import { getDatasets, getBatteries } from '../api/data'
import {
  getBatteryStats,
  getTrendData,
  getScatterData,
  getPCLDistribution,
  getCorrelationMatrix,
  exportAnalysisReport,
} from '../api/analysis'
import type { BatteryUnit } from '../api/types'
import { ElMessage } from 'element-plus'

defineOptions({
  name: 'DataAnalysisView',
})

const datasetId = ref<number | null>(null)
const batteryId = ref<number | null>(null)
const batteryOptions = ref<{ value: number; label: string }[]>([])

// Feature selection for Line Chart
const selectedFeature = ref('feature_1')
const featureOptions = Array.from({ length: 8 }, (_, i) => ({
  value: `feature_${i + 1}`,
  label: `Feature ${i + 1}`,
}))

// Feature selection for Scatter Chart (RUL)
const selectedRulFeature = ref('feature_1')

// --- Data ---

// 1. Statistical Table Data
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const tableData = ref<any[]>([])

// 2. Line Chart Data (Trend)
const lineChartData = ref<number[]>([])
const lineChartXAxis = ref<string[]>([])

// 3. Scatter Chart Data (RUL)
const scatterChartData = ref<number[][]>([])

// 4. Histogram Data (PCL)
const histData = ref<number[]>([])
const histCategories = ref<string[]>([])

// 5. Heatmap Data (Correlation)
const heatmapData = ref<number[][]>([])
const heatmapLabels = ref<string[]>([])

const loading = ref(false)

// Initialize: Fetch datasets and batteries
const initData = async () => {
  try {
    // 1. Get Datasets (MVP: Built-in only)
    const datasets = await getDatasets()
    if (datasets[0] && datasets.length > 0) {
      datasetId.value = datasets[0].id // Store dataset ID

      // 2. Get Batteries
      const batteries = await getBatteries(datasetId.value)
      if (batteries) {
        batteryOptions.value = batteries.map((b: BatteryUnit) => ({
          value: b.id,
          label: `${b.battery_code} (Cycles: ${b.total_cycles})`,
        }))

        if (batteryOptions.value.length > 0 && batteryOptions.value[0]) {
          batteryId.value = batteryOptions.value[0].value
          // 3. Load Analysis Data
          await handleSearch()
        }
      }
    }
  } catch (error) {
    console.error('Failed to init data:', error)
  }
}

const handleSearch = async () => {
  if (!batteryId.value) return

  loading.value = true
  try {
    const id = batteryId.value

    // 1. Stats
    const stats = await getBatteryStats(id)
    if (stats) {
      tableData.value = stats.map((s) => ({
        name: s.feature_name,
        mean: s.mean.toFixed(4),
        variance: s.variance.toFixed(4),
        min: s.min_val.toFixed(4),
        max: s.max_val.toFixed(4),
        r2_rul: s.corr_rul?.toFixed(4) || 'N/A',
        r2_pcl: s.corr_pcl?.toFixed(4) || 'N/A',
      }))
    }

    // 2. Trend
    await updateTrendChart()

    // 3. Scatter
    await updateScatterChart()

    // 4. PCL Distribution
    const pclDist = await getPCLDistribution(id)
    // Simple binning for histogram
    if (pclDist && pclDist.pcl_values) {
      const values = pclDist.pcl_values
      if (values.length > 0) {
        const min = Math.min(...values)
        const max = Math.max(...values)
        const binCount = 10
        const step = (max - min) / binCount

        const bins: number[] = Array.from({ length: binCount }).fill(0) as number[]
        const categories: string[] = []

        for (let i = 0; i < binCount; i++) {
          const start = min + i * step
          const end = min + (i + 1) * step
          categories.push(`${start.toFixed(2)}-${end.toFixed(2)}`)
        }

        values.forEach((v) => {
          const index = Math.max(0, Math.min(Math.floor((v - min) / step), binCount - 1))
          if (bins[index] !== undefined) {
            bins[index]++
          }
        })

        histData.value = bins
        histCategories.value = categories
      } else {
        histData.value = []
        histCategories.value = []
      }
    }

    // 5. Correlation Matrix
    const matrix = await getCorrelationMatrix(id)
    if (matrix && matrix.features && matrix.matrix) {
      heatmapLabels.value = matrix.features
      const heatData: number[][] = []
      matrix.matrix.forEach((row, i) => {
        row.forEach((val, j) => {
          heatData.push([i, j, parseFloat(val.toFixed(2))])
        })
      })
      heatmapData.value = heatData
    }
  } catch (error) {
    console.error('Analysis failed:', error)
    ElMessage.error('Failed to load analysis data')
  } finally {
    loading.value = false
  }
}

const updateTrendChart = async () => {
  if (!batteryId.value) return
  try {
    const trend = await getTrendData(batteryId.value, selectedFeature.value)
    if (trend && trend.cycles && trend.values) {
      lineChartXAxis.value = trend.cycles.map(String)
      lineChartData.value = trend.values
    }
  } catch (error) {
    console.error(error)
  }
}

const updateScatterChart = async () => {
  if (!batteryId.value) return
  try {
    const scatter = await getScatterData(batteryId.value, selectedRulFeature.value)
    if (scatter && scatter.points) {
      scatterChartData.value = scatter.points
    }
  } catch (error) {
    console.error(error)
  }
}

const handleFeatureChange = () => {
  updateTrendChart()
}

const handleRulFeatureChange = () => {
  updateScatterChart()
}

// 导出分析报告
const handleExportReport = async () => {
  if (!datasetId.value || !batteryId.value) {
    ElMessage.warning('请先选择电池组')
    return
  }

  try {
    await exportAnalysisReport(datasetId.value, batteryId.value, 'XLSX')
    ElMessage.success('分析报告导出成功')
  } catch (error) {
    console.error('Export failed:', error)
    ElMessage.error('导出分析报告失败')
  }
}

// Utility for conditional formatting
const getCorrelationStyle = (value: string) => {
  if (value === 'N/A') return {}
  const num = parseFloat(value)
  if (Math.abs(num) > 0.7) {
    return { color: '#F56C6C', fontWeight: 'bold' } // Red for high correlation
  }
  return {}
}

onMounted(() => {
  initData()
})
</script>

<template>
  <div class="analysis-container" v-loading="loading">
    <!-- 1. Top Header Controls -->
    <el-card shadow="never" class="header-card">
      <div class="header-content">
        <div class="left-panel">
          <span class="label">电池组</span>
          <el-select
            v-model="batteryId"
            placeholder="请选择电池组"
            style="width: 240px"
            filterable
            @change="handleSearch"
          >
            <el-option
              v-for="item in batteryOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </div>
        <div class="right-panel">
          <el-button type="primary" icon="Download" @click="handleExportReport">
            导出报告
          </el-button>
        </div>
      </div>
    </el-card>

    <!-- 2. Statistical Overview Table -->
    <el-card shadow="never" class="mt-20" header="统计概览">
      <el-table :data="tableData" style="width: 100%">
        <el-table-column prop="name" label="特征名称" width="180" />
        <el-table-column prop="mean" label="均值 (Mean)" />
        <el-table-column prop="variance" label="方差 (Variance)" />
        <el-table-column prop="min" label="最小值 (Min)" />
        <el-table-column prop="max" label="最大值 (Max)" />
        <el-table-column label="与 RUL 相关性 (R²)">
          <template #default="scope">
            <span :style="getCorrelationStyle(scope.row.r2_rul)">
              {{ scope.row.r2_rul }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="与 PCL 相关性 (R²)">
          <template #default="scope">
            <span :style="getCorrelationStyle(scope.row.r2_pcl)">
              {{ scope.row.r2_pcl }}
            </span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 3. Visualization Dashboard -->
    <el-row :gutter="24" class="mt-20">
      <!-- Top-Left: Trend Analysis -->
      <el-col :span="12">
        <el-card shadow="never" class="chart-card">
          <template #header>
            <div class="card-header">
              <span>趋势分析</span>
              <el-select
                v-model="selectedFeature"
                size="small"
                style="width: 120px"
                @change="handleFeatureChange"
              >
                <el-option
                  v-for="opt in featureOptions"
                  :key="opt.value"
                  :label="opt.label"
                  :value="opt.value"
                />
              </el-select>
            </div>
          </template>
          <LineChart :data="lineChartData" :x-axis-data="lineChartXAxis" y-axis-name="Value" />
        </el-card>
      </el-col>

      <!-- Top-Right: RUL Scatter -->
      <el-col :span="12">
        <el-card shadow="never" class="chart-card">
          <template #header>
            <div class="card-header">
              <span>RUL 影响因子分析</span>
              <el-select
                v-model="selectedRulFeature"
                size="small"
                style="width: 120px"
                @change="handleRulFeatureChange"
              >
                <el-option
                  v-for="opt in featureOptions"
                  :key="opt.value"
                  :label="opt.label"
                  :value="opt.value"
                />
              </el-select>
            </div>
          </template>
          <ScatterChart
            :data="scatterChartData"
            x-axis-name="Feature Value"
            y-axis-name="RUL (Cycles)"
          />
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="24" class="mt-20">
      <!-- Bottom-Left: PCL Distribution -->
      <el-col :span="12">
        <el-card shadow="never" class="chart-card" header="容量衰减分布 (PCL)">
          <HistogramChart :data="histData" :categories="histCategories" />
        </el-card>
      </el-col>

      <!-- Bottom-Right: Correlation Heatmap -->
      <el-col :span="12">
        <el-card shadow="never" class="chart-card" header="相关性矩阵">
          <HeatmapChart :data="heatmapData" :x-labels="heatmapLabels" :y-labels="heatmapLabels" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.mt-20 {
  margin-top: 24px;
}
.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.label {
  margin-right: 12px;
  font-weight: 500;
  color: var(--color-text-main);
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.chart-card {
  height: 460px;
  display: flex;
  flex-direction: column;
}
:deep(.el-card__body) {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}
</style>
