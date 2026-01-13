<script setup lang="ts">
import { ref, onMounted } from 'vue'
import LineChart from '../components/charts/LineChart.vue'
import ScatterChart from '../components/charts/ScatterChart.vue'
import HistogramChart from '../components/charts/HistogramChart.vue'
import HeatmapChart from '../components/charts/HeatmapChart.vue'

const batteryId = ref('')
const batteryOptions = Array.from({ length: 124 }, (_, i) => ({
  value: `B${String(i + 1).padStart(4, '0')}`,
  label: `电池组 B${String(i + 1).padStart(4, '0')}`
}))

// Feature selection for Line Chart
const selectedFeature = ref('feature_1')
const featureOptions = Array.from({ length: 8 }, (_, i) => ({
  value: `feature_${i + 1}`,
  label: `Feature ${i + 1}`
}))

// Feature selection for Scatter Chart (RUL)
const selectedRulFeature = ref('feature_1')

// --- Mock Data ---

// 1. Statistical Table Data
const tableData = ref(Array.from({ length: 8 }, (_, i) => ({
  name: `feature_${i + 1}`,
  mean: (Math.random() * 10).toFixed(2),
  variance: (Math.random() * 2).toFixed(2),
  min: (Math.random() * 5).toFixed(2),
  max: (Math.random() * 15).toFixed(2),
  r2_rul: (Math.random() * 2 - 1).toFixed(3), // -1 to 1
  r2_pcl: (Math.random() * 2 - 1).toFixed(3)
})))

// 2. Line Chart Data (Trend)
const lineChartData = ref<number[]>([])
const lineChartXAxis = ref<string[]>([])

// 3. Scatter Chart Data (RUL)
const scatterChartData = ref<number[][]>([])

// 4. Histogram Data (PCL)
const histData = ref<number[]>([12, 30, 45, 20, 17])
const histCategories = ['0-10%', '10-20%', '20-30%', '30-40%', '>40%']

// 5. Heatmap Data (Correlation)
const heatmapData = ref<number[][]>([])
const heatmapLabels = Array.from({ length: 8 }, (_, i) => `F${i + 1}`)

const handleSearch = () => {
  // TODO: Fetch real data based on batteryId
  console.log('Searching for battery:', batteryId.value)
  // Simulate data refresh
  generateMockData()
}

const handleFeatureChange = () => {
  // Update line chart data based on selected feature
  // Mocking update:
  lineChartData.value = Array.from({ length: 50 }, () => Math.random() * 10)
}

const handleRulFeatureChange = () => {
  // Update scatter chart data based on selected feature
  // Mocking update:
  scatterChartData.value = Array.from({ length: 50 }, () => [Math.random() * 5, Math.random() * 1000])
}

const generateMockData = () => {
  // Line Chart
  lineChartXAxis.value = Array.from({ length: 50 }, (_, i) => `${i * 20}`)
  lineChartData.value = Array.from({ length: 50 }, () => Math.random() * 10)

  // Scatter Chart
  scatterChartData.value = Array.from({ length: 50 }, () => [Math.random() * 5, Math.random() * 1000])

  // Heatmap
  const data: number[][] = []
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      data.push([i, j, parseFloat((Math.random() * 2 - 1).toFixed(2))])
    }
  }
  heatmapData.value = data
}

// Utility for conditional formatting
const getCorrelationStyle = (value: string) => {
  const num = parseFloat(value)
  if (Math.abs(num) > 0.7) {
    return { color: '#F56C6C', fontWeight: 'bold' } // Red for high correlation
  }
  return {}
}

onMounted(() => {
  generateMockData()
})
</script>

<template>
  <div class="analysis-container">
    <!-- 1. Top Header Controls -->
    <el-card shadow="hover" class="header-card">
      <div class="header-content">
        <div class="left-panel">
          <span class="label">电池组编号：</span>
          <el-select v-model="batteryId" placeholder="请选择电池组" style="width: 240px" filterable @change="handleSearch">
            <el-option
              v-for="item in batteryOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </div>
        <div class="right-panel">
          <el-button type="primary" icon="Download">导出分析报告</el-button>
        </div>
      </div>
    </el-card>

    <!-- 2. Statistical Overview Table -->
    <el-card shadow="hover" class="mt-20" header="特征统计概览">
      <el-table :data="tableData" border stripe style="width: 100%">
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
    <el-row :gutter="20" class="mt-20">
      <!-- Top-Left: Trend Analysis -->
      <el-col :span="12">
        <el-card shadow="hover" class="chart-card">
          <template #header>
            <div class="card-header">
              <span>特征趋势分析(各特征随循环次数的变化)</span>
              <el-select v-model="selectedFeature" size="small" style="width: 120px" @change="handleFeatureChange">
                <el-option v-for="opt in featureOptions" :key="opt.value" :label="opt.label" :value="opt.value" />
              </el-select>
            </div>
          </template>
          <LineChart
            :data="lineChartData"
            :x-axis-data="lineChartXAxis"
            y-axis-name="Value"
          />
        </el-card>
      </el-col>

      <!-- Top-Right: RUL Scatter -->
      <el-col :span="12">
        <el-card shadow="hover" class="chart-card">
          <template #header>
            <div class="card-header">
              <span>RUL 关键因子分析 (Top Feature vs RUL)</span>
              <el-select v-model="selectedRulFeature" size="small" style="width: 120px" @change="handleRulFeatureChange">
                <el-option v-for="opt in featureOptions" :key="opt.value" :label="opt.label" :value="opt.value" />
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

    <el-row :gutter="20" class="mt-20">
      <!-- Bottom-Left: PCL Distribution -->
      <el-col :span="12">
        <el-card shadow="hover" class="chart-card" header="容量衰减分布 (PCL Distribution)">
          <HistogramChart
            :data="histData"
            :categories="histCategories"
          />
        </el-card>
      </el-col>

      <!-- Bottom-Right: Correlation Heatmap -->
      <el-col :span="12">
        <el-card shadow="hover" class="chart-card" header="多特征相关性矩阵">
          <HeatmapChart
            :data="heatmapData"
            :x-labels="heatmapLabels"
            :y-labels="heatmapLabels"
          />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.mt-20 {
  margin-top: 20px;
}
.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.label {
  margin-right: 10px;
  font-weight: bold;
  color: #606266;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.chart-card {
  height: 420px; /* Fixed height for consistency */
  display: flex;
  flex-direction: column;
}
/* Ensure chart components take available height */
:deep(.el-card__body) {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px;
}
</style>
