<script setup lang="ts">
import { ref, onMounted } from 'vue'
import * as echarts from 'echarts'

const batteryId = ref('')
const batteryOptions = Array.from({ length: 124 }, (_, i) => ({
  value: `B${String(i + 1).padStart(4, '0')}`,
  label: `电池组 B${String(i + 1).padStart(4, '0')}`
}))

// Mock Data
const statsData = ref([
  { label: '电压均值', value: '3.65 V' },
  { label: '电流均值', value: '1.2 A' },
  { label: '最高温度', value: '42 °C' },
  { label: '循环次数', value: '1200' },
])

const handleSearch = () => {
  // TODO: Fetch data from API
  console.log('Searching for battery:', batteryId.value)
}

// Chart Refs
const lineChartRef = ref(null)
const scatterChartRef = ref(null)
const histChartRef = ref(null)
const heatmapChartRef = ref(null)

onMounted(() => {
  // Initialize charts (mock)
  initCharts()
})

const initCharts = () => {
  if (lineChartRef.value) {
    const chart = echarts.init(lineChartRef.value)
    chart.setOption({
      title: { text: '特征参数随循环次数变化' },
      tooltip: { trigger: 'axis' },
      legend: { data: ['电压', '电流', '温度'] },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', data: ['0', '200', '400', '600', '800', '1000', '1200'] },
      yAxis: { type: 'value' },
      series: [
        { data: [3.8, 3.75, 3.7, 3.65, 3.6, 3.55, 3.5], type: 'line', name: '电压', smooth: true },
        { data: [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2], type: 'line', name: '电流', smooth: true },
        { data: [25, 26, 28, 30, 35, 38, 42], type: 'line', name: '温度', yAxisIndex: 0, smooth: true }
      ]
    })
    window.addEventListener('resize', () => chart.resize())
  }

  if (scatterChartRef.value) {
    const chart = echarts.init(scatterChartRef.value)
    chart.setOption({
      title: { text: 'RUL 与 电压 相关性' },
      tooltip: { trigger: 'item' },
      xAxis: { name: '电压 (V)', type: 'value', scale: true },
      yAxis: { name: 'RUL (Cycles)', type: 'value', scale: true },
      series: [{
        symbolSize: 10,
        data: [
          [3.8, 1200], [3.75, 1000], [3.7, 800], [3.65, 600], [3.6, 400], [3.55, 200], [3.5, 0]
        ],
        type: 'scatter'
      }]
    })
    window.addEventListener('resize', () => chart.resize())
  }

  if (histChartRef.value) {
    const chart = echarts.init(histChartRef.value)
    chart.setOption({
      title: { text: 'PCL 分布情况' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: ['0-10%', '10-20%', '20-30%', '30-40%', '>40%'] },
      yAxis: { type: 'value' },
      series: [{
        data: [12, 30, 45, 20, 17],
        type: 'bar',
        itemStyle: { color: '#409EFF' }
      }]
    })
    window.addEventListener('resize', () => chart.resize())
  }

  if (heatmapChartRef.value) {
    const chart = echarts.init(heatmapChartRef.value)
    const hours = ['电压', '电流', '温度', 'RUL', 'PCL'];
    const days = ['电压', '电流', '温度', 'RUL', 'PCL'];
    const data = [
      [0, 0, 1], [0, 1, 0.1], [0, 2, 0.2], [0, 3, 0.8], [0, 4, 0.7],
      [1, 0, 0.1], [1, 1, 1], [1, 2, 0.3], [1, 3, 0.1], [1, 4, 0.1],
      [2, 0, 0.2], [2, 1, 0.3], [2, 2, 1], [2, 3, -0.5], [2, 4, -0.4],
      [3, 0, 0.8], [3, 1, 0.1], [3, 2, -0.5], [3, 3, 1], [3, 4, 0.9],
      [4, 0, 0.7], [4, 1, 0.1], [4, 2, -0.4], [4, 3, 0.9], [4, 4, 1]
    ].map(function (item) {
      return [item[1], item[0], item[2] || '-'];
    });

    chart.setOption({
      title: { text: '特征相关性矩阵' },
      tooltip: { position: 'top' },
      grid: { height: '50%', top: '10%' },
      xAxis: { type: 'category', data: hours, splitArea: { show: true } },
      yAxis: { type: 'category', data: days, splitArea: { show: true } },
      visualMap: {
        min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: '15%'
      },
      series: [{
        name: 'Correlation',
        type: 'heatmap',
        data: data,
        label: { show: true },
        emphasis: {
          itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' }
        }
      }]
    })
    window.addEventListener('resize', () => chart.resize())
  }
}
</script>

<template>
  <div class="analysis-container">
    <el-card shadow="hover" class="filter-card">
      <el-form :inline="true">
        <el-form-item label="电池组编号">
          <el-select v-model="batteryId" placeholder="请选择电池组" style="width: 200px" filterable>
            <el-option
              v-for="item in batteryOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" icon="Search" @click="handleSearch">查询分析</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-row :gutter="20" class="mt-20">
      <el-col :span="24">
        <el-card shadow="hover" header="统计指标">
          <el-descriptions border>
            <el-descriptions-item v-for="item in statsData" :key="item.label" :label="item.label">
              {{ item.value }}
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="mt-20">
      <el-col :span="12">
        <el-card shadow="hover">
          <div ref="lineChartRef" style="height: 300px;"></div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover">
          <div ref="scatterChartRef" style="height: 300px;"></div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="mt-20">
      <el-col :span="12">
        <el-card shadow="hover">
          <div ref="histChartRef" style="height: 300px;"></div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover">
          <div ref="heatmapChartRef" style="height: 300px;"></div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.mt-20 {
  margin-top: 20px;
}
</style>
