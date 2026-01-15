<script setup lang="ts">
import { ref, onMounted, watch, onUnmounted } from 'vue'
import * as echarts from 'echarts'

const props = defineProps<{
  data: number[][] // [x, y]
  title?: string
  xAxisName?: string
  yAxisName?: string
}>()

const chartRef = ref<HTMLElement | null>(null)
let chartInstance: echarts.ECharts | null = null

const initChart = () => {
  if (!chartRef.value) return
  chartInstance = echarts.init(chartRef.value)
  updateChart()
  window.addEventListener('resize', handleResize)
}

const updateChart = () => {
  if (!chartInstance) return
  
  chartInstance.setOption({
    title: { text: props.title, left: 'center' },
    tooltip: { 
      trigger: 'item',
      formatter: (params: { data: number[] }) => `${props.xAxisName}: ${params.data[0]}<br/>${props.yAxisName}: ${params.data[1]}`
    },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value', name: props.xAxisName, scale: true },
    yAxis: { type: 'value', name: props.yAxisName, scale: true },
    series: [
      {
        symbolSize: 10,
        data: props.data,
        type: 'scatter',
        itemStyle: { color: '#67C23A' }
      }
    ]
  })
}

const handleResize = () => chartInstance?.resize()

watch(() => props.data, updateChart)

onMounted(initChart)
onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  chartInstance?.dispose()
})
</script>

<template>
  <div ref="chartRef" class="chart-container"></div>
</template>

<style scoped>
.chart-container {
  width: 100%;
  height: 100%;
  min-height: 300px;
}
</style>
