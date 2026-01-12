<script setup lang="ts">
import { ref, onMounted, watch, onUnmounted } from 'vue'
import * as echarts from 'echarts'

const props = defineProps<{
  data: number[][] // [yIndex, xIndex, value]
  xLabels: string[]
  yLabels: string[]
  title?: string
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
  
  const formattedData = props.data.map(item => [item[1], item[0], item[2] || '-'])

  chartInstance.setOption({
    title: { text: props.title, left: 'center' },
    tooltip: { position: 'top' },
    grid: { height: '70%', top: '15%' },
    xAxis: { type: 'category', data: props.xLabels, splitArea: { show: true } },
    yAxis: { type: 'category', data: props.yLabels, splitArea: { show: true } },
    visualMap: {
      min: -1,
      max: 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '0%'
    },
    series: [{
      name: 'Correlation',
      type: 'heatmap',
      data: formattedData,
      label: { show: true },
      emphasis: {
        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' }
      }
    }]
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
