<template>
  <div ref="chartRef" style="width: 100%; height: 300px;"></div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({ data: Array })
const chartRef = ref(null)
let chart = null

const render = () => {
  if (!chart || !props.data?.length) return
  const sorted = [...props.data].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 10)
  chart.setOption({
    title: { text: '特征贡献度 (SHAP)', left: 'center' },
    tooltip: {},
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: sorted.map(d => d.feature) },
    series: [{
      type: 'bar',
      data: sorted.map(d => ({ value: d.value, itemStyle: { color: d.value > 0 ? '#f56c6c' : '#67c23a' } }))
    }]
  })
}

onMounted(() => {
  chart = echarts.init(chartRef.value)
  render()
})

watch(() => props.data, render)
</script>
