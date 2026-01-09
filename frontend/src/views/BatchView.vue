<template>
  <div class="batch-view">
    <el-card>
      <template #header><span>批量数据分析</span></template>
      
      <el-radio-group v-model="mode" style="margin-bottom: 20px;">
        <el-radio-button value="strict">精准模式</el-radio-button>
        <el-radio-button value="balanced">均衡模式</el-radio-button>
        <el-radio-button value="sensitive">全面模式</el-radio-button>
      </el-radio-group>
      <div class="mode-tip">{{ modeTips[mode] }}</div>
      
      <el-upload
        drag
        :auto-upload="false"
        :limit="1"
        accept=".xlsx,.xls,.csv"
        :on-change="handleFile"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">拖拽文件到此处，或 <em>点击上传</em></div>
        <template #tip>
          <div class="el-upload__tip">支持 Excel (.xlsx/.xls) 或 CSV 文件</div>
        </template>
      </el-upload>
      
      <div style="margin-top: 20px;">
        <el-button type="primary" @click="submit" :disabled="!file" :loading="loading">
          开始分析
        </el-button>
        <el-button type="success" @click="exportExcel" :disabled="!file" :loading="exporting">
          导出 Excel
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { UploadFilled } from '@element-plus/icons-vue'
import { usePredictionStore } from '@/stores/prediction'

const router = useRouter()
const store = usePredictionStore()
const file = ref(null)
const loading = ref(false)
const exporting = ref(false)
const mode = ref('balanced')
const modeTips = {
  strict: '精准模式：仅标记高置信度风险（概率≥60%），减少误报',
  balanced: '均衡模式：平衡精确率与召回率（概率≥52%或挂科≥0.7）',
  sensitive: '全面模式：尽可能发现潜在风险（概率≥45%或挂科≥0.4）'
}

const handleFile = (f) => { file.value = f.raw }

const submit = async () => {
  loading.value = true
  try {
    await store.uploadBatch(file.value, mode.value)
    router.push('/result')
  } finally {
    loading.value = false
  }
}

const exportExcel = async () => {
  exporting.value = true
  try {
    const formData = new FormData()
    formData.append('file', file.value)
    formData.append('mode', mode.value)
    const response = await fetch('http://localhost:8000/api/predict/batch/export', {
      method: 'POST',
      body: formData
    })
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'prediction_results.xlsx'
    a.click()
    URL.revokeObjectURL(url)
  } finally {
    exporting.value = false
  }
}
</script>

<style scoped>
.batch-view { padding: 20px; max-width: 600px; margin: 0 auto; }
.mode-tip { color: #909399; font-size: 12px; margin-bottom: 20px; }
</style>
