<template>
  <div class="page-container">
    <div class="page-header">
      <h2>
        <el-icon class="header-icon"><EditPen /></el-icon>
        手动数据录入
      </h2>
      <p class="sub-title">适用于已有纸质量表或第三方数据的情况，请输入学生原始因子分进行风险预测。</p>
    </div>

    <el-row :gutter="24">
      <el-col :xs="24" :lg="16">
        <el-card class="modern-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span class="title">SCL-90 症状自评量表</span>
              <el-tag effect="plain" type="primary" round>10 个因子</el-tag>
            </div>
          </template>

          <el-form :model="form" label-position="top">
            <el-row :gutter="20">
              <el-col :span="12" :sm="12" :md="8" v-for="(label, key) in scl90Labels" :key="key">
                <div class="input-group">
                  <div class="label-row">
                    <span class="input-label">{{ label }}</span>
                    <span class="input-value">{{ form.scl90[key].toFixed(1) }}</span>
                  </div>
                  <div class="control-row">
                    <el-slider 
                      v-model="form.scl90[key]" 
                      :min="1" :max="5" :step="0.1" 
                      size="small"
                      :show-tooltip="false"
                      class="custom-slider"
                    />
                    <el-input-number 
                      v-model="form.scl90[key]" 
                      :min="1" :max="5" :step="0.1" 
                      controls-position="right"
                      size="small"
                      class="mini-input"
                    />
                  </div>
                </div>
              </el-col>
            </el-row>
          </el-form>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="8">
        <el-card class="modern-card mb-20" shadow="hover">
          <template #header>
            <div class="card-header">
              <span class="title">EPQ 人格测验</span>
              <el-tag effect="plain" type="warning" round>4 个维度</el-tag>
            </div>
          </template>
          
          <el-form :model="form" label-width="90px">
            <div v-for="(label, key) in epqLabels" :key="key" class="epq-item">
              <el-form-item :label="label" class="mb-10">
                <el-input-number 
                  v-model="form.epq[key]" 
                  :min="0" :max="100" 
                  controls-position="right"
                  style="width: 100%"
                />
              </el-form-item>
            </div>
          </el-form>
        </el-card>

        <el-card class="modern-card action-card" shadow="always">
          <div class="action-area">
            <h3>准备就绪?</h3>
            <p>系统将基于 CatBoost 模型计算风险概率及 SHAP 特征值。</p>
            <el-button 
              type="primary" 
              size="large" 
              class="submit-btn" 
              @click="submit" 
              :loading="store.loading"
              round
            >
              <el-icon class="el-icon--left"><DataAnalysis /></el-icon>
              立即分析风险
            </el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { reactive } from 'vue'
import { useRouter } from 'vue-router'
import { usePredictionStore } from '@/stores/prediction'
// 确保图标已引入
import { EditPen, DataAnalysis } from '@element-plus/icons-vue'

const router = useRouter()
const store = usePredictionStore()

const scl90Labels = {
  somatization: '躯体化', obsessive_compulsive: '强迫症状', interpersonal_sensitivity: '人际敏感',
  depression: '抑郁', anxiety: '焦虑', hostility: '敌对',
  phobic_anxiety: '恐怖', paranoid_ideation: '偏执', psychoticism: '精神病性', other: '其他'
}

const epqLabels = { E: '内外向 (E)', P: '精神质 (P)', N: '神经质 (N)', L: '掩饰性 (L)' }

const form = reactive({
  scl90: Object.fromEntries(Object.keys(scl90Labels).map(k => [k, 1.5])), // 默认给个安全值
  epq: { E: 50, P: 50, N: 50, L: 50 }
})

const submit = async () => {
  try {
    await store.submitManual(form.scl90, form.epq)
    router.push('/result')
  } catch (error) {
    // 简单的错误提示，实际逻辑在 store 或 request 中处理
    console.error(error)
  }
}
</script>

<style scoped>
.page-container {
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 24px;
}
.page-header h2 {
  font-size: 24px;
  color: #1f2f3d;
  margin: 0 0 8px 0;
  display: flex;
  align-items: center;
}
.header-icon {
  margin-right: 10px;
  color: #409EFF;
}
.sub-title {
  color: #909399;
  font-size: 14px;
  margin: 0;
}

/* 卡片通用样式 */
.modern-card {
  border: none;
  border-radius: 8px;
  margin-bottom: 20px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.card-header .title {
  font-weight: 600;
  font-size: 16px;
  color: #303133;
}

/* SCL-90 输入组样式 */
.input-group {
  background-color: #f9fafc;
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 20px;
  border: 1px solid #e4e7ed;
  transition: all 0.3s;
}
.input-group:hover {
  border-color: #409EFF;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
  background-color: #fff;
}

.label-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 500;
  color: #606266;
}
.input-value {
  color: #409EFF;
  font-weight: bold;
}

.control-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.custom-slider {
  flex: 1;
}
.mini-input {
  width: 90px !important;
}

/* 右侧边栏样式 */
.mb-20 { margin-bottom: 20px; }
.mb-10 { margin-bottom: 12px; }

.action-card {
  background: linear-gradient(135deg, #409EFF 0%, #3a8ee6 100%);
  color: white;
  text-align: center;
}
.action-area h3 {
  margin: 10px 0;
  font-size: 18px;
}
.action-area p {
  font-size: 13px;
  opacity: 0.8;
  margin-bottom: 20px;
}
.submit-btn {
  width: 100%;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  font-weight: bold;
  height: 44px;
}
</style>