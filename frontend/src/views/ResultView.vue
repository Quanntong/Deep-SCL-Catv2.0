<template>
  <div class="result-page">
    <div v-if="store.result" class="animate-fade-in">
      
      <div class="status-banner" :class="riskLevel.class">
        <div class="banner-content">
          <div class="icon-wrapper">
            <el-icon :size="40">
              <component :is="riskLevel.icon" />
            </el-icon>
          </div>
          <div class="status-text">
            <h1>{{ riskLevel.title }}</h1>
            <p>{{ riskLevel.desc }}</p>
          </div>
        </div>
        <div class="banner-action">
          <el-button @click="$router.push('/manual')" plain round size="small">重新测评</el-button>
          <el-button @click="printReport" type="primary" round size="small">
            <el-icon style="margin-right: 4px"><Printer /></el-icon> 打印报告
          </el-button>
        </div>
      </div>

      <el-row :gutter="24" style="margin-top: -30px; position: relative; z-index: 2;">
        <el-col :xs="24" :md="8">
          <el-card class="metric-card" shadow="hover">
            <template #header>
              <div class="card-title">风险概率仪表盘</div>
            </template>
            <div class="gauge-container">
              <el-progress 
                type="dashboard" 
                :percentage="percentage" 
                :color="customColors"
                :width="180"
                :stroke-width="12"
              >
                <template #default="{ percentage }">
                  <span class="percentage-value">{{ percentage }}%</span>
                  <span class="percentage-label">风险指数</span>
                </template>
              </el-progress>
            </div>
            
            <div class="prediction-details">
              <div class="detail-row">
                <span class="label">预测挂科数</span>
                <span class="value" :class="failedCountClass">
                  {{ formatFailedCount }}
                </span>
              </div>
              <div class="detail-row">
                <span class="label">置信度</span>
                <el-tag size="small" effect="plain">高</el-tag>
              </div>
            </div>
          </el-card>

          <el-card class="suggestion-card" shadow="hover">
            <template #header>
              <div class="card-title">
                <el-icon class="suggestion-icon"><FirstAidKit /></el-icon> 
                <span>AI 辅导建议</span>
              </div>
            </template>
            <div class="suggestion-content">
              <ul v-if="store.result.is_risk" class="risk-list">
                <li>⚠️ <strong>立即关注：</strong>学生存在较高学业或心理预警。</li>
                <li>🗣️ <strong>访谈建议：</strong>请在3个工作日内安排线下谈话。</li>
                <li>📊 <strong>重点排查：</strong>请对照右侧图表，了解主要压力源。</li>
              </ul>
              <ul v-else class="safe-list">
                <li>✅ <strong>状态良好：</strong>当前各项指标处于正常范围。</li>
                <li>💪 <strong>持续保持：</strong>鼓励学生保持当前的心理调节方式。</li>
                <li>📅 <strong>定期复查：</strong>建议每学期进行一次例行测评。</li>
              </ul>
            </div>
          </el-card>
        </el-col>

        <el-col :xs="24" :md="16">
          <el-card class="chart-card" shadow="hover">
            <template #header>
              <div class="chart-header">
                <div class="header-left">
                  <h3>特征贡献度分析 (SHAP)</h3>
                  <el-tag size="small" type="info">可解释性模型</el-tag>
                </div>
                <el-tooltip content="红色条目代表推高风险的因素，蓝色条目代表降低风险的因素" placement="top">
                  <el-icon class="info-icon"><InfoFilled /></el-icon>
                </el-tooltip>
              </div>
            </template>
            
            <div class="chart-wrapper">
              <ShapChart :data="store.result.shap_values" />
            </div>
            
            <div class="chart-footer">
              <el-alert
                title="图表说明：条形图越长，代表该因子对本次预测结果（风险/正常）的影响权重越大。"
                type="info"
                :closable="false"
                show-icon
              />
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <div v-else class="empty-container">
      <el-empty description="暂无分析数据" :image-size="200">
        <template #extra>
          <el-button type="primary" size="large" @click="$router.push('/manual')">
            前往数据录入
          </el-button>
        </template>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { usePredictionStore } from '@/stores/prediction'
import ShapChart from '@/components/results/ShapChart.vue'
import { Warning, CircleCheck, InfoFilled, FirstAidKit, Printer } from '@element-plus/icons-vue'

const router = useRouter()
const store = usePredictionStore()

// 1. 处理进度条数值 (0-100)
const percentage = computed(() => {
  if (!store.result?.risk_probability) return 0
  return +(store.result.risk_probability * 100).toFixed(1)
})

// 2. 仪表盘颜色配置 (绿 -> 黄 -> 红)
const customColors = [
  { color: '#67c23a', percentage: 40 },
  { color: '#e6a23c', percentage: 70 },
  { color: '#f56c6c', percentage: 100 },
]

// 3. 计算风险等级展示逻辑
const riskLevel = computed(() => {
  if (store.result?.is_risk) {
    return {
      title: '高风险预警',
      desc: '系统检测到该学生存在潜在的心理危机或学业挂科风险。',
      class: 'bg-danger',
      icon: Warning
    }
  }
  return {
    title: '风险评估：低',
    desc: '各项指标平稳，未检测到显著的心理或学业风险信号。',
    class: 'bg-success',
    icon: CircleCheck
  }
})

// 4. 处理挂科数显示
const formatFailedCount = computed(() => {
  const count = store.result?.failed_subjects_predicted
  if (count === undefined || count === null) return '--'
  if (count < 0.5) return '0 科 (无风险)'
  return `约 ${count.toFixed(1)} 科`
})

const failedCountClass = computed(() => {
  const count = store.result?.failed_subjects_predicted
  return (count && count > 0.5) ? 'text-danger' : 'text-success'
})

// 5. 打印功能
const printReport = () => {
  window.print()
}
</script>

<style scoped>
.result-page {
  max-width: 1280px;
  margin: 0 auto;
  padding-bottom: 40px;
}

/* 顶部 Banner */
.status-banner {
  padding: 30px 40px 60px 40px; /* 底部留白给卡片上浮 */
  border-radius: 12px;
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}
.bg-danger { 
  background: linear-gradient(135deg, #f56c6c 0%, #ff8e8e 100%); 
}
.bg-success { 
  background: linear-gradient(135deg, #67c23a 0%, #85ce61 100%); 
}

.banner-content { display: flex; align-items: center; gap: 20px; }
.icon-wrapper {
  background: rgba(255,255,255,0.2);
  border-radius: 50%;
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.status-text h1 { margin: 0; font-size: 24px; font-weight: 700; letter-spacing: 1px; }
.status-text p { margin: 6px 0 0 0; opacity: 0.9; font-size: 14px; }

/* 卡片通用 */
.metric-card, .suggestion-card, .chart-card {
  border: none;
  border-radius: 8px;
  margin-bottom: 20px;
}
.card-title { font-weight: 600; color: #303133; font-size: 16px; display: flex; align-items: center; gap: 8px; }

/* 仪表盘区域 */
.gauge-container {
  display: flex;
  justify-content: center;
  padding: 20px 0;
}
.percentage-value { display: block; font-size: 32px; font-weight: bold; color: #303133; line-height: 1.2; }
.percentage-label { font-size: 12px; color: #909399; }

/* 预测详情 */
.prediction-details {
  background: #f8f9fa;
  border-radius: 6px;
  padding: 15px;
}
.detail-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 14px;
}
.detail-row:last-child { margin-bottom: 0; }
.detail-row .label { color: #606266; }
.detail-row .value { font-weight: 600; font-family: monospace; font-size: 15px; }
.text-danger { color: #f56c6c; }
.text-success { color: #67c23a; }

/* 建议列表 */
.suggestion-content ul {
  padding-left: 0;
  margin: 0;
  list-style: none;
}
.suggestion-content li {
  margin-bottom: 12px;
  font-size: 14px;
  line-height: 1.6;
  color: #606266;
  background: #fcfcfc;
  padding: 8px 12px;
  border-radius: 4px;
  border-left: 3px solid #ebeef5;
}
.risk-list li { border-left-color: #f56c6c; }
.safe-list li { border-left-color: #67c23a; }

/* 图表区域 */
.chart-header { display: flex; justify-content: space-between; align-items: center; }
.header-left { display: flex; align-items: center; gap: 10px; }
.chart-header h3 { margin: 0; font-size: 16px; color: #303133; }
.chart-wrapper { height: 420px; width: 100%; }
.chart-footer { margin-top: 10px; }

/* 动画与空状态 */
.animate-fade-in { animation: fadeInUp 0.5s ease-out; }
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.empty-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 60vh;
}

/* 打印优化 */
@media print {
  .aside-menu, .banner-action, .el-button { display: none !important; }
  .result-page { padding: 0; }
  .status-banner { color: black !important; background: none !important; border: 1px solid #000; box-shadow: none; padding: 20px; }
  .chart-wrapper { height: 300px; }
}
</style>