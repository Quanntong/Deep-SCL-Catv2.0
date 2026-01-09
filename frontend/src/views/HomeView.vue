<template>
  <div class="dashboard-container">
    <el-row :gutter="20" class="stat-row">
      <el-col :xs="24" :sm="12" :md="6">
        <el-card class="stat-card" :body-style="{ padding: '20px' }" shadow="hover">
          <div class="stat-content">
            <div class="stat-icon">
              <img src="@/assets/logo.png" class="stat-img" />
            </div>
            <div class="stat-info">
              <div class="stat-label">模型训练样本</div>
              <div class="stat-value">
                <span class="num">{{ stats.total }}</span>
                <span class="unit">人</span>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="12" :md="6">
        <el-card class="stat-card" :body-style="{ padding: '20px' }" shadow="hover">
          <div class="stat-content">
            <div class="stat-icon">
              <img src="@/assets/logo.png" class="stat-img" />
            </div>
            <div class="stat-info">
              <div class="stat-label">历史高危案例</div>
              <div class="stat-value">
                <span class="num">{{ stats.high_risk }}</span>
                <span class="unit">人</span>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="12" :md="6">
        <el-card class="stat-card" :body-style="{ padding: '20px' }" shadow="hover">
          <div class="stat-content">
            <div class="stat-icon">
              <img src="@/assets/logo.png" class="stat-img" />
            </div>
            <div class="stat-info">
              <div class="stat-label">样本风险率</div>
              <div class="stat-value">
                <span class="num">{{ stats.ratio }}</span>
                <span class="unit">%</span>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="12" :md="6">
        <el-card class="stat-card" :body-style="{ padding: '20px' }" shadow="hover">
          <div class="stat-content">
            <div class="stat-icon">
              <img src="@/assets/logo.png" class="stat-img" />
            </div>
            <div class="stat-info">
              <div class="stat-label">数据最后更新</div>
              <div class="stat-value">
                <span class="num" style="font-size: 14px;">{{ stats.update_time }}</span>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="main-row">
      <el-col :xs="24" :md="16">
        <el-card class="box-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span><el-icon><Menu /></el-icon> 常用功能直达</span>
            </div>
          </template>
          <div class="shortcut-grid">
            <div class="shortcut-item" @click="$router.push('/manual')">
              <div class="sc-icon bg-blue"><el-icon><EditPen /></el-icon></div>
              <h4>手动录入</h4>
              <p>单人快速筛查</p>
            </div>
            <div class="shortcut-item" @click="$router.push('/batch')">
              <div class="sc-icon bg-green"><el-icon><UploadFilled /></el-icon></div>
              <h4>批量上传</h4>
              <p>Excel 数据分析</p>
            </div>
            <div class="shortcut-item" @click="$router.push('/test')">
              <div class="sc-icon bg-purple"><el-icon><Monitor /></el-icon></div>
              <h4>在线测评</h4>
              <p>学生自测终端</p>
            </div>
            <div class="shortcut-item" @click="$router.push('/result')">
              <div class="sc-icon bg-orange"><el-icon><TrendCharts /></el-icon></div>
              <h4>历史记录</h4>
              <p>查看过往报告</p>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :md="8">
        <el-card class="box-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span><el-icon><Bell /></el-icon> 系统动态</span>
            </div>
          </template>
          <el-timeline>
            <el-timeline-item timestamp="刚刚" type="primary" hollow>
              系统初始化完成，后端连接正常。
            </el-timeline-item>
            <el-timeline-item timestamp="模型信息" color="#0bbd87">
              训练样本 {{ stats.total }} 人，风险率 {{ stats.ratio }}%
            </el-timeline-item>
            <el-timeline-item timestamp="2024-03-01" type="info">
              Deep-SCL-Cat V2.0 版本上线。
            </el-timeline-item>
          </el-timeline>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '@/api/client'
import { 
  DataLine, User, Warning, Files, Timer, 
  Menu, EditPen, UploadFilled, Monitor, TrendCharts, Bell 
} from '@element-plus/icons-vue'

const greeting = computed(() => {
  const hour = new Date().getHours()
  if (hour < 9) return '早上好'
  if (hour < 12) return '上午好'
  if (hour < 14) return '中午好'
  if (hour < 18) return '下午好'
  return '晚上好'
})

// 真实统计数据
const stats = ref({ total: 0, high_risk: 0, ratio: 0, update_time: '加载中...' })

onMounted(async () => {
  try {
    const res = await api.get('/dashboard/stats')
    if (res.data) stats.value = res.data
  } catch (e) {
    console.error('获取统计数据失败', e)
  }
})
</script>

<style scoped>
.dashboard-container { padding-bottom: 20px; }

.welcome-banner {
  background: linear-gradient(to right, #2b4b6b, #1f2f3d);
  color: white; padding: 30px 40px; border-radius: 8px;
  margin-bottom: 24px; display: flex; justify-content: space-between;
  align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  position: relative; overflow: hidden;
}
.banner-text h2 { margin: 0 0 10px 0; font-size: 24px; }
.banner-text p { margin: 0; opacity: 0.8; font-size: 14px; }
.banner-img { opacity: 0.1; transform: rotate(-15deg) scale(1.5); position: absolute; right: 20px; }

.stat-row { margin-bottom: 20px; }
.stat-card { border: none; border-radius: 8px; transition: transform 0.3s; }
.stat-card:hover { transform: translateY(-4px); }
.stat-content { display: flex; align-items: center; }
.stat-icon {
  width: 50px; height: 50px; border-radius: 50%;
  display: flex; justify-content: center; align-items: center;
  margin-right: 15px; overflow: hidden;
}
.stat-img { width: 100%; height: 100%; object-fit: cover; }
.stat-label { color: #909399; font-size: 13px; margin-bottom: 4px; }
.stat-value .num { font-size: 24px; font-weight: bold; color: #303133; }
.stat-value .unit { font-size: 12px; margin-left: 4px; color: #909399; }

.shortcut-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; padding: 10px 0; }
.shortcut-item {
  text-align: center; cursor: pointer; padding: 20px; border-radius: 8px;
  transition: all 0.3s; background-color: #f8f9fa;
}
.shortcut-item:hover { background-color: #fff; box-shadow: 0 6px 16px rgba(0,0,0,0.08); transform: translateY(-2px); }
.sc-icon {
  width: 48px; height: 48px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 12px; color: white; font-size: 20px;
}
.bg-blue { background: #409EFF; }
.bg-green { background: #67C23A; }
.bg-purple { background: #a265e6; }
.bg-orange { background: #E6A23C; }
.shortcut-item h4 { margin: 0; font-size: 15px; color: #303133; }
.shortcut-item p { margin: 5px 0 0; font-size: 12px; color: #909399; }

@media (max-width: 768px) { .shortcut-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
