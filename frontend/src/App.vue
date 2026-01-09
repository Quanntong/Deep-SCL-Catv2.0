<template>
  <el-container class="layout-container">
    <el-aside width="260px" class="aside-menu">
      <div class="logo-area">
        <img src="@/assets/icon_wide.png" alt="Logo" class="school-logo" />
        <div class="logo-text-group">
          <h1 class="system-title">心理风险预测</h1>
          <span class="system-sub">Intelligent System</span>
        </div>
      </div>
      
      <el-menu
        :default-active="$route.path"
        class="el-menu-vertical"
        background-color="transparent"
        text-color="#bfcbd9"
        active-text-color="#ffffff"
        router
        :unique-opened="true"
      >
        <el-menu-item index="/">
          <el-icon><HomeFilled /></el-icon>
          <span>系统首页</span>
        </el-menu-item>
        
        <el-menu-item index="/test">
          <el-icon><Monitor /></el-icon>
          <span>在线测评 (90题)</span>
        </el-menu-item>

        <el-menu-item index="/manual">
          <el-icon><EditPen /></el-icon>
          <span>手动因子录入</span>
        </el-menu-item>

        <el-menu-item index="/batch">
          <el-icon><UploadFilled /></el-icon>
          <span>批量Excel分析</span>
        </el-menu-item>

        <el-menu-item index="/result">
          <el-icon><DataAnalysis /></el-icon>
          <span>风险分析报告</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container class="main-wrapper">
      <el-header class="main-header">
        <div class="header-left">
          <el-breadcrumb separator-class="el-icon-arrow-right">
            <el-breadcrumb-item :to="{ path: '/' }">
              <span class="crumb-text">首页</span>
            </el-breadcrumb-item>
            <el-breadcrumb-item>
              <span class="crumb-current">{{ currentRouteName }}</span>
            </el-breadcrumb-item>
          </el-breadcrumb>
        </div>
        
        <div class="header-right">
          <div class="user-profile">
            <el-avatar :size="38" icon="UserFilled" class="user-avatar" />
            <div class="user-info">
              <span class="username">管理员</span>
              <span class="role-badge">Super Admin</span>
            </div>
          </div>
        </div>
      </el-header>

      <el-main class="main-content">
        <router-view v-slot="{ Component }">
          <transition name="fade-slide" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { HomeFilled, Monitor, EditPen, UploadFilled, DataAnalysis } from '@element-plus/icons-vue'

const route = useRoute()

const currentRouteName = computed(() => {
  const map = {
    '/': '仪表盘',
    '/test': '在线测评',
    '/manual': '手动录入',
    '/batch': '批量上传',
    '/result': '分析结果'
  }
  return map[route.path] || '当前页面'
})
</script>

<style>
/* 全局样式重置与字体优化 */
body, html {
  margin: 0;
  padding: 0;
  height: 100%;
  font-family: 'Inter', -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background-color: #f3f6f9; /* 更柔和的护眼灰 */
  -webkit-font-smoothing: antialiased;
}
#app { height: 100%; }

/* 滚动条美化 */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #c0c4cc; border-radius: 4px; }
::-webkit-scrollbar-track { background: #f1f1f1; }
</style>

<style scoped>
.layout-container { height: 100vh; display: flex; }

/* === 1. 侧边栏深度美化 === */
.aside-menu {
  /* 使用深邃的渐变蓝，看起来更和谐、高级 */
  background: linear-gradient(180deg, #182b46 0%, #0d1b2a 100%);
  color: white;
  display: flex;
  flex-direction: column;
  box-shadow: 4px 0 15px rgba(0,0,0,0.15); /* 侧边栏立体投影 */
  z-index: 20;
}

/* Logo 区域：背景与侧边栏融合 */
.logo-area {
  height: 86px; /* 加高区域 */
  display: flex;
  align-items: center;
  padding: 0 20px;
  /* 底部增加一条若隐若现的分隔线 */
  border-bottom: 1px solid rgba(255,255,255,0.06);
  background: rgba(255,255,255,0.02); /* 极淡的提亮 */
}

.school-logo {
  height: 48px; /* 图片加大 */
  width: auto;
  max-width: 120px;
  object-fit: contain;
  margin-right: 12px;
  filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2)); /* 图片加一点投影 */
}

/* 文字区域 */
.logo-text-group {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.system-title {
  margin: 0;
  font-size: 20px; /* 字号加大！ */
  font-weight: 700;
  color: #fff;
  letter-spacing: 0.5px;
  white-space: nowrap;
  line-height: 1.2;
}

.system-sub {
  font-size: 11px;
  color: #8b9eb9; /* 灰蓝色副标题 */
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-top: 2px;
  font-weight: 500;
}

/* 菜单项圆角化设计 */
.el-menu-vertical {
  border-right: none;
  padding: 12px; /* 给菜单留出边距 */
}

:deep(.el-menu-item) {
  height: 52px;
  line-height: 52px;
  margin-bottom: 6px; /* 菜单项间隔 */
  border-radius: 8px; /* 圆角 */
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid transparent;
}

/* 悬停效果 */
:deep(.el-menu-item:hover) {
  background-color: rgba(255,255,255,0.08) !important;
  transform: translateX(4px); /* 悬停微动 */
  color: #fff !important;
}

/* 选中态：蓝色渐变背景 + 发光 */
:deep(.el-menu-item.is-active) {
  background: linear-gradient(90deg, #409EFF 0%, #3a8ee6 100%) !important;
  color: white;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.3); /* 蓝色光晕 */
  font-weight: 600;
  border: 1px solid rgba(255,255,255,0.1);
}

:deep(.el-menu-item .el-icon) {
  font-size: 18px; /* 图标稍微加大 */
}

/* === 2. 顶部栏美化 === */
.main-header {
  background-color: #fff;
  box-shadow: 0 2px 8px rgba(0,0,0,0.03); /* 轻微阴影 */
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  height: 64px;
  z-index: 10;
}

.crumb-text { 
  color: #606266; 
  font-weight: normal; 
  transition: color 0.3s;
}
.crumb-text:hover { color: #409EFF; }

.crumb-current { 
  color: #182b46; 
  font-weight: 700; 
  font-size: 15px; 
  background: rgba(64, 158, 255, 0.08); /* 浅蓝底色 */
  padding: 4px 10px;
  border-radius: 4px;
  color: #409EFF;
}

.user-profile {
  display: flex;
  align-items: center;
  cursor: pointer;
  padding: 6px 12px;
  border-radius: 30px;
  transition: all 0.3s;
}

.user-profile:hover {
  background-color: #f5f7fa;
}

.user-avatar {
  border: 2px solid #eef1f6;
  background-color: #409EFF;
  transition: transform 0.3s;
}
.user-profile:hover .user-avatar {
  transform: scale(1.05);
}

.user-info {
  margin-left: 10px;
  display: flex;
  flex-direction: column;
  line-height: 1.3;
}

.username { font-size: 14px; font-weight: 600; color: #333; }
.role-badge { font-size: 10px; color: #909399; }

/* === 3. 内容区与动画 === */
.main-content {
  padding: 24px;
  background-color: #f3f6f9; /* 浅灰底色突出卡片 */
  overflow-x: hidden;
}

/* 页面切换动画：平滑上浮淡入 */
.fade-slide-leave-active,
.fade-slide-enter-active {
  transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
}

.fade-slide-enter-from {
  opacity: 0;
  transform: translateY(15px);
}

.fade-slide-leave-to {
  opacity: 0;
  transform: translateY(-15px);
}
</style>