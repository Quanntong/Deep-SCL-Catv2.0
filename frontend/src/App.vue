<template>
  <el-container class="layout-container">
    <el-aside width="240px" class="aside-menu">
      <div class="logo-area">
        <img src="@/assets/icon_wide.png" alt="Logo" class="school-logo" />
      </div>
      
      <el-menu
        :default-active="$route.path"
        class="el-menu-vertical"
        background-color="#1f2f3d"
        text-color="#bfcbd9"
        active-text-color="#409EFF"
        router
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

    <el-container>
      <el-header class="main-header">
        <div class="header-left">
          <el-breadcrumb separator="/">
            <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>
            <el-breadcrumb-item>{{ currentRouteName }}</el-breadcrumb-item>
          </el-breadcrumb>
        </div>
        <div class="header-right">
          <div class="user-profile">
            <el-avatar :size="32" icon="UserFilled" style="background-color: #409EFF;" />
            <span class="username">管理员</span>
          </div>
        </div>
      </el-header>

      <el-main class="main-content">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
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
body, html {
  margin: 0;
  padding: 0;
  height: 100%;
  font-family: 'PingFang SC', 'Helvetica Neue', Arial, sans-serif;
  background-color: #f0f2f5;
}
#app { height: 100%; }
</style>

<style scoped>
.layout-container { height: 100vh; }

/* 侧边栏样式 */
.aside-menu {
  background-color: #1f2f3d;
  color: white;
  display: flex;
  flex-direction: column;
  box-shadow: 2px 0 6px rgba(0,0,0,0.1);
}

.logo-area {
  height: 80px; /* 增加高度以容纳学校Logo */
  display: flex;
  align-items: center;
  padding: 0 16px;
  background-color: #162432;
  border-bottom: 1px solid #2b3a4d;
}

.school-logo {
  width: 180px;
  height: auto;
  object-fit: contain;
}

.logo-text {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.school-name {
  font-size: 14px;
  font-weight: bold;
  color: #fff;
  letter-spacing: 1px;
}

.system-name {
  font-size: 12px;
  color: #909399;
  margin-top: 2px;
}

.el-menu-vertical { border-right: none; }

/* 顶部栏样式 */
.main-header {
  background-color: #fff;
  border-bottom: 1px solid #dcdfe6;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 60px;
}

.header-right { display: flex; align-items: center; }
.user-profile { display: flex; align-items: center; cursor: pointer; }
.username { margin-left: 8px; font-size: 14px; color: #606266; }

/* 内容区 */
.main-content {
  padding: 20px;
  overflow-y: auto;
}

/* 简单的淡入淡出动画 */
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
