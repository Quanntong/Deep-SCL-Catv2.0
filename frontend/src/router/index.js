import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'home', component: () => import('@/views/HomeView.vue') },
  { path: '/test', name: 'test', component: () => import('@/views/TestView.vue') },
  { path: '/manual', name: 'manual', component: () => import('@/views/ManualView.vue') },
  { path: '/batch', name: 'batch', component: () => import('@/views/BatchView.vue') },
  { path: '/result', name: 'result', component: () => import('@/views/ResultView.vue') }
]

export default createRouter({
  history: createWebHistory(),
  routes
})
