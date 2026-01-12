import { createRouter, createWebHistory } from 'vue-router'
import MainLayout from '../layout/MainLayout.vue'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: MainLayout,
      children: [
        {
          path: '',
          name: 'home',
          component: HomeView,
          meta: { title: '仪表盘' }
        },
        {
          path: 'analysis',
          name: 'analysis',
          component: () => import('../views/DataAnalysis.vue'),
          meta: { title: '数据管理与分析' }
        },
        {
          path: 'training',
          name: 'training',
          component: () => import('../views/Training.vue'),
          meta: { title: '算法训练平台' }
        },
        {
          path: 'prediction',
          name: 'prediction',
          component: () => import('../views/Prediction.vue'),
          meta: { title: '算法测试与预测' }
        }
      ]
    }
  ],
})

export default router
