import { createRouter, createWebHistory } from 'vue-router'
import MainLayout from '../layout/MainLayout.vue'
import HomeView from '../views/HomeView.vue'
import Login from '../views/Login.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: Login,
      meta: { title: '登录', requiresAuth: false }
    },
    {
      path: '/',
      component: MainLayout,
      children: [
        {
          path: '',
          name: 'home',
          component: HomeView,
          meta: { title: '仪表盘', requiresAuth: true }
        },
        {
          path: 'analysis',
          name: 'analysis',
          component: () => import('../views/DataAnalysis.vue'),
          meta: { title: '数据管理与分析', requiresAuth: true }
        },
        {
          path: 'training',
          name: 'training',
          component: () => import('../views/Training.vue'),
          meta: { title: '算法训练平台', requiresAuth: true }
        },
        {
          path: 'prediction',
          name: 'prediction',
          component: () => import('../views/Prediction.vue'),
          meta: { title: '算法测试与预测', requiresAuth: true }
        }
      ]
    }
  ],
})

// Navigation Guard
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  if (to.meta.requiresAuth && !token) {
    next('/login')
  } else {
    console.log('token:', token)
    next()
  }
})

export default router
