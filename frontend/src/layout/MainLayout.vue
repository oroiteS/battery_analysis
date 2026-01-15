<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { logout, getMe } from '../api/auth'

const route = useRoute()
const router = useRouter()
const username = ref('Admin')
const email = ref('')
const isDark = ref(false)

onMounted(async () => {
  // Theme initialization
  const savedTheme = localStorage.getItem('theme')
  if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    isDark.value = true
    document.documentElement.classList.add('dark')
  } else {
    isDark.value = false
    document.documentElement.classList.remove('dark')
  }

  try {
    const user = await getMe()
    username.value = user.user_name
    email.value = user.email
    // Sync to local storage
    localStorage.setItem('username', user.user_name)
    localStorage.setItem('email', user.email)
  } catch {
    // Fallback to local storage if API fails
    const storedUser = localStorage.getItem('username')
    if (storedUser) {
      username.value = storedUser
    }
    const storedEmail = localStorage.getItem('email')
    if (storedEmail) {
      email.value = storedEmail
    }
  }
})

const toggleTheme = () => {
  isDark.value = !isDark.value
  if (isDark.value) {
    document.documentElement.classList.add('dark')
    localStorage.setItem('theme', 'dark')
  } else {
    document.documentElement.classList.remove('dark')
    localStorage.setItem('theme', 'light')
  }
}

const handleLogout = () => {
  ElMessageBox.confirm('确定要退出登录吗？', '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning',
  })
    .then(async () => {
      try {
        await logout()
      } catch (e) {
        console.error('Logout failed', e)
      } finally {
        localStorage.removeItem('token')
        localStorage.removeItem('username')
        localStorage.removeItem('email')
        ElMessage.success('已退出登录')
        router.push('/login')
      }
    })
    .catch(() => {
      // cancel
    })
}
</script>

<template>
  <el-container class="layout-container">
    <el-aside width="260px">
      <div class="logo">
        <div class="logo-icon">
          <el-icon :size="20"><Lightning /></el-icon>
        </div>
        <span class="logo-text">BATTERY<span class="logo-highlight">AI</span></span>
      </div>
      
      <div class="menu-wrapper">
        <el-menu
          :default-active="route.path"
          router
          class="el-menu-vertical"
          :collapse-transition="false"
        >
          <div class="menu-label">菜单</div>
          <el-menu-item index="/">
            <el-icon><Odometer /></el-icon>
            <span>仪表盘</span>
          </el-menu-item>
          <el-menu-item index="/analysis">
            <el-icon><DataLine /></el-icon>
            <span>数据分析</span>
          </el-menu-item>
          <el-menu-item index="/training">
            <el-icon><Cpu /></el-icon>
            <span>模型训练</span>
          </el-menu-item>
          <el-menu-item index="/prediction">
            <el-icon><TrendCharts /></el-icon>
            <span>状态预测</span>
          </el-menu-item>
        </el-menu>
      </div>
      
      <div class="user-profile-mini">
        <el-avatar :size="32" class="user-avatar">{{ username.charAt(0).toUpperCase() }}</el-avatar>
        <div class="user-details">
          <span class="user-name">{{ username }}</span>
          <span class="user-role">{{ email }}</span>
        </div>
        <el-dropdown trigger="click" @command="(c: string | number | object) => c === 'logout' && handleLogout()">
          <el-icon class="settings-icon"><Setting /></el-icon>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="logout">退出登录</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </el-aside>

    <el-container>
      <el-header>
        <div class="header-content">
          <div class="page-title">{{ route.meta.title || 'Dashboard' }}</div>
          <div class="header-actions">
            <el-button
              circle
              text
              @click="toggleTheme"
              class="theme-toggle"
            >
              <el-icon :size="20" :color="isDark ? '#F59E0B' : '#606266'">
                <Sunny v-if="isDark" />
                <Moon v-else />
              </el-icon>
            </el-button>
          </div>
        </div>
      </el-header>

      <el-main>
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.layout-container {
  height: 100vh;
  background-color: var(--color-bg-page);
}

.el-aside {
  background-color: var(--color-bg-surface);
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  z-index: 10;
}

.logo {
  height: 80px;
  display: flex;
  align-items: center;
  padding: 0 24px;
  gap: 12px;
}

.logo-icon {
  width: 32px;
  height: 32px;
  background: var(--color-primary);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-bg-page);
}

.logo-icon .el-icon {
  color: var(--color-bg-page);
}

.logo-text {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.05em;
  color: var(--color-text-main);
}

.logo-highlight {
  font-weight: 300;
  margin-left: 2px;
}

.menu-wrapper {
  flex: 1;
  padding: 0 12px;
  overflow-y: auto;
}

.menu-label {
  padding: 16px 12px 8px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  color: var(--color-text-secondary);
  text-transform: uppercase;
}

.el-header {
  height: 80px;
  padding: 0 40px;
  background: transparent;
}

.header-content {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  color: var(--color-text-main);
  letter-spacing: -0.02em;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 16px;
}

.theme-toggle:hover {
  background-color: rgba(0, 0, 0, 0.05);
}

.el-main {
  padding: 0 40px 40px;
  overflow-x: hidden;
}

.user-profile-mini {
  margin: 24px;
  padding: 16px;
  background: #F9FAFB;
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.user-avatar {
  background: var(--color-primary);
  color: var(--color-bg-page);
  font-weight: 600;
  font-size: 14px;
}

.user-details {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.user-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-main);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.user-role {
  font-size: 11px;
  color: var(--color-text-secondary);
}

.settings-icon {
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: color 0.2s;
}

.settings-icon:hover {
  color: var(--color-primary);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.fade-enter-from {
  opacity: 0;
  transform: translateY(10px);
}

.fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>
