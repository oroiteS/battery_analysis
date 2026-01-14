<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { User, Lock, Message, Lightning } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { login, register, getMe } from '../api/auth'

defineOptions({
  name: 'LoginView',
})

const router = useRouter()
const loginFormRef = ref()
const registerFormRef = ref()

// Toggle between Login and Register
const isRegister = ref(false)

const loginForm = reactive({
  username: '',
  password: '',
})

const registerForm = reactive({
  user_name: '',
  email: '',
  password: '',
  confirmPassword: '',
})

const loginRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const validatePass2 = (rule: any, value: any, callback: any) => {
  if (value === '') {
    callback(new Error('请再次输入密码'))
  } else if (value !== registerForm.password) {
    callback(new Error('两次输入密码不一致!'))
  } else {
    callback()
  }
}

const registerRules = {
  user_name: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  email: [
    { required: true, message: '请输入邮箱', trigger: 'blur' },
    { type: 'email', message: '请输入正确的邮箱地址', trigger: ['blur', 'change'] },
  ],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
  confirmPassword: [{ validator: validatePass2, trigger: 'blur' }],
}

const loading = ref(false)

const handleLogin = async () => {
  if (!loginFormRef.value) return

  await loginFormRef.value.validate(async (valid: boolean) => {
    if (valid) {
      loading.value = true
      try {
        const tokenRes = await login({
          username: loginForm.username,
          password: loginForm.password
        })

        localStorage.setItem('token', tokenRes.access_token)

        // Fetch user info
        const userRes = await getMe()
        localStorage.setItem('username', userRes.user_name)
        localStorage.setItem('email', userRes.email)

        ElMessage.success('欢迎回来')
        router.push('/')
      } catch (error) {
        // Error is handled by interceptor
        console.error(error)
      } finally {
        loading.value = false
      }
    }
  })
}

const handleRegister = async () => {
  if (!registerFormRef.value) return

  await registerFormRef.value.validate(async (valid: boolean) => {
    if (valid) {
      loading.value = true
      try {
        await register({
          user_name: registerForm.user_name,
          email: registerForm.email,
          password: registerForm.password
        })
        ElMessage.success('注册成功，请登录')
        isRegister.value = false
      } catch (error) {
        console.error(error)
      } finally {
        loading.value = false
      }
    }
  })
}

const toggleMode = () => {
  isRegister.value = !isRegister.value
  // Reset forms
  if (loginFormRef.value) loginFormRef.value.resetFields()
  if (registerFormRef.value) registerFormRef.value.resetFields()
}
</script>

<template>
  <div class="login-container">
    <div class="login-content">
      <div class="brand-section">
        <div class="logo-circle">
          <el-icon :size="24" color="white"><Lightning /></el-icon>
        </div>
        <h1 class="brand-title">BatteryAI</h1>
        <p class="brand-subtitle">储能电池寿命分析平台</p>
      </div>

      <div class="form-section">
        <h2 class="form-title">{{ isRegister ? '注册账号' : '登录' }}</h2>
        <p class="form-subtitle">
          {{ isRegister ? '输入您的详细信息以开始使用' : '欢迎回来，请输入您的账号信息' }}
        </p>

        <!-- Login Form -->
        <el-form
          v-if="!isRegister"
          ref="loginFormRef"
          :model="loginForm"
          :rules="loginRules"
          class="auth-form"
          size="large"
          @submit.prevent
        >
          <el-form-item prop="username">
            <el-input 
              v-model="loginForm.username" 
              placeholder="用户名或邮箱" 
              :prefix-icon="User" 
            />
          </el-form-item>

          <el-form-item prop="password">
            <el-input
              v-model="loginForm.password"
              type="password"
              placeholder="密码"
              :prefix-icon="Lock"
              show-password
              @keyup.enter="handleLogin"
            />
          </el-form-item>

          <el-button type="primary" :loading="loading" class="submit-button" @click="handleLogin">
            登录
          </el-button>

          <div class="form-footer">
            <span class="text-muted">还没有账号？</span>
            <el-button link type="primary" @click="toggleMode">去注册</el-button>
          </div>
        </el-form>

        <!-- Register Form -->
        <el-form
          v-else
          ref="registerFormRef"
          :model="registerForm"
          :rules="registerRules"
          class="auth-form"
          size="large"
          @submit.prevent
        >
          <el-form-item prop="user_name">
            <el-input v-model="registerForm.user_name" placeholder="用户名" :prefix-icon="User" />
          </el-form-item>

          <el-form-item prop="email">
            <el-input v-model="registerForm.email" placeholder="邮箱" :prefix-icon="Message" />
          </el-form-item>

          <el-form-item prop="password">
            <el-input
              v-model="registerForm.password"
              type="password"
              placeholder="密码"
              :prefix-icon="Lock"
              show-password
            />
          </el-form-item>

          <el-form-item prop="confirmPassword">
            <el-input
              v-model="registerForm.confirmPassword"
              type="password"
              placeholder="确认密码"
              :prefix-icon="Lock"
              show-password
              @keyup.enter="handleRegister"
            />
          </el-form-item>

          <el-button type="primary" :loading="loading" class="submit-button" @click="handleRegister">
            创建账号
          </el-button>

          <div class="form-footer">
            <span class="text-muted">已有账号？</span>
            <el-button link type="primary" @click="toggleMode">去登录</el-button>
          </div>
        </el-form>
      </div>
    </div>
  </div>
</template>

<style scoped>
.login-container {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #F3F4F6;
}

.login-content {
  display: flex;
  background: white;
  border-radius: 24px;
  overflow: hidden;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.01);
  width: 900px;
  max-width: 95%;
  height: 600px;
}

.brand-section {
  flex: 1;
  background-color: var(--color-primary);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: white;
  padding: 40px;
  position: relative;
  overflow: hidden;
}

/* Abstract shapes for decoration */
.brand-section::before {
  content: '';
  position: absolute;
  top: -50px;
  left: -50px;
  width: 200px;
  height: 200px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 50%;
}

.brand-section::after {
  content: '';
  position: absolute;
  bottom: -50px;
  right: -50px;
  width: 300px;
  height: 300px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 50%;
}

.logo-circle {
  width: 64px;
  height: 64px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
  backdrop-filter: blur(10px);
}

.brand-title {
  font-size: 32px;
  font-weight: 700;
  margin: 0 0 12px 0;
  letter-spacing: -0.02em;
}

.brand-subtitle {
  font-size: 16px;
  opacity: 0.7;
  font-weight: 300;
  margin: 0;
  text-align: center;
  max-width: 80%;
}

.form-section {
  flex: 1;
  padding: 60px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.form-title {
  font-size: 24px;
  font-weight: 600;
  color: var(--color-text-main);
  margin: 0 0 8px 0;
}

.form-subtitle {
  color: var(--color-text-secondary);
  font-size: 14px;
  margin: 0 0 32px 0;
}

.auth-form {
  width: 100%;
}

.submit-button {
  width: 100%;
  height: 44px;
  font-size: 14px;
  font-weight: 600;
  margin-top: 16px;
}

.form-footer {
  margin-top: 24px;
  text-align: center;
  font-size: 14px;
}

.text-muted {
  color: var(--color-text-secondary);
  margin-right: 8px;
}

@media (max-width: 768px) {
  .login-content {
    flex-direction: column;
    height: auto;
    width: 100%;
    margin: 20px;
    border-radius: 16px;
  }
  
  .brand-section {
    padding: 30px;
    min-height: 200px;
  }
  
  .form-section {
    padding: 30px;
  }
}
</style>
