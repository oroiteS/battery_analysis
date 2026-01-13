import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import Login from '../views/Login.vue'
import { createRouter, createWebHistory } from 'vue-router'
import ElementPlus from 'element-plus'

// Mock API
const mockLogin = vi.fn()
const mockRegister = vi.fn()
const mockGetMe = vi.fn()

vi.mock('../api/auth', () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  login: (data: any) => mockLogin(data),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  register: (data: any) => mockRegister(data),
  getMe: () => mockGetMe()
}))

// Mock Router
const router = createRouter({
  history: createWebHistory(),
  routes: [{ path: '/', component: { template: '<div>Home</div>' } }]
})

describe('Login.vue', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('renders login form by default', () => {
    const wrapper = mount(Login, {
      global: {
        plugins: [ElementPlus, router],
        stubs: {
          'el-icon': true
        }
      }
    })
    expect(wrapper.find('input[type="password"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('登录')
    expect(wrapper.text()).not.toContain('确认密码')
  })

  it('switches to register form', async () => {
    const wrapper = mount(Login, {
      global: {
        plugins: [ElementPlus, router],
        stubs: {
          'el-icon': true
        }
      }
    })

    await wrapper.find('.form-footer .el-link--primary').trigger('click')
    expect(wrapper.text()).toContain('注册')
    expect(wrapper.findAll('input[type="password"]').length).toBe(2) // Password + Confirm
  })

  it('handles login success', async () => {
    mockLogin.mockResolvedValue({ access_token: 'test-token', token_type: 'bearer' })
    mockGetMe.mockResolvedValue({ id: 1, user_name: 'testuser', email: 'test@example.com' })

    const wrapper = mount(Login, {
      global: {
        plugins: [ElementPlus, router],
        stubs: {
          'el-icon': true
        }
      }
    })

    // Fill form
    const inputs = wrapper.findAll('input')
    if (inputs[0]) await inputs[0].setValue('testuser')
    if (inputs[1]) await inputs[1].setValue('password123')

    // Click login
    await wrapper.find('.login-button').trigger('click')
    await flushPromises()

    expect(mockLogin).toHaveBeenCalledWith({
      username: 'testuser',
      password: 'password123'
    })
    expect(localStorage.getItem('token')).toBe('test-token')
    expect(localStorage.getItem('username')).toBe('testuser')
  })
})
