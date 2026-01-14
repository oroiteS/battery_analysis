<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { listTrainingJobs } from '../api/training'
import { listTestJobs } from '../api/testing'
import type { TrainingJobResponse, TestJobResponse } from '../api/types'

const router = useRouter()

// 统计数据
const stats = ref([
  { title: '电池总数', value: 124, icon: 'DataLine' },
  { title: '训练任务', value: 0, icon: 'Cpu' },
  { title: '测试任务', value: 0, icon: 'TrendCharts' },
])

// 最近训练任务
const recentTrainingJobs = ref<TrainingJobResponse[]>([])
const trainingLoading = ref(false)

// 最近测试任务
const recentTestJobs = ref<TestJobResponse[]>([])
const testLoading = ref(false)

// 初始化数据
onMounted(async () => {
  await loadDashboardData()
})

const loadDashboardData = async () => {
  try {
    // 加载训练任务总数和最近任务
    trainingLoading.value = true
    const allTrainingRes = await listTrainingJobs()
    if (stats.value[1]) stats.value[1].value = allTrainingRes.length
    recentTrainingJobs.value = allTrainingRes.slice(0, 5)

    // 加载测试任务总数和最近任务
    testLoading.value = true
    const allTestRes = await listTestJobs()
    if (stats.value[2]) stats.value[2].value = allTestRes.length
    recentTestJobs.value = allTestRes.slice(0, 5)
  } catch (error) {
    console.error('Failed to load dashboard data:', error)
    ElMessage.error('加载数据失败')
  } finally {
    trainingLoading.value = false
    testLoading.value = false
  }
}

// 格式化日期
const formatDate = (dateStr: string) => {
  return new Date(dateStr).toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 跳转到训练平台
const goToTraining = () => {
  router.push('/training')
}

// 跳转到测试平台
const goToTesting = () => {
  router.push('/prediction')
}

// 跳转到数据分析
const goToDataAnalysis = () => {
  router.push('/analysis')
}

// 查看训练任务详情
const viewTrainingDetail = (jobId: number) => {
  router.push({ path: '/training', query: { jobId: String(jobId), tab: 'monitor' } })
}

// 查看测试任务详情
const viewTestDetail = (jobId: number) => {
  router.push({ path: '/prediction', query: { jobId: String(jobId), tab: 'monitor' } })
}
</script>

<template>
  <div class="workspace-container">
    <!-- Stat Cards -->
    <div class="stats-grid">
      <el-card v-for="item in stats" :key="item.title" class="stat-card" shadow="hover">
        <div class="stat-content">
          <div class="stat-icon-wrapper">
            <el-icon class="stat-icon"><component :is="item.icon" /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ item.value }}</div>
            <div class="stat-title">{{ item.title }}</div>
          </div>
        </div>
      </el-card>
    </div>

    <!-- Quick Actions -->
    <div class="section-header">
      <h3>快捷操作</h3>
    </div>
    <div class="quick-actions-grid">
      <div class="action-card" @click="goToTraining">
        <div class="action-icon"><Cpu /></div>
        <div class="action-text">新建训练任务</div>
      </div>
      <div class="action-card" @click="goToTesting">
        <div class="action-icon"><TrendCharts /></div>
        <div class="action-text">新建测试任务</div>
      </div>
      <div class="action-card" @click="goToDataAnalysis">
        <div class="action-icon"><DataLine /></div>
        <div class="action-text">数据分析</div>
      </div>
    </div>

    <!-- Recent Training Jobs -->
    <el-card class="table-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span>最近训练任务</span>
          <el-button text @click="goToTraining">查看全部</el-button>
        </div>
      </template>
      <el-table
        :data="recentTrainingJobs"
        v-loading="trainingLoading"
        empty-text="暂无训练任务"
        style="width: 100%"
        :header-cell-style="{ background: 'transparent' }"
      >
        <el-table-column prop="id" label="ID" width="80">
          <template #default="scope">
            <span class="id-cell">#{{ scope.row.id }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="target" label="目标" width="120" />
        <el-table-column prop="created_at" label="创建日期">
          <template #default="scope">
            {{ formatDate(scope.row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="120">
          <template #default="scope">
            <div class="status-dot" 
              :class="{
                'success': scope.row.status === 'SUCCEEDED',
                'running': scope.row.status === 'RUNNING',
                'failed': scope.row.status === 'FAILED'
              }"
            ></div>
            {{ scope.row.status }}
          </template>
        </el-table-column>
        <el-table-column label="" width="120" align="right">
          <template #default="scope">
            <el-button type="primary" size="small" @click="viewTrainingDetail(scope.row.id)">
              查看详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- Recent Test Jobs -->
    <el-card class="table-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span>最近测试任务</span>
          <el-button text @click="goToTesting">查看全部</el-button>
        </div>
      </template>
      <el-table
        :data="recentTestJobs"
        v-loading="testLoading"
        empty-text="暂无测试任务"
        style="width: 100%"
        :header-cell-style="{ background: 'transparent' }"
      >
        <el-table-column prop="id" label="ID" width="80">
          <template #default="scope">
            <span class="id-cell">#{{ scope.row.id }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="target" label="目标" width="120" />
        <el-table-column prop="created_at" label="创建日期">
          <template #default="scope">
            {{ formatDate(scope.row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="120">
          <template #default="scope">
            <div class="status-dot" 
              :class="{
                'success': scope.row.status === 'SUCCEEDED',
                'running': scope.row.status === 'RUNNING',
                'failed': scope.row.status === 'FAILED'
              }"
            ></div>
            {{ scope.row.status }}
          </template>
        </el-table-column>
        <el-table-column label="" width="120" align="right">
          <template #default="scope">
            <el-button type="primary" size="small" @click="viewTestDetail(scope.row.id)">
              查看详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.workspace-container {
  max-width: 1200px;
  margin: 0 auto;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 24px;
  margin-bottom: 40px;
}

.stat-card {
  height: 100%;
}

:deep(.el-card__body) {
  padding: 24px;
}

.stat-content {
  display: flex;
  align-items: center;
  gap: 20px;
}

.stat-icon-wrapper {
  width: 64px;
  height: 64px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  transition: transform 0.3s ease;
  background-color: var(--color-text-main);
  color: var(--color-bg-page);
}

.stat-card:hover .stat-icon-wrapper {
  transform: scale(1.1);
}

.stat-info {
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 36px;
  font-weight: 700;
  line-height: 1.2;
  font-family: 'D-DIN', sans-serif;
  color: var(--color-text-main);
}

.stat-title {
  color: var(--color-text-secondary);
  font-size: 14px;
  font-weight: 500;
  margin-top: 4px;
}

.section-header {
  margin-bottom: 24px;
}

.section-header h3 {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-main);
  margin: 0;
}

.quick-actions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 24px;
  margin-bottom: 40px;
}

.action-card {
  background: var(--color-bg-surface);
  border: 1px solid var(--color-border);
  border-radius: 12px;
  padding: 24px;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  text-align: center;
}

.action-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
  border-color: var(--color-primary);
}

.action-icon {
  width: 48px;
  height: 48px;
  background: var(--color-bg-page);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: var(--color-text-main);
  transition: all 0.2s;
}

.action-card:hover .action-icon {
  background: var(--color-primary);
  color: white;
}

.action-text {
  font-weight: 500;
  color: var(--color-text-main);
}

.table-card {
  margin-bottom: 40px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.id-cell {
  font-family: monospace;
  color: var(--color-text-secondary);
  background: #F3F4F6;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}

.status-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 6px;
  background-color: var(--color-text-secondary);
}

.status-dot.success { background-color: var(--color-success); }
.status-dot.running { background-color: var(--color-primary); }
.status-dot.failed { background-color: var(--color-danger); }
</style>
