<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { listTrainingJobs } from '../api/training'
import { listTestJobs } from '../api/testing'
import type { TrainingJobResponse } from '../api/types'

const router = useRouter()

// 统计数据
const stats = ref([
  { title: '电池总数', value: 124, icon: 'DataLine', color: '#409EFF' },
  { title: '训练任务', value: 0, icon: 'Cpu', color: '#67C23A' },
  { title: '测试任务', value: 0, icon: 'TrendCharts', color: '#E6A23C' },
])

// 最近训练任务
const recentTrainingJobs = ref<TrainingJobResponse[]>([])
const trainingLoading = ref(false)

// 最近测试任务
const recentTestJobs = ref<any[]>([])
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
    stats.value[1].value = allTrainingRes.length
    recentTrainingJobs.value = allTrainingRes.slice(0, 5)

    // 加载测试任务总数和最近任务
    testLoading.value = true
    const allTestRes = await listTestJobs()
    stats.value[2].value = allTestRes.length
    recentTestJobs.value = allTestRes.slice(0, 5)
  } catch (error: any) {
    console.error('加载工作台数据失败:', error)
    ElMessage.error('加载数据失败')
  } finally {
    trainingLoading.value = false
    testLoading.value = false
  }
}

// 格式化日期
const formatDate = (dateStr: string) => {
  return new Date(dateStr).toLocaleString('zh-CN')
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
    <!-- 统计卡片 -->
    <el-row :gutter="20">
      <el-col :span="8" v-for="item in stats" :key="item.title">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-icon" :style="{ backgroundColor: item.color }">
              <el-icon><component :is="item.icon" /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ item.value }}</div>
              <div class="stat-title">{{ item.title }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 最近训练任务 -->
    <el-card shadow="hover" class="section-card" header="最近训练任务">
      <template #extra>
        <el-button type="primary" link @click="goToTraining">查看全部</el-button>
      </template>
      <el-table
        :data="recentTrainingJobs"
        v-loading="trainingLoading"
        empty-text="暂无训练任务"
        style="width: 100%"
      >
        <el-table-column prop="id" label="任务ID" width="80" />
        <el-table-column prop="created_at" label="创建时间" width="180">
          <template #default="scope">
            {{ formatDate(scope.row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column prop="target" label="目标" width="100">
          <template #default="scope">
            <el-tag>{{ scope.row.target }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="120">
          <template #default="scope">
            <el-tag
              :type="
                scope.row.status === 'SUCCEEDED'
                  ? 'success'
                  : scope.row.status === 'RUNNING'
                    ? 'primary'
                    : scope.row.status === 'FAILED'
                      ? 'danger'
                      : 'info'
              "
            >
              {{ scope.row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="progress" label="进度" width="200">
          <template #default="scope">
            <el-progress :percentage="Math.round(scope.row.progress * 100)" />
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120">
          <template #default="scope">
            <el-button type="primary" link @click="viewTrainingDetail(scope.row.id)">
              查看详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 最近测试任务 -->
    <el-card shadow="hover" class="section-card" header="最近测试任务">
      <template #extra>
        <el-button type="primary" link @click="goToTesting">查看全部</el-button>
      </template>
      <el-table
        :data="recentTestJobs"
        v-loading="testLoading"
        empty-text="暂无测试任务"
        style="width: 100%"
      >
        <el-table-column prop="id" label="任务ID" width="80" />
        <el-table-column prop="created_at" label="创建时间" width="180">
          <template #default="scope">
            {{ formatDate(scope.row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column prop="model_name" label="模型" width="200">
          <template #default="scope">
            <el-tag type="success">{{ scope.row.model_name || 'N/A' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="120">
          <template #default="scope">
            <el-tag
              :type="
                scope.row.status === 'SUCCEEDED'
                  ? 'success'
                  : scope.row.status === 'RUNNING'
                    ? 'primary'
                    : scope.row.status === 'FAILED'
                      ? 'danger'
                      : 'info'
              "
            >
              {{ scope.row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120">
          <template #default="scope">
            <el-button type="primary" link @click="viewTestDetail(scope.row.id)">
              查看结果
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 快速操作 -->
    <el-card shadow="hover" class="section-card" header="快速操作">
      <div class="quick-actions">
        <el-button type="primary" size="large" @click="goToTraining">
          <el-icon><Cpu /></el-icon>
          新建训练任务
        </el-button>
        <el-button type="success" size="large" @click="goToTesting">
          <el-icon><TrendCharts /></el-icon>
          新建测试任务
        </el-button>
        <el-button type="info" size="large" @click="goToDataAnalysis">
          <el-icon><DataLine /></el-icon>
          数据分析
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.workspace-container {
  padding: 0;
}

.stat-card {
  height: 100px;
  margin-bottom: 20px;
}

.stat-content {
  display: flex;
  align-items: center;
}

.stat-icon {
  width: 60px;
  height: 60px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 24px;
  margin-right: 15px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #303133;
}

.stat-title {
  font-size: 14px;
  color: #909399;
  margin-top: 5px;
}

.section-card {
  margin-top: 20px;
}

.quick-actions {
  display: flex;
  gap: 20px;
  justify-content: center;
  padding: 20px 0;
}

.quick-actions .el-button {
  min-width: 180px;
}
</style>
