<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { getDatasets, getBatteries } from '../api/data'
import { getAlgorithms } from '../api/models'
import { createTrainingJob, listTrainingJobs } from '../api/training'
import type { Dataset, BatteryUnit, Algorithm, CreateTrainingJobRequest, BatterySelection, TrainingJobResponse } from '../api/types'

defineOptions({
  name: 'TrainingView'
})

// --- State ---
const datasets = ref<Dataset[]>([])
const batteries = ref<BatteryUnit[]>([])
const algorithms = ref<Algorithm[]>([])
const trainingJobs = ref<TrainingJobResponse[]>([])

const form = reactive({
  datasetId: null as number | null,
  target: 'RUL',
  selectedAlgorithms: [] as string[],
  batteries: [] as BatterySelection[],
  // Hyperparameters
  num_epoch: 2000,
  batch_size: 1024,
  lr: 0.001,
  dropout_rate: 0.2,
  seq_len: 1,
  perc_val: 0.2
})

// Battery Selection Table Data
interface BatteryRow extends BatteryUnit {
  role: 'train' | 'val' | 'test' | 'ignore'
}
const batteryTableData = ref<BatteryRow[]>([])

const isSubmitting = ref(false)
const activeTab = ref('config') // 'config' | 'monitor'

// --- Methods ---

const initData = async () => {
  try {
    const [datasetsRes, algosRes, jobsRes] = await Promise.all([
      getDatasets(),
      getAlgorithms(),
      listTrainingJobs({ limit: 10 })
    ])
    datasets.value = datasetsRes
    algorithms.value = algosRes.algorithms
    trainingJobs.value = jobsRes

    if (datasets.value.length > 0) {
      form.datasetId = datasets.value[0].id
      await handleDatasetChange()
    }
  } catch (error) {
    console.error('Init failed:', error)
  }
}

const handleDatasetChange = async () => {
  if (!form.datasetId) return
  try {
    const res = await getBatteries(form.datasetId)
    batteries.value = res
    // Initialize battery table with default roles
    // Simple logic: 60% train, 20% val, 20% test
    const total = res.length
    const trainCount = Math.floor(total * 0.6)
    const valCount = Math.floor(total * 0.2)

    batteryTableData.value = res.map((b, index) => {
      let role: 'train' | 'val' | 'test' = 'test'
      if (index < trainCount) role = 'train'
      else if (index < trainCount + valCount) role = 'val'

      return { ...b, role }
    })
  } catch (error) {
    console.error('Fetch batteries failed:', error)
  }
}

const handleSubmit = async () => {
  if (!form.datasetId) {
    ElMessage.warning('请选择数据集')
    return
  }
  if (form.selectedAlgorithms.length === 0) {
    ElMessage.warning('请至少选择一个算法')
    return
  }

  // Filter selected batteries
  const selectedBatteries = batteryTableData.value
    .filter(b => b.role !== 'ignore')
    .map(b => ({
      battery_id: b.id,
      split_role: b.role as 'train' | 'val' | 'test'
    }))

  if (selectedBatteries.length === 0) {
    ElMessage.warning('请至少选择一个电池用于训练')
    return
  }

  isSubmitting.value = true
  try {
    const payload: CreateTrainingJobRequest = {
      dataset_id: form.datasetId,
      target: form.target as 'RUL' | 'PCL' | 'BOTH',
      algorithms: form.selectedAlgorithms,
      batteries: selectedBatteries,
      num_epoch: form.num_epoch,
      batch_size: form.batch_size,
      lr: form.lr,
      dropout_rate: form.dropout_rate,
      seq_len: form.seq_len,
      perc_val: form.perc_val
    }

    await createTrainingJob(payload)
    ElMessage.success('训练任务创建成功')
    // Refresh job list and switch tab
    const jobsRes = await listTrainingJobs({ limit: 10 })
    trainingJobs.value = jobsRes
    activeTab.value = 'monitor'
  } catch (error) {
    console.error('Create job failed:', error)
    ElMessage.error('创建任务失败')
  } finally {
    isSubmitting.value = false
  }
}

const getStatusType = (status: string) => {
  switch (status) {
    case 'SUCCEEDED': return 'success'
    case 'FAILED': return 'danger'
    case 'RUNNING': return 'primary'
    default: return 'info'
  }
}

onMounted(() => {
  initData()
})
</script>

<template>
  <div class="training-container">
    <el-tabs v-model="activeTab" type="border-card">
      <!-- Tab 1: Configuration -->
      <el-tab-pane label="新建训练任务" name="config">
        <el-row :gutter="20">
          <!-- Left: Basic Config -->
          <el-col :span="12">
            <el-card shadow="never" header="基础配置">
              <el-form :model="form" label-width="100px">
                <el-form-item label="数据集">
                  <el-select v-model="form.datasetId" style="width: 100%" @change="handleDatasetChange">
                    <el-option v-for="d in datasets" :key="d.id" :label="d.name" :value="d.id" />
                  </el-select>
                </el-form-item>

                <el-form-item label="预测目标">
                  <el-radio-group v-model="form.target">
                    <el-radio label="RUL">RUL (剩余寿命)</el-radio>
                    <el-radio label="PCL">PCL (预测容量)</el-radio>
                  </el-radio-group>
                </el-form-item>

                <el-form-item label="选择算法">
                  <el-checkbox-group v-model="form.selectedAlgorithms">
                    <el-checkbox v-for="algo in algorithms" :key="algo.code" :label="algo.code">
                      {{ algo.name }}
                    </el-checkbox>
                  </el-checkbox-group>
                </el-form-item>
              </el-form>
            </el-card>

            <el-card shadow="never" header="超参数设置" class="mt-20">
              <el-form :model="form" label-width="120px">
                <el-row :gutter="20">
                  <el-col :span="12">
                    <el-form-item label="Epochs">
                      <el-input-number v-model="form.num_epoch" :min="1" :max="10000" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="Batch Size">
                      <el-select v-model="form.batch_size">
                        <el-option :value="32" label="32" />
                        <el-option :value="64" label="64" />
                        <el-option :value="128" label="128" />
                        <el-option :value="256" label="256" />
                        <el-option :value="1024" label="1024" />
                      </el-select>
                    </el-form-item>
                  </el-col>
                </el-row>
                <el-row :gutter="20">
                  <el-col :span="12">
                    <el-form-item label="Learning Rate">
                      <el-input-number v-model="form.lr" :step="0.0001" :min="0.00001" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="Dropout">
                      <el-input-number v-model="form.dropout_rate" :step="0.1" :min="0" :max="0.9" />
                    </el-form-item>
                  </el-col>
                </el-row>
              </el-form>
            </el-card>
          </el-col>

          <!-- Right: Battery Selection -->
          <el-col :span="12">
            <el-card shadow="never" header="电池数据集划分">
              <el-table :data="batteryTableData" height="500" border stripe>
                <el-table-column prop="battery_code" label="电池编号" width="120" />
                <el-table-column prop="total_cycles" label="循环次数" width="100" />
                <el-table-column label="用途划分">
                  <template #default="scope">
                    <el-select v-model="scope.row.role" size="small">
                      <el-option label="训练集 (Train)" value="train" />
                      <el-option label="验证集 (Val)" value="val" />
                      <el-option label="测试集 (Test)" value="test" />
                      <el-option label="忽略 (Ignore)" value="ignore" />
                    </el-select>
                  </template>
                </el-table-column>
              </el-table>
            </el-card>
          </el-col>
        </el-row>

        <div class="action-bar mt-20">
          <el-button type="primary" size="large" :loading="isSubmitting" @click="handleSubmit">
            开始训练任务
          </el-button>
        </div>
      </el-tab-pane>

      <!-- Tab 2: Monitor -->
      <el-tab-pane label="任务监控" name="monitor">
        <el-table :data="trainingJobs" border stripe>
          <el-table-column prop="id" label="ID" width="80" />
          <el-table-column prop="created_at" label="创建时间" width="180">
            <template #default="scope">
              {{ new Date(scope.row.created_at).toLocaleString() }}
            </template>
          </el-table-column>
          <el-table-column prop="target" label="目标" width="80" />
          <el-table-column label="状态" width="120">
            <template #default="scope">
              <el-tag :type="getStatusType(scope.row.status)">{{ scope.row.status }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="进度">
            <template #default="scope">
              <el-progress :percentage="Math.round(scope.row.progress * 100)" />
            </template>
          </el-table-column>
          <el-table-column label="操作" width="150">
            <template #default>
              <el-button size="small" type="primary" link>查看详情</el-button>
              <el-button size="small" type="danger" link>删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<style scoped>
.mt-20 {
  margin-top: 20px;
}
.action-bar {
  display: flex;
  justify-content: center;
  padding: 20px 0;
}
</style>
