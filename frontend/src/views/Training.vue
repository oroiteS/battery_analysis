<script setup lang="ts">
import { ref, reactive } from 'vue'

defineOptions({
  name: 'TrainingView'
})

const trainingForm = reactive({
  model: 'BiLSTM',
  epochs: 100,
  batchSize: 32,
  learningRate: 0.001,
  optimizer: 'Adam',
  splitRatio: 0.8
})

const isTraining = ref(false)
const progress = ref(0)
const logs = ref<string[]>([])

const startTraining = () => {
  isTraining.value = true
  progress.value = 0
  logs.value = []
  logs.value.push(`[${new Date().toLocaleTimeString()}] 开始训练 ${trainingForm.model} 模型...`)

  // Mock training process
  const interval = setInterval(() => {
    progress.value += 10
    logs.value.push(`[${new Date().toLocaleTimeString()}] Epoch ${progress.value/10}: Loss = ${(Math.random() * 0.1).toFixed(4)}`)
    if (progress.value >= 100) {
      clearInterval(interval)
      isTraining.value = false
      logs.value.push(`[${new Date().toLocaleTimeString()}] 训练完成!`)
    }
  }, 500)
}
</script>

<template>
  <div class="training-container">
    <el-row :gutter="20">
      <el-col :span="8">
        <el-card shadow="hover" header="训练配置">
          <el-form :model="trainingForm" label-width="100px">
            <el-form-item label="算法模型">
              <el-select v-model="trainingForm.model" style="width: 100%">
                <el-option label="Baseline (Machine Learning)" value="Baseline" />
                <el-option label="BiLSTM (Deep Learning)" value="BiLSTM" />
                <el-option label="DeepHPM (Physics Informed)" value="DeepHPM" />
              </el-select>
            </el-form-item>

            <el-divider content-position="left">超参数设置</el-divider>

            <el-form-item label="训练轮次">
              <el-input-number v-model="trainingForm.epochs" :min="1" :max="1000" />
            </el-form-item>
            <el-form-item label="批次大小">
              <el-select v-model="trainingForm.batchSize" style="width: 100%">
                <el-option label="16" :value="16" />
                <el-option label="32" :value="32" />
                <el-option label="64" :value="64" />
                <el-option label="128" :value="128" />
              </el-select>
            </el-form-item>
            <el-form-item label="学习率">
              <el-input-number v-model="trainingForm.learningRate" :step="0.0001" :min="0.0001" />
            </el-form-item>
            <el-form-item label="优化器">
              <el-select v-model="trainingForm.optimizer" style="width: 100%">
                <el-option label="Adam" value="Adam" />
                <el-option label="SGD" value="SGD" />
                <el-option label="RMSprop" value="RMSprop" />
              </el-select>
            </el-form-item>
            <el-form-item label="数据集划分">
              <el-slider v-model="trainingForm.splitRatio" :step="0.1" :min="0.5" :max="0.9" show-input />
            </el-form-item>

            <el-form-item>
              <el-button type="primary" :loading="isTraining" @click="startTraining" style="width: 100%">
                {{ isTraining ? '训练中...' : '开始训练' }}
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>

      <el-col :span="16">
        <el-card shadow="hover" header="训练监控">
          <div v-if="isTraining || progress > 0">
             <el-progress :percentage="progress" :status="progress === 100 ? 'success' : ''" />
             <div class="log-container">
               <div v-for="(log, index) in logs" :key="index" class="log-item">{{ log }}</div>
             </div>
             <!-- TODO: Add Real-time Loss Chart here -->
             <div style="height: 300px; background: #f9f9f9; margin-top: 20px; display: flex; align-items: center; justify-content: center;">
               Loss / Accuracy Curve (Placeholder)
             </div>
          </div>
          <div v-else class="empty-state">
            <el-empty description="请配置参数并开始训练" />
          </div>
        </el-card>

        <el-card shadow="hover" header="模型评估结果" class="mt-20" v-if="progress === 100">
           <el-descriptions border>
             <el-descriptions-item label="RMSPE">1.25%</el-descriptions-item>
             <el-descriptions-item label="MSE">0.045</el-descriptions-item>
             <el-descriptions-item label="R²">0.98</el-descriptions-item>
           </el-descriptions>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.mt-20 {
  margin-top: 20px;
}
.log-container {
  height: 200px;
  overflow-y: auto;
  background: #2b2b2b;
  color: #00ff00;
  padding: 10px;
  font-family: monospace;
  margin-top: 15px;
  border-radius: 4px;
}
.log-item {
  margin-bottom: 5px;
}
</style>
