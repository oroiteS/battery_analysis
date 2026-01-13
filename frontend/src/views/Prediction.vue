<script setup lang="ts">
import { ref } from 'vue'

defineOptions({
  name: 'PredictionView'
})

const predictForm = ref({
  batteryId: '',
  modelId: '',
  step: 10,
  // target: 'RUL'
})

const batteryOptions = Array.from({ length: 124 }, (_, i) => ({
  value: `B${String(i + 1).padStart(4, '0')}`,
  label: `电池组 B${String(i + 1).padStart(4, '0')}`
}))

const modelOptions = [
  { value: 'm1', label: 'BiLSTM_v1 (2023-10-01)' },
  { value: 'm2', label: 'DeepHPM_best (2023-10-05)' },
  { value: 'm3', label: 'Baseline_RF (2023-09-20)' }
]

const isPredicting = ref(false)
const hasResult = ref(false)

const handlePredict = () => {
  isPredicting.value = true
  setTimeout(() => {
    isPredicting.value = false
    hasResult.value = true
    // TODO: Init charts with result data
  }, 1500)
}
</script>

<template>
  <div class="prediction-container">
    <el-card shadow="hover" class="config-card">
      <el-form :inline="true" :model="predictForm">
        <el-form-item label="待预测电池">
          <el-select v-model="predictForm.batteryId" placeholder="选择电池组" filterable>
            <el-option
              v-for="item in batteryOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="预测模型">
          <el-select v-model="predictForm.modelId" placeholder="选择模型" filterable>
             <el-option
              v-for="item in modelOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="选择步长">
          <el-input-number
            v-model="predictForm.step"
            :min="1"
            :max="500"
            :step="1"
          />
        </el-form-item>
        <!-- <el-form-item label="预测目标">
          <el-radio-group v-model="predictForm.target">
            <el-radio label="RUL">RUL</el-radio>
            <el-radio label="PCL">PCL</el-radio>
            <el-radio label="RUL+PCL">RUL+PCL</el-radio>
          </el-radio-group>
        </el-form-item> -->

        <el-form-item>
          <el-button type="primary" icon="VideoPlay" :loading="isPredicting" @click="handlePredict">
            开始预测
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <div v-if="hasResult" class="result-area mt-20">
      <el-row :gutter="20">
        <el-col :span="12">
          <el-card shadow="hover" header="RUL 预测结果">
            <div style="height: 350px; display: flex; align-items: center; justify-content: center; background: #f5f7fa;">
              RUL Prediction Curve vs Actual
            </div>
          </el-card>
        </el-col>
        <el-col :span="12">
          <el-card shadow="hover" header="PCL 预测结果">
             <div style="height: 350px; display: flex; align-items: center; justify-content: center; background: #f5f7fa;">
              PCL Prediction Curve vs Actual
            </div>
          </el-card>
        </el-col>
      </el-row>

      <el-row class="mt-20">
        <el-col :span="24" style="text-align: right;">
          <el-button type="success" icon="Download">导出预测报告 (CSV)</el-button>
        </el-col>
      </el-row>
    </div>

    <el-empty v-else description="请选择电池和模型进行预测" class="mt-20" />
  </div>
</template>

<style scoped>
.mt-20 {
  margin-top: 20px;
}
</style>
