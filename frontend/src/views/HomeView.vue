<script setup lang="ts">
import { ref } from 'vue'

// Placeholder for Dashboard stats
const stats = ref([
  { title: '电池总数', value: 124, icon: 'BatteryFull', color: '#409EFF' },
  { title: '已训练模型', value: 12, icon: 'Cpu', color: '#67C23A' },
  { title: '平均 RUL', value: '850 Cycles', icon: 'Timer', color: '#E6A23C' },
  { title: '告警数量', value: 3, icon: 'Warning', color: '#F56C6C' },
])
</script>

<template>
  <div class="dashboard-container">
    <el-row :gutter="20">
      <el-col :span="6" v-for="item in stats" :key="item.title">
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

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="16">
        <el-card shadow="hover" header="系统概况">
          <div style="height: 300px; display: flex; align-items: center; justify-content: center; color: #909399;">
            欢迎使用储能电池寿命分析及算法测试平台
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover" header="快速开始">
          <el-steps direction="vertical" :active="1">
            <el-step title="数据导入" description="已完成 124 组电池数据入库" />
            <el-step title="算法训练" description="选择模型进行训练" />
            <el-step title="寿命预测" description="对电池进行 RUL/PCL 预测" />
          </el-steps>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.stat-card {
  height: 100px;
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
</style>
