# Database Schema Design (MySQL 8.0)

## Overview

本文档定义了储能电池寿命分析及算法测试平台的完整数据库表结构设计。

**核心设计理念**：
- **数据集抽象**：通过 `dataset` 表统一管理内置数据和用户上传数据
- **训练任务分层**：`training_job`（共享配置）+ `training_job_run`（每个算法独立运行）
- **完整测试平台**：支持批量推理、分电池指标、预测曲线、结果导出
- **多用户隔离**：所有资源表绑定 `user_id`，文件存储按用户分目录

**关键特性**：
- ✅ 一个训练任务支持多算法同时运行（Baseline/BiLSTM/DeepHPM）
- ✅ 每个算法运行产出独立的模型版本
- ✅ 完整的测试任务管理（状态跟踪、指标汇总、结果导出）
- ✅ 用户上传数据支持
- ✅ 数据隔离与权限控制

---

## 1. 用户与认证

### 1.1 user - 用户表

```sql
CREATE TABLE `user` (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_name VARCHAR(50) NOT NULL,
  email VARCHAR(255) NOT NULL,
  password_hash VARCHAR(255) NOT NULL COMMENT 'bcrypt/argon2 hash',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_user_email (email),
  UNIQUE KEY uk_user_name (user_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表，支持JWT认证';
```

**字段说明**：
- `password_hash`：存储加密后的密码（bcrypt/argon2），禁止明文存储
- `email`：唯一，用于登录和找回密码
- `user_name`：唯一，用于展示

---

## 2. 数据管理

### 2.1 data_upload - 用户上传记录

```sql
CREATE TABLE data_upload (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  original_filename VARCHAR(255) NOT NULL COMMENT '用户上传的原始文件名',
  stored_path VARCHAR(500) NOT NULL COMMENT '服务器存储路径',
  file_size BIGINT NOT NULL COMMENT '文件大小（字节）',
  status ENUM('PENDING','PROCESSING','SUCCEEDED','FAILED') NOT NULL DEFAULT 'PENDING',
  error_message VARCHAR(2000) NULL COMMENT '处理失败时的错误信息',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  processed_at DATETIME NULL COMMENT '处理完成时间',
  PRIMARY KEY (id),
  KEY idx_upload_user (user_id),
  KEY idx_upload_status (status),
  CONSTRAINT fk_upload_user FOREIGN KEY (user_id) REFERENCES `user`(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户上传数据记录';
```

**存储路径建议**：`data/{user_id}/uploads/{upload_id}/raw_data.mat`

### 2.2 dataset - 数据集注册表

```sql
CREATE TABLE dataset (
  id BIGINT NOT NULL AUTO_INCREMENT,
  owner_user_id BIGINT NULL COMMENT 'NULL表示内置数据集',
  source_type ENUM('BUILTIN','UPLOAD') NOT NULL,
  name VARCHAR(120) NOT NULL COMMENT '数据集名称',
  upload_id BIGINT NULL COMMENT '关联的上传记录（仅UPLOAD类型）',
  feature_schema JSON NULL COMMENT '特征元信息（特征名称、单位等）',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted_at DATETIME NULL COMMENT '软删除时间戳（NULL表示未删除）',
  PRIMARY KEY (id),
  UNIQUE KEY uk_dataset_upload (upload_id) COMMENT '一个上传对应一个数据集',
  KEY idx_dataset_owner (owner_user_id),
  KEY idx_dataset_deleted (deleted_at) COMMENT '优化软删除过滤查询',
  CONSTRAINT fk_dataset_owner FOREIGN KEY (owner_user_id) REFERENCES `user`(id),
  CONSTRAINT fk_dataset_upload FOREIGN KEY (upload_id) REFERENCES data_upload(id),
  CONSTRAINT chk_builtin_nulls CHECK (
    (source_type = 'BUILTIN' AND owner_user_id IS NULL AND upload_id IS NULL) OR
    (source_type = 'UPLOAD' AND owner_user_id IS NOT NULL AND upload_id IS NOT NULL)
  )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='数据集注册表（内置+用户上传，支持软删除）';
```

**设计说明**：
- 内置数据集：`owner_user_id = NULL`, `source_type = 'BUILTIN'`
- 用户上传：`owner_user_id = {user_id}`, `source_type = 'UPLOAD'`, `upload_id` 指向上传记录

### 2.3 battery_unit - 电池元数据

```sql
CREATE TABLE battery_unit (
  id BIGINT NOT NULL AUTO_INCREMENT,
  dataset_id BIGINT NOT NULL COMMENT '所属数据集',
  battery_code VARCHAR(64) NOT NULL COMMENT '电池编号（原始编号）',
  group_tag ENUM('train','val','test') DEFAULT NULL COMMENT '数据集划分标签',
  total_cycles INT NOT NULL COMMENT '总循环次数',
  nominal_capacity DOUBLE NULL COMMENT '标称容量',
  PRIMARY KEY (id),
  UNIQUE KEY uk_dataset_battery (dataset_id, battery_code) COMMENT '同一数据集内电池编号唯一',
  KEY idx_battery_dataset (dataset_id),
  CONSTRAINT fk_battery_dataset FOREIGN KEY (dataset_id) REFERENCES dataset(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='电池元数据（支持多数据集）';
```

**设计说明**：
- `battery_code`：保留原始电池编号（如 "b1c0", "b1c1"）
- `UNIQUE(dataset_id, battery_code)`：避免跨数据集的编号冲突

### 2.4 cycle_data - 循环数据

```sql
CREATE TABLE cycle_data (
  id BIGINT NOT NULL AUTO_INCREMENT,
  battery_id BIGINT NOT NULL,
  cycle_num INT NOT NULL COMMENT '循环编号',
  feature_1 DOUBLE NOT NULL COMMENT '特征1（如：电压）',
  feature_2 DOUBLE NOT NULL COMMENT '特征2（如：电流）',
  feature_3 DOUBLE NOT NULL COMMENT '特征3（如：温度）',
  feature_4 DOUBLE NOT NULL,
  feature_5 DOUBLE NOT NULL,
  feature_6 DOUBLE NOT NULL,
  feature_7 DOUBLE NOT NULL,
  feature_8 DOUBLE NOT NULL,
  pcl DOUBLE NULL COMMENT 'Percentage Capacity Loss',
  rul INT NULL COMMENT 'Remaining Useful Life',
  PRIMARY KEY (id),
  UNIQUE KEY uk_battery_cycle (battery_id, cycle_num),
  KEY idx_cycle_battery (battery_id),
  CONSTRAINT fk_cycle_battery FOREIGN KEY (battery_id) REFERENCES battery_unit(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='电池循环数据（8项特征+标签）';
```

---

## 3. 训练平台

### 3.1 training_job - 训练任务（共享配置）

```sql
CREATE TABLE training_job (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  dataset_id BIGINT NOT NULL COMMENT '训练数据集',
  target ENUM('RUL','PCL','BOTH') NOT NULL COMMENT '预测目标',
  hyperparams JSON NULL COMMENT '共享超参数（学习率、窗口长度等）',
  status ENUM('PENDING','RUNNING','SUCCEEDED','FAILED','CANCELED') NOT NULL DEFAULT 'PENDING',
  progress DECIMAL(6,5) NOT NULL DEFAULT 0.0 COMMENT '整体进度（0.0-1.0）',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at DATETIME NULL,
  finished_at DATETIME NULL,
  deleted_at DATETIME NULL COMMENT '软删除时间戳（NULL表示未删除）',
  PRIMARY KEY (id),
  KEY idx_training_user (user_id),
  KEY idx_training_status (status),
  KEY idx_training_dataset (dataset_id),
  KEY idx_training_created (created_at),
  KEY idx_training_deleted (deleted_at) COMMENT '优化软删除过滤查询',
  CONSTRAINT fk_training_user FOREIGN KEY (user_id) REFERENCES `user`(id),
  CONSTRAINT fk_training_dataset FOREIGN KEY (dataset_id) REFERENCES dataset(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='训练任务主表（一个任务多算法，支持软删除）';
```

**设计说明**：
- 一个 `training_job` 可以包含多个算法运行（通过 `training_job_run` 关联）
- `status` 反映整体任务状态（所有算法运行完成后为 SUCCEEDED）

### 3.2 training_job_battery - 训练任务电池选择

```sql
CREATE TABLE training_job_battery (
  job_id BIGINT NOT NULL,
  battery_id BIGINT NOT NULL,
  split_role ENUM('train','val','test') NOT NULL DEFAULT 'train',
  PRIMARY KEY (job_id, battery_id) COMMENT '防止同一电池在同一任务中重复',
  KEY idx_job_battery (battery_id),
  CONSTRAINT fk_job_battery_job FOREIGN KEY (job_id) REFERENCES training_job(id),
  CONSTRAINT fk_job_battery_unit FOREIGN KEY (battery_id) REFERENCES battery_unit(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='训练任务电池选择（可选，用于内置数据集）';
```

**使用场景**：
- 用户从内置数据集中选择特定电池进行训练
- 如果不指定，则使用数据集的默认划分（`battery_unit.group_tag`）

### 3.3 training_job_run - 算法运行记录

```sql
CREATE TABLE training_job_run (
  id BIGINT NOT NULL AUTO_INCREMENT,
  job_id BIGINT NOT NULL,
  algorithm ENUM('BASELINE','BILSTM','DEEPHPM') NOT NULL,
  status ENUM('PENDING','RUNNING','SUCCEEDED','FAILED','CANCELED') NOT NULL DEFAULT 'PENDING',
  current_epoch INT NOT NULL DEFAULT 0,
  total_epochs INT NOT NULL DEFAULT 0,
  started_at DATETIME NULL,
  finished_at DATETIME NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_job_algorithm (job_id, algorithm) COMMENT '每个任务每个算法只能运行一次',
  KEY idx_run_job (job_id),
  KEY idx_run_status (status),
  CONSTRAINT fk_run_job FOREIGN KEY (job_id) REFERENCES training_job(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='算法运行记录（一个任务多个算法）';
```

**核心设计**：
- `UNIQUE(job_id, algorithm)`：确保每个算法在同一任务中只运行一次
- 每个 `training_job_run` 产出一个 `model_version`

### 3.4 training_job_run_metric - 训练指标

```sql
CREATE TABLE training_job_run_metric (
  id BIGINT NOT NULL AUTO_INCREMENT,
  run_id BIGINT NOT NULL,
  epoch INT NOT NULL,
  train_loss DOUBLE NULL,
  val_loss DOUBLE NULL,
  metrics JSON NULL COMMENT '其他指标（RMSPE, MSE, R²等）',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_run_epoch (run_id, epoch),
  KEY idx_metric_run (run_id),
  CONSTRAINT fk_metric_run FOREIGN KEY (run_id) REFERENCES training_job_run(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='训练指标（按epoch记录）';
```

**metrics JSON 示例**：
```json
{
  "rmspe": 0.0234,
  "mse": 0.0012,
  "r2": 0.9876,
  "mae": 0.0089
}
```

### 3.5 training_job_run_log - 训练日志

```sql
CREATE TABLE training_job_run_log (
  id BIGINT NOT NULL AUTO_INCREMENT,
  run_id BIGINT NOT NULL,
  user_id BIGINT NOT NULL COMMENT '日志所属用户（数据隔离）',
  level ENUM('DEBUG','INFO','WARNING','ERROR') NOT NULL,
  message VARCHAR(2000) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_log_run (run_id),
  KEY idx_log_user (user_id),
  KEY idx_log_run_time (run_id, created_at) COMMENT '优化按运行查询并排序',
  CONSTRAINT fk_log_run FOREIGN KEY (run_id) REFERENCES training_job_run(id),
  CONSTRAINT fk_log_user FOREIGN KEY (user_id) REFERENCES `user`(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='训练日志（按算法运行记录）';
```

---

## 4. 模型版本管理

### 4.1 model_version - 模型版本

```sql
CREATE TABLE model_version (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  run_id BIGINT NOT NULL COMMENT '关联的训练运行',
  algorithm ENUM('BASELINE','BILSTM','DEEPHPM') NOT NULL,
  name VARCHAR(120) NOT NULL COMMENT '模型名称（用户自定义）',
  version VARCHAR(32) NOT NULL COMMENT '版本号（如：v1.0.0）',
  config JSON NULL COMMENT '模型配置（超参数、架构等）',
  metrics JSON NULL COMMENT '最终评估指标',
  checkpoint_path VARCHAR(500) NOT NULL COMMENT 'checkpoint文件路径',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted_at DATETIME NULL COMMENT '软删除时间戳（NULL表示未删除）',
  PRIMARY KEY (id),
  UNIQUE KEY uk_model_run (run_id) COMMENT '每个训练运行只产出一个模型版本',
  UNIQUE KEY uk_model_version (user_id, name, version) COMMENT '用户内模型名称+版本唯一',
  KEY idx_model_user (user_id),
  KEY idx_model_algorithm (algorithm),
  KEY idx_model_deleted (deleted_at) COMMENT '优化软删除过滤查询',
  CONSTRAINT fk_model_run FOREIGN KEY (run_id) REFERENCES training_job_run(id),
  CONSTRAINT fk_model_user FOREIGN KEY (user_id) REFERENCES `user`(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型版本（每个算法运行产出一个版本，支持软删除）';
```

**存储路径建议**：`data/{user_id}/models/{model_version_id}/checkpoint.pt`

**config JSON 示例**：
```json
{
  "learning_rate": 0.001,
  "window_length": 20,
  "hidden_size": 128,
  "num_layers": 2,
  "dropout": 0.2
}
```

**metrics JSON 示例**：
```json
{
  "val_rmspe": 0.0234,
  "val_mse": 0.0012,
  "val_r2": 0.9876,
  "test_rmspe": 0.0256,
  "test_mse": 0.0015,
  "test_r2": 0.9845
}
```

---

## 5. 测试平台

### 5.1 test_job - 测试任务

```sql
CREATE TABLE test_job (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  model_version_id BIGINT NOT NULL COMMENT '使用的模型版本',
  dataset_id BIGINT NOT NULL COMMENT '测试数据集',
  target ENUM('RUL','PCL','BOTH') NOT NULL COMMENT '预测目标',
  horizon INT NOT NULL DEFAULT 1 COMMENT '预测步长',
  status ENUM('PENDING','RUNNING','SUCCEEDED','FAILED','CANCELED') NOT NULL DEFAULT 'PENDING',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at DATETIME NULL,
  finished_at DATETIME NULL,
  deleted_at DATETIME NULL COMMENT '软删除时间戳（NULL表示未删除）',
  PRIMARY KEY (id),
  KEY idx_test_user (user_id),
  KEY idx_test_status (status),
  KEY idx_test_model (model_version_id),
  KEY idx_test_dataset (dataset_id),
  KEY idx_test_created (created_at),
  KEY idx_test_deleted (deleted_at) COMMENT '优化软删除过滤查询',
  CONSTRAINT fk_test_user FOREIGN KEY (user_id) REFERENCES `user`(id),
  CONSTRAINT fk_test_model FOREIGN KEY (model_version_id) REFERENCES model_version(id),
  CONSTRAINT fk_test_dataset FOREIGN KEY (dataset_id) REFERENCES dataset(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='测试任务（基于模型版本的批量推理评估，支持软删除）';
```

**设计说明**：
- 一个测试任务使用一个模型版本对指定数据集进行批量推理
- `horizon`：预测步长（用于多步预测场景）

### 5.2 test_job_battery - 测试任务电池选择

```sql
CREATE TABLE test_job_battery (
  test_job_id BIGINT NOT NULL,
  battery_id BIGINT NOT NULL,
  PRIMARY KEY (test_job_id, battery_id),
  KEY idx_test_battery (battery_id),
  CONSTRAINT fk_tjb_test FOREIGN KEY (test_job_id) REFERENCES test_job(id),
  CONSTRAINT fk_tjb_battery FOREIGN KEY (battery_id) REFERENCES battery_unit(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='测试任务电池选择（可选，用于选择特定电池）';
```

**使用场景**：
- 用户从数据集中选择特定电池进行测试
- 如果不指定，则对数据集中所有电池进行测试

### 5.3 test_job_metric_overall - 测试总体指标

```sql
CREATE TABLE test_job_metric_overall (
  id BIGINT NOT NULL AUTO_INCREMENT,
  test_job_id BIGINT NOT NULL,
  target ENUM('RUL','PCL') NOT NULL,
  metrics JSON NOT NULL COMMENT '总体指标（RMSPE, MSE, R², MAE等）',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_test_target (test_job_id, target),
  KEY idx_metric_test (test_job_id),
  CONSTRAINT fk_test_metric_job FOREIGN KEY (test_job_id) REFERENCES test_job(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='测试总体指标（按预测目标）';
```

**metrics JSON 示例**：
```json
{
  "rmspe": 0.0256,
  "mse": 0.0015,
  "r2": 0.9845,
  "mae": 0.0098,
  "rmse": 0.0387
}
```

### 5.4 test_job_battery_metric - 测试分电池指标

```sql
CREATE TABLE test_job_battery_metric (
  id BIGINT NOT NULL AUTO_INCREMENT,
  test_job_id BIGINT NOT NULL,
  battery_id BIGINT NOT NULL,
  target ENUM('RUL','PCL') NOT NULL,
  metrics JSON NOT NULL COMMENT '该电池的指标',
  PRIMARY KEY (id),
  UNIQUE KEY uk_test_battery_target (test_job_id, battery_id, target),
  KEY idx_test_battery_metric (test_job_id),
  KEY idx_battery_metric (battery_id),
  CONSTRAINT fk_tbm_test FOREIGN KEY (test_job_id) REFERENCES test_job(id),
  CONSTRAINT fk_tbm_battery FOREIGN KEY (battery_id) REFERENCES battery_unit(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='测试分电池指标（用于识别困难样本）';
```

**使用场景**：
- 快速定位预测效果差的电池
- 支持分电池误差对比可视化（条形图/箱线图）

### 5.5 test_job_prediction - 测试预测结果

```sql
CREATE TABLE test_job_prediction (
  id BIGINT NOT NULL AUTO_INCREMENT,
  test_job_id BIGINT NOT NULL,
  battery_id BIGINT NOT NULL,
  cycle_num INT NOT NULL COMMENT '循环编号',
  target ENUM('RUL','PCL') NOT NULL,
  y_true DOUBLE NULL COMMENT '真实值（离线评估时必填）',
  y_pred DOUBLE NOT NULL COMMENT '预测值',
  PRIMARY KEY (id),
  UNIQUE KEY uk_pred_sample (test_job_id, battery_id, target, cycle_num) COMMENT '每个样本唯一',
  KEY idx_pred_test (test_job_id),
  KEY idx_pred_battery (battery_id),
  CONSTRAINT fk_pred_test FOREIGN KEY (test_job_id) REFERENCES test_job(id),
  CONSTRAINT fk_pred_battery FOREIGN KEY (battery_id) REFERENCES battery_unit(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='测试预测结果（逐样本，支持曲线绘制和导出）';
```

**设计说明**：
- 存储每个样本的预测值和真实值
- 支持前端绘制"预测 vs 真实"曲线
- 支持导出到 CSV/Excel

### 5.6 test_job_log - 测试日志

```sql
CREATE TABLE test_job_log (
  id BIGINT NOT NULL AUTO_INCREMENT,
  test_job_id BIGINT NOT NULL,
  user_id BIGINT NOT NULL COMMENT '日志所属用户（数据隔离）',
  level ENUM('DEBUG','INFO','WARNING','ERROR') NOT NULL,
  message VARCHAR(2000) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_test_log (test_job_id),
  KEY idx_log_user (user_id),
  KEY idx_test_log_time (test_job_id, created_at) COMMENT '优化按测试任务查询并排序',
  CONSTRAINT fk_test_log_job FOREIGN KEY (test_job_id) REFERENCES test_job(id),
  CONSTRAINT fk_test_log_user FOREIGN KEY (user_id) REFERENCES `user`(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='测试日志';
```

### 5.7 test_export - 测试结果导出

```sql
CREATE TABLE test_export (
  id BIGINT NOT NULL AUTO_INCREMENT,
  test_job_id BIGINT NOT NULL,
  user_id BIGINT NOT NULL,
  format ENUM('CSV','XLSX') NOT NULL,
  file_path VARCHAR(500) NOT NULL COMMENT '导出文件路径',
  file_size BIGINT NULL COMMENT '文件大小（字节）',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_export_test (test_job_id),
  KEY idx_export_user (user_id),
  KEY idx_export_created (created_at),
  CONSTRAINT fk_export_test FOREIGN KEY (test_job_id) REFERENCES test_job(id),
  CONSTRAINT fk_export_user FOREIGN KEY (user_id) REFERENCES `user`(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='测试结果导出记录';
```

**存储路径建议**：`data/{user_id}/exports/{test_job_id}/result_{timestamp}.xlsx`

**导出内容建议**：
- 元信息：模型版本、算法类型、测试配置、数据来源、电池编号列表
- 总体指标：RMSPE、MSE、R²、MAE、RMSE
- 分电池指标：每个电池的指标汇总
- 预测明细：每条样本的 cycle_num、y_true、y_pred

---

## 6. 设计总结

### 6.1 表关系概览

**核心关系链**：
```
user (用户)
  ├─> data_upload (上传记录)
  │     └─> dataset (数据集)
  │           ├─> battery_unit (电池)
  │           │     └─> cycle_data (循环数据)
  │           ├─> training_job (训练任务)
  │           └─> test_job (测试任务)
  │
  ├─> training_job (训练任务)
  │     ├─> training_job_battery (电池选择)
  │     └─> training_job_run (算法运行)
  │           ├─> training_job_run_metric (指标)
  │           ├─> training_job_run_log (日志)
  │           └─> model_version (模型版本)
  │
  └─> test_job (测试任务)
        ├─> test_job_battery (电池选择)
        ├─> test_job_metric_overall (总体指标)
        ├─> test_job_battery_metric (分电池指标)
        ├─> test_job_prediction (预测结果)
        ├─> test_job_log (日志)
        └─> test_export (导出记录)
```

### 6.2 关键设计决策

**1. 数据集抽象层（dataset）**
- **优势**：统一管理内置数据和用户上传数据，支持未来扩展多数据源
- **实现**：`battery_unit` 通过 `dataset_id` 关联，使用 `battery_code` 保留原始编号
- **隔离**：内置数据集 `owner_user_id = NULL`，用户上传数据集绑定 `user_id`

**2. 训练任务分层（training_job + training_job_run）**
- **优势**：支持"一个任务多算法"，每个算法独立运行和状态管理
- **实现**：`training_job` 存储共享配置，`training_job_run` 存储每个算法的运行状态
- **约束**：`UNIQUE(job_id, algorithm)` 确保每个算法在同一任务中只运行一次

**3. 模型版本管理（model_version）**
- **优势**：每个算法运行产出独立的模型版本，支持版本追溯和复用
- **实现**：`model_version` 关联 `training_job_run`，存储 checkpoint、配置、指标
- **命名**：`UNIQUE(user_id, name, version)` 确保用户内模型名称+版本唯一

**4. 完整测试平台（test_job + 多级指标）**
- **优势**：支持批量推理、总体指标、分电池指标、预测曲线、结果导出
- **实现**：
  - `test_job_metric_overall`：总体指标（按 RUL/PCL 分别存储）
  - `test_job_battery_metric`：分电池指标（识别困难样本）
  - `test_job_prediction`：逐样本预测结果（支持曲线绘制）
  - `test_export`：导出记录（CSV/XLSX）

**5. 多用户数据隔离**
- **数据库层**：所有用户资源表绑定 `user_id`
- **文件系统层**：按用户分目录存储
  - 上传：`data/{user_id}/uploads/{upload_id}/`
  - 模型：`data/{user_id}/models/{model_version_id}/`
  - 导出：`data/{user_id}/exports/{test_job_id}/`
