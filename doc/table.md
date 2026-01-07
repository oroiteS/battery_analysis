# Database Tables (MySQL)

```sql
-- User
CREATE TABLE user (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_name varchar(50) NOT NULL,
  email VARCHAR(255) NOT NULL,
  password VARCHAR(255) NOT NULL,
  path VARCHAR(500) NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Battery metadata
CREATE TABLE battery_unit (
  id INT NOT NULL,
  group_tag ENUM('train','val','test') DEFAULT NULL,
  total_cycles INT NOT NULL,
  nominal_capacity DOUBLE NULL,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Cycle data (features + labels)
CREATE TABLE cycle_data (
  id BIGINT NOT NULL AUTO_INCREMENT,
  battery_id INT NOT NULL,
  cycle_num INT NOT NULL,
  feature_1 DOUBLE NOT NULL,
  feature_2 DOUBLE NOT NULL,
  feature_3 DOUBLE NOT NULL,
  feature_4 DOUBLE NOT NULL,
  feature_5 DOUBLE NOT NULL,
  feature_6 DOUBLE NOT NULL,
  feature_7 DOUBLE NOT NULL,
  feature_8 DOUBLE NOT NULL,
  pcl DOUBLE NULL,
  rul INT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_battery_cycle (battery_id, cycle_num),
  KEY idx_cycle_battery (battery_id),
  CONSTRAINT fk_cycle_battery FOREIGN KEY (battery_id) REFERENCES battery_unit(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Training jobs
CREATE TABLE training_job (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  algorithm ENUM('baseline','bilstm','deephpm') NOT NULL,
  status ENUM('queued','running','succeeded','failed','stopped') NOT NULL,
  model_name VARCHAR(120) NULL,
  hyperparams JSON NULL,
  progress DECIMAL(6,5) NOT NULL DEFAULT 0.0,
  current_epoch INT NOT NULL DEFAULT 0,
  total_epochs INT NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at DATETIME NULL,
  finished_at DATETIME NULL,
  PRIMARY KEY (id),
  KEY idx_job_status (status),
  KEY idx_job_user (user_id),
  CONSTRAINT fk_job_user FOREIGN KEY (user_id) REFERENCES user(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Training job battery selection
CREATE TABLE training_job_battery (
  job_id BIGINT NOT NULL,
  battery_id INT NOT NULL,
  split_role ENUM('train','val','test') NOT NULL DEFAULT 'train',
  PRIMARY KEY (job_id, battery_id, split_role),
  KEY idx_job_battery (battery_id),
  CONSTRAINT fk_job_battery_job FOREIGN KEY (job_id) REFERENCES training_job(id),
  CONSTRAINT fk_job_battery_unit FOREIGN KEY (battery_id) REFERENCES battery_unit(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Training metrics per epoch
CREATE TABLE training_job_metric (
  id BIGINT NOT NULL AUTO_INCREMENT,
  job_id BIGINT NOT NULL,
  epoch INT NOT NULL,
  train_loss DOUBLE NULL,
  val_loss DOUBLE NULL,
  metrics JSON NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_job_epoch (job_id, epoch),
  KEY idx_metric_job (job_id),
  CONSTRAINT fk_metric_job FOREIGN KEY (job_id) REFERENCES training_job(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Training logs
CREATE TABLE training_job_log (
  id BIGINT NOT NULL AUTO_INCREMENT,
  job_id BIGINT NOT NULL,
  level ENUM('DEBUG','INFO','WARNING','ERROR') NOT NULL,
  message VARCHAR(2000) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_log_job (job_id),
  CONSTRAINT fk_log_job FOREIGN KEY (job_id) REFERENCES training_job(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Models
CREATE TABLE model (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  algorithm ENUM('baseline','bilstm','deephpm') NOT NULL,
  name VARCHAR(120) NOT NULL,
  version VARCHAR(32) NOT NULL,
  job_id BIGINT NULL,
  metrics JSON NULL,
  file_path VARCHAR(255) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_model_version (algorithm, name, version),
  KEY idx_model_job (job_id),
  KEY idx_model_user (user_id),
  CONSTRAINT fk_model_job FOREIGN KEY (job_id) REFERENCES training_job(id),
  CONSTRAINT fk_model_user FOREIGN KEY (user_id) REFERENCES user(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Predictions (single model)
CREATE TABLE prediction (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  model_id BIGINT NOT NULL,
  battery_id INT NULL,
  input_source ENUM('battery','payload') NOT NULL,
  input_payload JSON NULL,
  soh DOUBLE NULL,
  rul DOUBLE NULL,
  result JSON NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_pred_model (model_id),
  KEY idx_pred_battery (battery_id),
  KEY idx_pred_user (user_id),
  CONSTRAINT fk_pred_model FOREIGN KEY (model_id) REFERENCES model(id),
  CONSTRAINT fk_pred_battery FOREIGN KEY (battery_id) REFERENCES battery_unit(id),
  CONSTRAINT fk_pred_user FOREIGN KEY (user_id) REFERENCES user(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Ensemble configuration
CREATE TABLE model_ensemble (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  name VARCHAR(120) NOT NULL,
  strategy ENUM('compare','avg','weighted_avg','stacking') NOT NULL,
  config JSON NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_ensemble_user (user_id),
  CONSTRAINT fk_ensemble_user FOREIGN KEY (user_id) REFERENCES user(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Ensemble members
CREATE TABLE model_ensemble_member (
  ensemble_id BIGINT NOT NULL,
  model_id BIGINT NOT NULL,
  weight DECIMAL(8,6) NULL,
  PRIMARY KEY (ensemble_id, model_id),
  CONSTRAINT fk_ensemble_member_ensemble FOREIGN KEY (ensemble_id) REFERENCES model_ensemble(id),
  CONSTRAINT fk_ensemble_member_model FOREIGN KEY (model_id) REFERENCES model(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Ensemble prediction (aggregate result)
CREATE TABLE ensemble_prediction (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  ensemble_id BIGINT NOT NULL,
  battery_id INT NULL,
  input_payload JSON NULL,
  soh DOUBLE NULL,
  rul DOUBLE NULL,
  result JSON NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_ensemble_pred (ensemble_id),
  KEY idx_ensemble_pred_user (user_id),
  CONSTRAINT fk_ensemble_pred_ensemble FOREIGN KEY (ensemble_id) REFERENCES model_ensemble(id),
  CONSTRAINT fk_ensemble_pred_battery FOREIGN KEY (battery_id) REFERENCES battery_unit(id),
  CONSTRAINT fk_ensemble_pred_user FOREIGN KEY (user_id) REFERENCES user(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Ensemble prediction details (per model)
CREATE TABLE ensemble_prediction_detail (
  id BIGINT NOT NULL AUTO_INCREMENT,
  ensemble_prediction_id BIGINT NOT NULL,
  model_id BIGINT NOT NULL,
  soh DOUBLE NULL,
  rul DOUBLE NULL,
  result JSON NULL,
  PRIMARY KEY (id),
  KEY idx_ensemble_pred_detail (ensemble_prediction_id),
  CONSTRAINT fk_ensemble_detail_pred FOREIGN KEY (ensemble_prediction_id) REFERENCES ensemble_prediction(id),
  CONSTRAINT fk_ensemble_detail_model FOREIGN KEY (model_id) REFERENCES model(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```
