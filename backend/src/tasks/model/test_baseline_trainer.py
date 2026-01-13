"""
测试 Baseline 训练器的功能

使用内置数据集测试从数据库加载数据并训练
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models import BatteryUnit, SessionLocal
from src.tasks.model.baseline_trainer import (
    BaselineTrainingConfig,
    TrainingCallbacks,
    train_baseline_from_database,
)


def test_baseline_trainer():
    """测试 Baseline 训练器"""
    print("=" * 80)
    print("测试 Baseline 训练器")
    print("=" * 80)

    # 创建数据库会话
    db = SessionLocal()

    try:
        # 查询可用的电池（假设已经有内置数据集）
        batteries = db.query(BatteryUnit).limit(10).all()

        if not batteries:
            print("错误: 数据库中没有电池数据，请先导入内置数据集")
            return

        print(f"\n找到 {len(batteries)} 个电池单元:")
        for battery in batteries:
            print(
                f"  - ID={battery.id}, Code={battery.battery_code}, Cycles={battery.total_cycles}"
            )

        # 选择电池进行训练（前8个训练，后2个测试）
        if len(batteries) < 3:
            print("错误: 至少需要3个电池单元进行测试")
            return

        battery_ids_train = [b.id for b in batteries[:2]]
        battery_ids_test = [b.id for b in batteries[2:3]]

        print(f"\n训练电池 IDs: {battery_ids_train}")
        print(f"测试电池 IDs: {battery_ids_test}")

        # 创建训练配置（简化参数以加快测试）
        config = BaselineTrainingConfig(
            seq_len=1,
            perc_val=0.2,
            num_layers=[2],  # 只测试一组超参数
            num_neurons=[50],  # 更小的网络
            num_epoch=10,  # 只训练10个epoch
            batch_size=32,
            lr=0.001,
            num_rounds=1,
        )

        print("\n训练配置:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")

        # 创建回调函数
        def on_log(level: str, message: str):
            print(f"[{level}] {message}")

        def on_epoch_end(
            epoch: int,
            train_loss: float,
            val_loss: float,
            metrics: dict[str, float],
            round_idx: int,
            num_rounds: int,
        ) -> None:
            print(
                f"  Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

        def on_hyperparameter_search(
            l_idx: int, n_idx: int, num_l: int, num_n: int, metrics: dict
        ):
            print(f"\n超参数搜索结果 (层数={num_l}, 神经元={num_n}):")
            print(f"  训练集 RMSPE: {metrics['train_rmspe']:.6f}")
            print(f"  验证集 RMSPE: {metrics['val_rmspe']:.6f}")
            print(f"  测试集 RMSPE: {metrics['test_rmspe']:.6f}")

        callbacks = TrainingCallbacks(
            on_log=on_log,
            on_epoch_end=on_epoch_end,
            on_hyperparameter_search=on_hyperparameter_search,
        )

        # 开始训练
        print("\n" + "=" * 80)
        print("开始训练")
        print("=" * 80 + "\n")

        results = train_baseline_from_database(
            db=db,
            battery_ids_train=battery_ids_train,  # type: ignore
            battery_ids_test=battery_ids_test,  # type: ignore
            config=config,
            callbacks=callbacks,
        )

        # 打印最终结果
        print("\n" + "=" * 80)
        print("训练完成")
        print("=" * 80)

        if "best_results" in results:
            print("\n最佳模型评估结果:")
            for split, metrics in results["best_results"].items():
                print(f"\n{split.upper()} 集:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.6f}")

        print("\n测试成功! ✓")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    test_baseline_trainer()
