"""
Baseline 算法训练脚本（数据库版本）

此脚本是原始 Baseline.py 的重构版本，现在从数据库加载数据
支持通过命令行参数或配置文件指定训练参数
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch

from src.models import BatteryUnit, SessionLocal
from src.tasks.model.baseline_trainer import (
    BaselineTrainingConfig,
    TrainingCallbacks,
    train_baseline_from_database,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Baseline 算法训练（数据库版本）")

    # 数据选择
    parser.add_argument(
        "--dataset-id",
        type=int,
        help="数据集ID（如果不指定则使用电池ID）",
    )
    parser.add_argument(
        "--battery-ids-train",
        type=int,
        nargs="+",
        help="训练电池ID列表（例如: --battery-ids-train 1 2 3）",
    )
    parser.add_argument(
        "--battery-ids-test",
        type=int,
        nargs="+",
        help="测试电池ID列表（例如: --battery-ids-test 4 5）",
    )

    # 训练参数
    parser.add_argument("--seq-len", type=int, default=1, help="序列长度")
    parser.add_argument("--perc-val", type=float, default=0.2, help="验证集比例")
    parser.add_argument(
        "--num-layers", type=int, nargs="+", default=[2, 3], help="网络层数列表"
    )
    parser.add_argument(
        "--num-neurons",
        type=int,
        nargs="+",
        default=[100, 150],
        help="每层神经元数列表",
    )
    parser.add_argument("--num-epoch", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--step-size", type=int, default=100, help="学习率衰减步长")
    parser.add_argument("--gamma", type=float, default=0.5, help="学习率衰减系数")
    parser.add_argument("--num-rounds", type=int, default=1, help="实验重复次数")
    parser.add_argument("--random-seed", type=int, default=1234, help="随机种子")

    # 输出
    parser.add_argument(
        "--output-path",
        type=str,
        default="./results/SoH_CaseA_Baseline_DB.pth",
        help="结果保存路径",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建数据库会话
    db = SessionLocal()

    try:
        # 1. 确定训练和测试电池
        if args.dataset_id:
            # 从数据集加载所有电池
            print(f"从数据集 ID={args.dataset_id} 加载电池...")
            batteries = (
                db.query(BatteryUnit)
                .filter(BatteryUnit.dataset_id == args.dataset_id)
                .all()
            )

            if not batteries:
                raise ValueError(f"数据集 ID={args.dataset_id} 中没有电池数据")

            print(f"找到 {len(batteries)} 个电池单元")

            # 使用默认分割（前80%训练，后20%测试）
            split_idx = int(len(batteries) * 0.8)
            battery_ids_train: list[int] = [int(b.id) for b in batteries[:split_idx]]  # type: ignore[arg-type]
            battery_ids_test: list[int] = [int(b.id) for b in batteries[split_idx:]]  # type: ignore[arg-type]

        elif args.battery_ids_train and args.battery_ids_test:
            # 使用指定的电池ID
            battery_ids_train = args.battery_ids_train
            battery_ids_test = args.battery_ids_test

        else:
            raise ValueError(
                "必须指定 --dataset-id 或同时指定 --battery-ids-train 和 --battery-ids-test"
            )

        print(f"训练电池 IDs: {battery_ids_train}")
        print(f"测试电池 IDs: {battery_ids_test}")

        # 2. 创建训练配置
        config = BaselineTrainingConfig(
            seq_len=args.seq_len,
            perc_val=args.perc_val,
            num_layers=args.num_layers,
            num_neurons=args.num_neurons,
            num_epoch=args.num_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            step_size=args.step_size,
            gamma=args.gamma,
            num_rounds=args.num_rounds,
            random_seed=args.random_seed,
        )

        print("\n训练配置:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")

        # 3. 创建回调函数
        epoch_logs = []

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
            if epoch % 50 == 0 or epoch == config.num_epoch - 1:
                print(
                    f"  Epoch {epoch + 1}/{config.num_epoch}: "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                )
            epoch_logs.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "round_idx": round_idx,
                    "num_rounds": num_rounds,
                    **metrics,
                }
            )

        def on_hyperparameter_search(
            l_idx: int, n_idx: int, num_l: int, num_n: int, metrics: dict
        ):
            print(
                f"\n超参数搜索 [{l_idx + 1}/{len(config.num_layers)}, {n_idx + 1}/{len(config.num_neurons)}]:"
            )
            print(f"  层数={num_l}, 神经元={num_n}")
            print(f"  训练集 RMSPE: {metrics['train_rmspe']:.6f}")
            print(f"  验证集 RMSPE: {metrics['val_rmspe']:.6f}")
            print(f"  测试集 RMSPE: {metrics['test_rmspe']:.6f}")

        callbacks = TrainingCallbacks(
            on_log=on_log,
            on_epoch_end=on_epoch_end,  # y
            on_hyperparameter_search=on_hyperparameter_search,
        )

        # 4. 开始训练
        print("\n" + "=" * 80)
        print("开始训练")
        print("=" * 80 + "\n")

        results = train_baseline_from_database(
            db=db,
            battery_ids_train=battery_ids_train,
            battery_ids_test=battery_ids_test,
            config=config,
            callbacks=callbacks,
        )

        # 5. 保存结果
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 添加额外信息到结果
        results["battery_ids_train"] = battery_ids_train
        results["battery_ids_test"] = battery_ids_test
        results["epoch_logs"] = epoch_logs

        torch.save(results, str(output_path))
        print(f"\n结果已保存到: {output_path}")

        # 6. 打印最终结果
        print("\n" + "=" * 80)
        print("训练完成")
        print("=" * 80)

        if "best_results" in results:
            print("\n最佳模型评估结果:")
            for split, metrics in results["best_results"].items():
                print(f"\n{split.upper()} 集:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.6f}")

        print("\n所有超参数配置的测试集 RMSPE:")
        print(results["metric_mean"]["test"])

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    exit(main())
