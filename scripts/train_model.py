# -*- coding: utf-8 -*-
"""
模型训练脚本

命令行入口，用于训练风险预测模型。
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_core.training.trainer import train_model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练风险预测模型")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(project_root / "data" / "raw"),
        help="数据目录路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "artifacts" / "models"),
        help="模型输出目录"
    )
    parser.add_argument(
        "--use-optuna",
        action="store_true",
        help="是否使用 Optuna 进行超参数优化"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Optuna 试验次数"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 检查数据路径
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"错误: 数据目录不存在: {data_path}")
        print("请确保数据文件位于 data/raw/ 目录下")
        sys.exit(1)
    
    # 训练模型
    try:
        pipeline = train_model(
            data_path=args.data_path,
            output_dir=args.output_dir,
            use_optuna=args.use_optuna,
            n_trials=args.n_trials
        )
        print("\n训练成功完成!")
    except Exception as e:
        print(f"\n训练失败: {e}")
        logging.exception("训练过程中发生错误")
        sys.exit(1)


if __name__ == "__main__":
    main()
