"""
V3.2配置优化建议
针对真实数据场景的参数调整
"""

# 当前V3.2配置（过于保守）
CURRENT_CONFIG = {
    'limit_dist_threshold': 0.02,  # 距离阈值2% - 过于严格
    'window_size': 10,              # 窗口大小10个tick
    'target_ratio': 5.0             # 正负比例1:5
}

# 建议的V3.2优化配置（适合真实数据）
RECOMMENDED_CONFIG = {
    'limit_dist_threshold': 0.05,  # 距离阈值5% - 更合理
    'window_size': 10,              # 保持10个tick
    'target_ratio': 5.0             # 保持1:5正负比例
}

# 保守配置（确保不丢失正样本）
CONSERVATIVE_CONFIG = {
    'limit_dist_threshold': 0.10,  # 距离阈值10% - 更宽松
    'window_size': 10,              # 保持10个tick
    'target_ratio': 8.0             # 1:8正负比例
}

"""
配置选择指南：

1. 测试验证阶段：使用 CONSERVATIVE_CONFIG
   - 确保不丢失正样本
   - 产生更多困难负样本

2. 正式训练阶段：使用 RECOMMENDED_CONFIG
   - 平衡样本质量和数量
   - 更符合真实交易场景

3. 性能优化阶段：使用 CURRENT_CONFIG（谨慎）
   - 仅在数据量充足时使用
   - 追求最高样本质量

修改方法：
在 scripts/training_set_builder.py 中修改：
    self.window_builder = EventWindowBuilder(
        window_size=10,
        limit_dist_threshold=0.05  # 修改为推荐值
    )

在调用脚本时使用：
    python scripts/training_set_builder.py --input "..." --output "..." --mode single
"""

if __name__ == "__main__":
    print("V3.2配置建议")
    print("="*60)
    print("\n当前配置（过于保守）:")
    for key, value in CURRENT_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n推荐配置（适合真实数据）:")
    for key, value in RECOMMENDED_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n保守配置（确保不丢失正样本）:")
    for key, value in CONSERVATIVE_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("建议：先将 limit_dist_threshold 从 0.02 改为 0.05")
    print("这将显著增加有效样本数量")