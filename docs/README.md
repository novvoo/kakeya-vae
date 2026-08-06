# Kakeya Image Codec 文档

本目录包含 Kakeya 图像编解码器的架构文档。

## 文档列表

- [architecture.md](./architecture.md) — 完整架构图集，按项目逻辑故事线排序：
  1. **挂谷猜想与潜空间正则化**（几何直觉、算法实现、训练集成、完整链路）
  2. 架构版本 9 主干（Base-first Haar-SCH Detail、Hyperprior v11、通道维分组上下文）
  3. 损失函数（含小图尺度条件与 Laplacian 高频约束）
  4. 多尺度数据流
  5. Capacity → Transition → Finetune 训练、优化器隔离与高清 checkpoint 保护
  6. 推理与真实字节流
  7. Web API 与任务管理
  8. 泛化能力、高清大图处理与历史问题诊断

## 快速导航

| 关注点 | 查看章节 |
|--------|---------|
| **项目核心创新** | **§1 挂谷猜想与潜空间正则化** |
| **挂谷正则算法** | **§1.2 算法实现 + §1.3 训练集成** |
| 模型长什么样 | §2 架构版本 8 主干 |
| BlendedInstanceNorm 与完整 Base 分支 | §2 关键组件细节 |
| 损失函数与小图策略 | §3 |
| 多尺度训练数据 | §4 |
| 训练、优化器与高清保护 | §5 |
| 推理/压缩/解压 | §6 |
| API 接口 | §8 |
| 泛化与大图历史诊断 | §9 |

图表使用 [Mermaid](https://mermaid.js.org/) 语法，可在 GitHub、VS Code（Mermaid 插件）或任何支持 Mermaid 的 Markdown 渲染器中查看。
