# Kakeya Image Codec 文档

本目录包含 Kakeya 图像编解码器的架构文档。

## 文档列表

- [architecture.md](./architecture.md) — 完整架构图集，按项目逻辑故事线排序：
  1. **挂谷猜想与潜空间正则化** (核心创新：几何直觉、算法实现、训练集成、完整链路、多项式扩展)
  2. 模型结构 (ImageCodecVAE: Encoder → Latent → EntropyBottleneck → Decoder)
  3. 损失函数组成 (reconstruction + mse + edge + structural + multiscale + kakeya + lab + rate)
  4. 数据流 (DataLoader 与 Dataset 关系)
  5. 训练流程 (Capacity → Transition → Finetune 三阶段，含闸门机制)
  6. 训练阶段时序图 (epoch 级别的训练循环细节)
  7. 推理流程 (Web API 与 CLI 的编解码路径)
  8. Web API 端点结构 (FastAPI 路由与 JobManager/子进程关系)
  9. **泛化能力与高清大图处理** (过拟合诊断、修复策略、推理流程、问题链路、码率一致性)

## 快速导航

| 关注点 | 查看章节 |
|--------|---------|
| **项目核心创新** | **§1 挂谷猜想与潜空间正则化** |
| **挂谷猜想原理** | **§1.1 几何直觉 + §1.5 完整链路** |
| **挂谷正则算法** | **§1.2 算法实现** |
| **挂谷在训练中** | **§1.3 训练集成** |
| 模型长什么样 | §2 模型结构 |
| 损失函数 | §3 损失函数组成 |
| 多尺度训练数据 | §4 数据流 |
| 训练怎么跑 | §5 训练流程, §6 时序图 |
| 闸门机制 | §5 中的 Capacity Stage + Gate Check |
| 推理/压缩/解压 | §7 推理流程 |
| API 接口 | §8 Web API 端点结构 |
| **泛化/大图颜色丢失** | **§9 泛化能力与高清大图处理** |
| **过拟合根因** | **§9.1 诊断 + §9.5 问题链路** |
| **大图推理流程** | **§9.3 推理流程** |

图表使用 [Mermaid](https://mermaid.js.org/) 语法，可在 GitHub、VS Code（Mermaid 插件）或任何支持 Mermaid 的 Markdown 渲染器中查看。
