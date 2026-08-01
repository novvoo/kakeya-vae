# Kakeya Image Codec 架构图

> 文档顺序遵循项目逻辑故事线：**为什么做（挂谷猜想）→ 怎么做（模型/损失/数据/训练）→ 怎么用（推理/API）→ 遇到什么问题（泛化/大图）**

---

## 1. 挂谷猜想 (Kakeya Conjecture) 与潜空间正则化

> 本项目的核心创新。受挂谷猜想启发，通过随机投影方向上的最大间距最大化潜变量的方向覆盖，避免 VAE 后验坍缩。

### 1.1 挂谷猜想的几何直觉

```mermaid
graph TD
    subgraph Kakeya猜想
        K1[Kakeya 集定义<br/>欧氏空间中包含<br/>任意方向单位线段的点集]
        K2[Besicovitch 构造<br/>在 R^d 中存在<br/>Lebesgue 测度任意小<br/>却覆盖所有方向的集合]
        K3[核心几何特征<br/>以最小体积<br/>覆盖所有方向]

        K1 --> K2
        K2 --> K3
    end

    subgraph 迁移到VAE潜空间
        V1[潜变量 z ∈ R^d<br/>d 维潜在空间]
        V2[问题: 后验坍缩<br/>部分维度失效<br/>信息集中在少数维度]
        V3[挂谷正则思路<br/>通过随机投影方向上的<br/>最大间距最大化<br/>潜变量的方向覆盖]
        V4[目标: 所有维度<br/>被充分且均匀利用<br/>避免维度退化]

        V1 --> V2
        V2 --> V3
        K3 -.->|几何直觉迁移| V3
        V3 --> V4
    end

    subgraph 与其他方法对比
        C1[β-VAE / FactorVAE<br/>关注: 每个维度是否有用<br/>解耦表示]
        C2[挂谷正则<br/>关注: 所有方向是否被覆盖<br/>方向覆盖完备性]
        C1 -.->|互补| C2
    end

    style K3 fill:#fff9c4
    style V4 fill:#c8e6c9
    style C2 fill:#e1f5fe
```

### 1.2 挂谷正则化算法 (kakeya_regularization)

```mermaid
flowchart TD
    subgraph 输入
        LATENT[潜变量 latent<br/>B x C x H x W<br/>C=config.latent_dim, 默认 8]
        RESHAP[重塑为点集<br/>permute → B*H*W x C<br/>每个像素位置一个 C 维向量]
        NORM[L2 归一化<br/>F.normalize dim=1<br/>投影到单位球面]
        LATENT --> RESHAP
        RESHAP --> NORM
    end

    subgraph 随机投影方向
        DIR[生成 num_projections 个<br/>随机方向向量<br/>32 x C<br/>config: num_projections=32]
        NORM_DIR[归一化到单位球面<br/>F.normalize dim=1]
        DIR --> NORM_DIR
    end

    subgraph 投影与排序
        PROJ[z @ directions.T<br/>N x num_projections<br/>每个点在各方向的投影值]
        SORT[按方向排序<br/>torch.sort dim=0]
        PROJ --> SORT
    end

    NORM --> PROJ
    NORM_DIR --> PROJ

    subgraph 间距计算
        DIFF[相邻投影差<br/>sorted.diff dim=0<br/>N-1 x num_projections]
        TOPK[取 top-k 最大间距<br/>k=min config.k=3, N-1<br/>torch.topk dim=0]
        MEAN[取均值<br/>负号: 最大化间距]
        SORT --> DIFF
        DIFF --> TOPK
        TOPK --> MEAN
    end

    MEAN --> OUT[coverage loss<br/>标量<br/>负的 top-k 间距均值]

    subgraph 直觉解释
        INT1[间距大 = 该方向上<br/>潜变量分布稀疏<br/>存在未覆盖区域]
        INT2[最大化间距最小化<br/>= 填补稀疏方向<br/>均匀覆盖所有方向]
        INT1 --> INT2
    end

    OUT -.-> INT2

    style LATENT fill:#e1f5fe
    style NORM fill:#f3e5f5
    style PROJ fill:#fff3e0
    style OUT fill:#c8e6c9
    style INT2 fill:#fff9c4
```

### 1.3 挂谷正则在训练中的集成

```mermaid
graph TD
    subgraph 训练步骤
        IMG[输入图像 batch<br/>B x 3 x H x W]
        ENC[g_a 分析变换<br/>→ y]
        QUANT[Hyperprior 熵量化<br/>→ y_hat, z_hat, rate_bpp]
        DEC[g_s 综合变换<br/>→ reconstructed]

        IMG --> ENC
        ENC --> QUANT
        QUANT --> DEC
    end

    subgraph 挂谷正则计算路径
        L1[y_hat: B x C x H/4 x W/4<br/>C 默认 8]
        L2[permute → B x H/4 x W/4 x C]
        L3[reshape → B*H/4*W/4 x C<br/>每个空间位置一个 C 维点]
        L4[F.normalize dim=1<br/>归一化到单位球面]
        L5[kakeya_regularization<br/>32 方向投影 + top-k 间距]
        L6[coverage = -mean top-k gaps<br/>负号: 最大化覆盖]

        L1 --> L2
        L2 --> L3
        L3 --> L4
        L4 --> L5
        L5 --> L6
    end

    QUANT -.->|latent| L1

    subgraph 损失加权
        KW[kakeya_weight = config.lambda_kakeya<br/>默认 0.001]
        KC[kakeya_contribution<br/>= lambda_kakeya * coverage]
        TOTAL[total loss<br/>= MSE + lambda·rate + lambda_k·kakeya]

        L6 --> KC
        KW --> KC
        KC --> TOTAL
    end
```

### 1.4 挂谷正则对潜空间的影响

```mermaid
graph LR
    subgraph 无挂谷正则
        BAD1[潜空间分布<br/>部分维度坍缩<br/>信息集中]
        BAD2[方向覆盖<br/>存在大量未覆盖方向<br/>投影间距大]
        BAD3[重建效果<br/>潜在容量浪费<br/>泛化能力差]
        BAD1 --> BAD2
        BAD2 --> BAD3
    end

    subgraph 有挂谷正则
        GOOD1[潜空间分布<br/>各维度充分利用<br/>分布均匀]
        GOOD2[方向覆盖<br/>所有方向被覆盖<br/>投影间距小]
        GOOD3[重建效果<br/>潜在容量最大化<br/>泛化能力强]
        GOOD1 --> GOOD2
        GOOD2 --> GOOD3
    end

    BAD2 -.->|挂谷正则<br/>最大化方向覆盖| GOOD2

    subgraph 可视化说明
        VIZ1[2D 示意: 潜变量在单位球面上的分布]
        VIZ2[无正则: 点集中在少数区域<br/>大块空白方向]
        VIZ3[有正则: 点均匀散布<br/>所有方向有覆盖]
        VIZ1 --> VIZ2
        VIZ1 --> VIZ3
    end

    style BAD2 fill:#ffcdd2
    style GOOD2 fill:#c8e6c9
    style VIZ2 fill:#fce4ec
    style VIZ3 fill:#c8e6c9
```

### 1.5 挂谷猜想 → 图像编解码 完整链路

```mermaid
flowchart TD
    subgraph 数学基础
        M1[Kakeya 猜想<br/>R^d 中包含所有方向线段<br/>的最小测度集合]
        M2[Besicovitch 构造<br/>测度可任意小<br/>但方向覆盖完备]
        M3[方向覆盖几何<br/>minimize volume<br/>maximize direction coverage]
        M1 --> M2
        M2 --> M3
    end

    subgraph 算法实现
        A1[潜变量点集<br/>latent → N x C 点<br/>L2 归一化到球面]
        A2[随机投影方向<br/>num_projections=32<br/>随机单位向量]
        A3[投影排序求间距<br/>sort → diff → top-k<br/>k=3]
        A4[覆盖损失<br/>-mean top-k gaps<br/>最大化最小间距]
        A1 --> A2
        A2 --> A3
        A3 --> A4
    end

    M3 -.->|方向覆盖思想| A2
    M3 -.->|最小化最大间距| A4

    subgraph 训练集成 (KakeyaHyperpriorCodec)
        T1[图像 → encoder → latent]
        T2[latent → 挂谷覆盖损失]
        T3[latent → decoder → 重建]
        T4[尺度条件总损失<br/>重建 + 高频 + 颜色 + λ·rate + λₖ·kakeya<br/>Capacity → Transition → Finetune]
        T1 --> T2
        T1 --> T3
        T2 --> T4
        T3 --> T4
    end

    A4 -.-> T2

    subgraph 最终效果
        F1[潜空间充分利用<br/>C 维方向覆盖更均匀]
        F2[目标: 提升重建质量<br/>在给定码率下提高信息利用率]
        F3[码率稳定<br/>熵编码效率优]
        F4[泛化能力强<br/>多尺度多内容]
        F1 --> F2
        F1 --> F3
        F1 --> F4
    end

    T4 -.-> F1

    style M3 fill:#fff9c4
    style A4 fill:#e1f5fe
    style T4 fill:#fff3e0
    style F1 fill:#c8e6c9
```


---

## 2. 模型结构 (KakeyaHyperpriorCodec — 超先验 + 挂谷正则)

```mermaid
graph TD
    subgraph INPUT["Input"]
        IMG[RGB Image<br/>H x W x 3]
    end

    subgraph BASE_STREAM["完整低频 Base 分支（无 InstanceNorm）"]
        IMG --> YCOCG[可逆 RGB → YCoCg<br/>保留 Y / Co / Cg]
        YCOCG --> CPOOL[Area Pool /8<br/>3×H/8×W/8]
        CPOOL --> BAE[Base Analysis<br/>3→4 幅度投影 + 两层残差细化]
        BAE --> CEB[Base EntropyBottleneck<br/>独立概率模型 p(y_base)]
        CEB --> BSD[Base Synthesis<br/>4→3 幅度投影 + 两层残差细化]
        BSD --> C_HAT[低频 Y / Co / Cg]
        C_HAT --> BUP[bilinear ×8<br/>解码低频参考]
    end

    subgraph ANALYSIS["g_a — Detail 分析变换"]
        YCOCG --> DSUB[有符号高频<br/>YCoCg - upsample Base]
        BUP --> DSUB
        DSUB --> HAAR[固定 Haar DWT /2<br/>3→12: LL/LH/HL/HH]
        HAAR --> P24[weight_norm Conv 3x3<br/>12→24]
        P24 --> BIN1[BlendedInstanceNorm<br/>初始 90% IN + 10% raw]
        BIN1 --> RB1[ResidualBlockGDN<br/>24]
        RB1 --> S2D2[SpaceToDepth<br/>24→32, /2]
        S2D2 --> BIN2[BlendedInstanceNorm]
        BIN2 --> RB2[ResidualBlockGDN<br/>32]
        RB2 --> CONV[weight_norm Conv 1x1<br/>32→64]
        CONV --> MS64[MultiScaleResidualBlock<br/>local 3x3 + dilated 3x3]
        MS64 --> SCH64[LightweightSCHBlock<br/>32 local + 32 window-channel<br/>4×4 windows, 4 heads]
        SCH64 --> RB64[ResidualBlockGDN<br/>64]
        RB64 --> PROJ[weight_norm Conv 1x1<br/>64→8]
        PROJ --> TANH[tanh bound ±5<br/>y: 8×H/4×W/4]
    end

    subgraph HYPER["超先验 Hyperprior"]
        TANH --> HA1[Conv 3x3 stride1<br/>8→8]
        HA1 --> HA2[Conv 5x5 stride2<br/>8, 下采样 /2]
        HA2 --> HA3[Conv 5x5 stride2<br/>8, 下采样 /2]
        HA3 --> EB[EntropyBottleneck<br/>8 通道]
        EB --> ATTN[Self-Attention<br/>16×16 → 256 tokens<br/>4 heads × 8-dim<br/>1×1 proj 8→32→8<br/>residual add]
        ATTN --> HD1[ConvTranspose 5x5 stride2<br/>8, 上采样 x2]
        HD1 --> HD2[ConvTranspose 5x5 stride2<br/>8, 上采样 x2]
        HD2 --> HD3[Conv 3x3<br/>8→16]
        HD3 --> H_SPLIT[按通道分为两组<br/>→ scale + mean]
    end

    subgraph CONDITIONAL["条件高斯熵模型"]
        TANH --> GC[GaussianConditional<br/>N(y_hat | mean, scale)]
        H_SPLIT -->|scale, mean| GC
        GC --> Y_HAT[y_hat<br/>量化潜变量]
        GC --> RATE[-log p(y_hat)<br/>空间自适应码率]
    end
    subgraph SYNTHESIS["g_s — Detail 综合变换"]

        Y_HAT --> D0[weight_norm Conv 3x3<br/>8→64]
        D0 --> DSCH[LightweightSCHBlock<br/>window-channel + local]
        DSCH --> DMS[MultiScaleResidualBlock<br/>local + context]
        DMS --> DRB64[ResidualBlockIGDN<br/>64]
        DRB64 --> D1[weight_norm Conv 3x3<br/>64→32]
        D1 --> DRB32[ResidualBlockIGDN<br/>32]
        DRB32 --> UPS[LearnedUpsample<br/>bilinear + sharp residual]
        UPS --> D2[weight_norm Conv 3x3<br/>32→24]
        D2 --> DRB24[ResidualBlockIGDN<br/>24]
        DRB24 --> DHEAD[Haar coefficient head<br/>24→12]
        DRB24 --> DREF[零初始化 coefficient refinement<br/>24→12]
        DHEAD --> IDWT[固定 Haar IDWT ×2<br/>12→3 YCoCg Detail]
        DREF --> IDWT
        IDWT --> DTANH[bound 2 × tanh<br/>有符号高频候选]
        C_HAT --> BASEUP[bilinear ×8<br/>绝对低频 YCoCg]
        DTANH --> CFUSE[expand Base + signed Detail]
        BASEUP --> CFUSE
        CFUSE --> INV[可逆 YCoCg → RGB]
        INV --> OUT[clamp RGB Output<br/>H x W x 3]
    end
    style INPUT fill:#e1f5fe
    style ANALYSIS fill:#f3e5f5
    style HYPER fill:#e8eaf6
    style CONDITIONAL fill:#fce4ec
    style SYNTHESIS fill:#e8f5e9
    style BASE_STREAM fill:#fff9c4
```

### 关键组件细节

```mermaid
graph LR
    subgraph SpaceToDepth
        A[Input] --> PU[PixelUnshuffle /2]
        PU --> C1[Conv]
        C1 --> BIN1[BlendedInstanceNorm]
    end

    subgraph FixedHaarBoundary
        B[3-channel Detail] --> DWT[固定 Haar DWT /2]
        DWT --> BANDS[12 channels<br/>LL/LH/HL/HH]
        BANDS --> IDWT[固定 Haar IDWT ×2]
        IDWT --> BR[3-channel Detail<br/>数值可逆]
    end

    subgraph BlendedInstanceNorm
        X[raw x] --> MIX[x + strength × IN x - x]
        X --> IN[InstanceNorm affine]
        IN --> MIX
        ALPHA[每通道 strength<br/>初始 0.9] --> MIX
    end

    subgraph MultiScaleResidualBlock
        M[Input] --> LOCAL[depthwise 3x3<br/>局部笔画]
        M --> CONTEXT[dilated 3x3<br/>版面上下文]
        LOCAL --> FUSE[concat + 1x1 Conv]
        CONTEXT --> FUSE
        FUSE --> RES[residual_scale 初始 0.1]
        M --> ADD[+]
        RES --> ADD
    end

    subgraph LearnedBaseBranch
        RGB[raw RGB] --> CO[可逆 YCoCg<br/>Y=(R+2G+B)/4]
        CO --> POOL[Area Pool /8]
        POOL --> BA[Base Analysis<br/>无 IN；3→4 latent]
        BA --> CODE[独立 EntropyBottleneck]
        CODE --> BS[Base Synthesis<br/>无 IN；4→3 YCoCg]
        BS --> REPLACE[替换全部低频 Y/Co/Cg<br/>Detail 高频保持不变]
    end

    subgraph DisjointDetailBranch
        SRC[full YCoCg] --> SUB[减去 upsample Base]
        SUB --> CODE_D[InstanceNorm/GDN Detail codec]
        CODE_D --> SIGNED[有符号 YCoCg residual]
        SIGNED --> COMPOSE[+ expanded decoded Base<br/>Laplacian pyramid synthesis]
    end

    subgraph LightweightSCH
        SX[64 channels] --> SPLIT_S[1×1 Conv + split]
        SPLIT_S --> LOCAL_S[32 local<br/>multi-scale depthwise Conv]
        SPLIT_S --> GLOBAL_S[32 window context<br/>4×4 window-channel attention]
        LOCAL_S --> FUSE_S[concat + 1×1 Conv]
        GLOBAL_S --> FUSE_S
        FUSE_S --> RES_S[residual scale 0.1]
        SX --> RES_S
    end
```

| 维度 | 说明 |
|---|---|
| 熵模型 | 独立因子分解 `p(z)·p(y_detail\|z)·p(y_base)`：EntropyBottleneck（z/Base）+ GaussianConditional（Detail） |
| 激活函数 | 分析侧 Residual GDN；合成侧 Residual IGDN |
| 归一化 | BlendedInstanceNorm，初始 90% IN + 10% 原始幅度路径 |
| 多尺度块 | 局部 3x3 与 dilation=2 分支融合 |
| 下采样 | Detail 边界固定 Haar DWT 2x + SpaceToDepth 2x = 4x |
| 上采样 | LearnedUpsample 2x + 固定 Haar IDWT 2x |
| SCH | 64 通道各放一个对称块；32 local + 32 window-channel，4×4 window / 4 heads；高清按 2048 windows 分块以限制峰值内存 |
| Detail 输入 | `YCoCg - upsample(low-frequency YCoCg)`；不再重复编码 Base |
| Detail 输出 | `2·tanh` 有符号 YCoCg 残差，直接与 expanded Base 相加；无 Sigmoid |
| Base 分支 | 完整低频 Y/Co/Cg、/8 采样、4 通道可学习 latent；无 IN，独立码流 |
| 损失函数 | MSE + edge + structural + multiscale + Laplacian + LAB + Base + Detail high-pass + λ·rate + λₖ·kakeya |
| h_s 增强 | Self-Attention；大图使用 16x16 窗口 |
| 潜空间 | 8 通道, ±5 bound, GaussianConditional 量化 |
| checkpoint | 架构版本 8，Kakeya Hyperprior v10 `[z,y_detail,y_base]`；明确不兼容旧架构 |
---

## 3. 损失函数组成

当前 KakeyaHyperpriorCodec 使用三阶段、尺度条件的重建与率失真损失：

```math
L = w_m·MSE + w_e·Edge + w_s·(1-SSIM) + w_{ms}·MultiL1
  + w_{hf}·Laplacian + w_l·ΔE + w_h·Hue + w_{sat}·Sat
  + w_b·BaseL1 + w_d·DetailL1
  + λₖ·Kakeya + λ·max(0, bpp - 2.5)
```

- MSE: 重建像素误差
- Edge: 梯度 L1 边缘损失
- Structural: 1 - SSIM 结构相似性
- Multiscale: 多尺度 L1（256/128/64 加权）
- Laplacian: 256 及以下启用的二阶高频损失
- LAB: CIELAB ΔE 色差 + 色相 + 饱和度
- BaseL1: `/8` 低频 Y/Co/Cg L1，约束绝对亮度、对比度和色度
- DetailL1: 去除 Base 后的有符号高频 Y/Co/Cg L1，直接监督文字与纹理
- Kakeya: 潜空间方向覆盖正则化
- Rate: 熵编码码率（`log2`），超出 2.5 bpp 的部分产生惩罚
- λ = 0.01（默认），λₖ = 0.001（默认）
- 各阶段权重由 `config.stage_weights()` 管理
- 256 及以下：rate ×0.35、edge ≥5、structural ≥1.2、multiscale ≤0.2
- 384 及以上：保持原阶段权重
---
---

---

## 4. 数据流 (Data Pipeline)

```mermaid
flowchart LR
    subgraph DataSources
        REF_IMG[参考图<br/>kakeya_codec_card_v2_256.png]
        PROC[程序生成图<br/>几何+文字<br/>512x512 源]
        HD_IMG[真实高清图<br/>assets/hd_images/<br/>用户提供的照片]
    end

    subgraph ProceduralDocumentDataset
        PD1[_pick_target_size<br/>50% → 256<br/>50% → 128/192/256/384/512/768]
        PD2[index % 2 == 0<br/>参考图 resize]
        PD3[index % 2 == 1<br/>程序图 resize]
    end

    subgraph RealImageDataset
        RI1[加载 assets/hd_images/<br/>jpg png webp bmp]
        RI2[center_crop_square<br/>取最短边居中裁剪]
        RI3[_pick_target_size<br/>多尺度 resize]
        RI1 --> RI2 --> RI3
        RI4[空目录 fallback<br/>→ 参考图 512x512]
        RI4 -.-> RI1
    end

    subgraph CalibrationCardDataset
        CC1[multiscale=False<br/>固定 256 参考图]
        CC2[multiscale=True<br/>参考图多尺寸]
    end

    REF_IMG --> PD2
    REF_IMG --> CC1
    REF_IMG --> CC2
    PROC --> PD3
    HD_IMG --> RI1
    PD1 --> PD2
    PD1 --> PD3

    PD2 --> COLLATE
    PD3 --> COLLATE
    RI3 --> COLLATE

    subgraph Collate
        COLLATE[_size_aware_collate<br/>按尺寸分组<br/>返回 mini_batches 列表]
    end

    CC1 --> BATCH1[batch_size=1]
    CC2 --> BATCH2[batch_size=1]
    COLLATE --> BATCH3[batch_size=config<br/>4-2048]

    BATCH1 --> LOADERS
    BATCH2 --> LOADERS
    BATCH3 --> LOADERS

    subgraph Loaders
        L1[train_loader<br/>ProceduralDocumentDataset<br/>程序图 128+ 样本]
        L2[real_loader<br/>RealImageDataset<br/>真实高清图 16+ 样本]
        L3[validation_loader<br/>ProceduralDocumentDataset<br/>16-64 样本]
    end


    L1 --> MIXER[Balanced step mixer<br/>每 10 个真实优化步骤]
    L2 --> MIXER
    CC2 --> MIXER
    MIXER --> RATIO[4 reference + 3 procedural + 3 real<br/>真实梯度比例 40/30/30]

    COLLATE --> L1
    COLLATE --> L2
    COLLATE --> L3
```

---

## 5. 训练流程 (train_image_codec / _hyperprior_epoch)

KakeyaHyperpriorCodec 使用 Capacity → Transition → Finetune 三阶段权重调度。
熵瓶颈 `quantiles` 只由辅助优化器更新，其他参数由 AdamW 更新；两组参数互斥，
每次更新前均执行有限性检查和梯度裁剪。

```mermaid
flowchart TD
    START([开始训练]) --> INIT[初始化架构版本 8<br/>Base-first Haar-SCH Detail + YCoCg Base]
    INIT --> SPLIT_OPT[分离优化器参数<br/>main 不含 quantiles<br/>aux 仅含 quantiles]
    SPLIT_OPT --> LOADERS[创建 DataLoaders<br/>Reference + Procedural + RealImage + Validation]
    LOADERS --> LOOP{每个 epoch<br/>1 → config.epochs}

    LOOP -->|是| MIX[Balanced step mixer<br/>4 ref + 3 proc + 3 real]
    MIX --> TRAIN[_hyperprior_epoch<br/>100 个实际优化步骤]
    TRAIN --> CLIP[finite gradient check<br/>clip norm ≤ 5]
    CLIP --> VALID[_hyperprior_epoch<br/>验证集<br/>train=False]
    VALID --> CALIB[_calibration_metrics<br/>确定性前向量化<br/>不构建 CDF]
    CALIB --> HD[512 高清校准<br/>PSNR/SSIM]
    HD --> CKPT{256 PSNR 创新高<br/>高清 PSNR/SSIM/Chroma<br/>回退均未越界?}
    CKPT -->|是| SAVE[保存 best.pt<br/>architecture version 8]
    SAVE --> LOOP
    CKPT -->|否| LOOP

    LOOP -->|否| FINAL[保存 final.pt]
    FINAL --> UPDATE[model.update<br/>熵模型 CDF 表]
    UPDATE --> EVAL[_evaluate_chart<br/>还原图 / 码流]
    EVAL --> RATE_CK[_rate_consistency_check]
    RATE_CK --> BASELINE[_compressai_baselines<br/>对比 JPEG/WebP/CompressAI]
    BASELINE --> DONE([训练完成])
```

### 5.1 _hyperprior_epoch 细节

每个 epoch 对每个 DataLoader 批次执行：

1. `images = images.to(device)` — 数据迁移到 GPU/MPS
2. `reconstructed, y, _, y_hat, y_likelihoods, z_likelihoods = model(images)` — 前向传播
3. `loss_mse = F.mse_loss(reconstructed, images)` — 重建损失
4. `rate = (-log p(y_hat) - log p(z_hat)) / batch_size` — 熵编码码率
5. `bpp = rate / (H * W)` — 每像素比特
6. `coverage = kakeya_regularization(unit_normalized_latent, num_projections=32, k=3)` — 挂谷覆盖正则
7. 按尺寸选择小图条件权重或原高清权重
8. 计算 MSE、edge、SSIM、multiscale、Laplacian、LAB、Base、Detail high-pass、rate 与 Kakeya 总损失
9. `total.backward()` 后检查梯度有限性并裁剪到 5，再更新主参数
10. `aux_loss` 只更新 EntropyBottleneck 的 `quantiles`

---

## 6. 推理流程 (Web API / CLI)

```mermaid
flowchart TD
    UPLOAD([上传图片<br/>或 API 请求]) --> PARSE[解析图片<br/>PIL.Image RGB]

    PARSE --> SIZE_CHK{尺寸检查<br/>16-4096px}
    SIZE_CHK -->|过大| ERR1[400: 图片过大]
    SIZE_CHK -->|过小| ERR2[400: 图片过小]
    SIZE_CHK -->|OK| PAD[16 倍数边缘复制对齐<br/>保证 Base / Haar / SCH 网格一致]

    PAD --> LOAD_CKPT[加载 checkpoint<br/>要求 architecture version 8<br/>严格校验学习参数]
    LOAD_CKPT --> MODEL[KakeyaHyperpriorCodec<br/>eval mode]

    MODEL --> BASE_ENC[YCoCg /8 Base Analysis]
    BASE_ENC --> BCODE[Base EntropyBottleneck<br/>编码并解码 y_base]
    BCODE --> BASE_DEC[Base Synthesis<br/>decoded Base reference]
    MODEL --> DETAIL_SUB[YCoCg - expand decoded Base]
    BASE_DEC --> DETAIL_SUB
    DETAIL_SUB --> HAAR_ENC[固定 Haar DWT<br/>方向子带]
    HAAR_ENC --> ENCODE[y_detail = Haar-SCH encode(detail)]
    ENCODE --> MODEL_UPDATE[model.update<br/>熵模型 CDF 表]
    MODEL_UPDATE --> ZCODE[EntropyBottleneck<br/>compress/decompress z]
    ZCODE --> PARAMS[h_s + Attention<br/>生成 mean/scale]
    ENCODE --> YCODE[GaussianConditional<br/>条件编码 y]
    PARAMS --> YCODE
    YCODE --> RECONSTRUCT[IGDN + SCH<br/>预测 12 Haar coefficients]
    RECONSTRUCT --> HAAR_DEC[固定 Haar IDWT<br/>有符号 YCoCg Detail]
    HAAR_DEC --> CFUSE[expand Base + signed Detail]
    BASE_DEC --> CFUSE
    CFUSE --> CLAMP[YCoCg → RGB<br/>clamp 0, 1]

    CLAMP --> CROP[去除 padding<br/>恢复原始尺寸]
    CROP --> METRICS[计算真实 bytes / bpp<br/>PSNR / SSIM]
    METRICS --> RESP[返回重建图、误差图 + metrics]
```

### 6.1 训练前向与推理前向的关系

每个 epoch 对每个 DataLoader 批次执行：

1. `images = images.to(device)` — 数据迁移到 GPU/MPS
2. `reconstructed, y, _, y_hat, y_likelihoods, z_likelihoods = model(images)` — 训练时使用可微熵模型前向
3. `loss_mse = F.mse_loss(reconstructed, images)` — 重建损失
4. `rate = (-log2 p(y_hat) - log2 p(z_hat)) / batch_size` — 熵编码码率（bits）
5. `bpp_bits = rate / (H * W)` — 每像素比特
6. `rate_penalty = ReLU(bpp_bits - 2.5)` — 仅超出目标的码率产生惩罚
7. 感知损失：edge、structural、multiscale、Laplacian、lab、hue、saturation
8. `total = w_m·MSE + w_e·edge + w_s·structural + ... + λₖ·coverage + λ·rate_penalty` — 总损失
9. 主损失反向传播后检查有限性、裁剪梯度，再更新非 quantiles 参数
10. `aux_loss` 仅更新 EntropyBottleneck 的 quantiles

推理时不调用这段训练循环，而是先 `model.update()` 构建 CDF，再走
`compress → decompress` 的真实字节流路径。epoch 校准使用确定性模型前向，
不会在训练过程中反复重建 CDF。
---



---

## 8. Web API 端点结构

```mermaid
graph TD
    subgraph Frontend
        UI[Next.js Frontend<br/>localhost:3000]
    end

    subgraph WebAPI [FastAPI :8000]
        HEALTH[GET /api/health]
        ENV[GET /api/environment]
        ENV_INST[POST /api/environment/install]

        EXP_LIST[GET /api/experiments]
        EXP_CREATE[POST /api/experiments]
        EXP_GET[GET /api/experiments/:id]
        EXP_STOP[POST /api/experiments/:id/stop]
        EXP_RESULT[GET /api/experiments/:id/result]
        EXP_IMG[GET /api/experiments/:id/image/:kind]
        EXP_EVENTS[GET /api/experiments/:id/events<br/>SSE]

        BITSTREAM[GET /api/experiments/:id/artifact/bitstream]
        CHECKPOINT[GET /api/experiments/:id/artifact/checkpoint]
        OPEN_DIR[POST /api/experiments/:id/artifact/open-checkpoint-dir]

        RECON_UP[POST /api/experiments/:id/reconstruct<br/>上传图片重建]
        RECON_CUSTOM[POST /api/reconstruct-custom<br/>上传模型+图片]

        TEST_IMG[GET /api/test-image]
    end

    subgraph JobManager
        JM[JobManager<br/>管理训练进程]
        JM_LIST[list]
        JM_CREATE[create]
        JM_GET[get]
        JM_STOP[stop]
        JM_RESULT[result]
    end

    subgraph TrainingProcess
        WORKER[子进程<br/>python -m kakeya.web_worker]
        RUNNER[Runner<br/>train_image_codec]
        EPOCH_CB[epoch_callback<br/>写入 metrics.json]
    end

    UI -->|HTTP/SSE| WebAPI
    EXP_CREATE --> JM_CREATE
    JM_CREATE --> WORKER
    WORKER --> RUNNER
    RUNNER --> EPOCH_CB
    EPOCH_CB -->|metrics.json| EXP_EVENTS
    EXP_EVENTS -->|SSE snapshot| UI

    EXP_LIST --> JM_LIST
    EXP_GET --> JM_GET
    EXP_STOP --> JM_STOP
    EXP_RESULT --> JM_RESULT

    RECON_UP --> DO_RECON[_do_reconstruct]
    RECON_CUSTOM --> DO_RECON
    DO_RECON --> ENCODE_DECODE[加载 checkpoint<br/>encode → compress → decompress → decode]

    style Frontend fill:#e1f5fe
    style WebAPI fill:#fff3e0
    style JobManager fill:#f3e5f5
    style TrainingProcess fill:#e8f5e9
```

---

## 9. 泛化能力与高清大图处理

> 本节记录项目实际遇到的问题、诊断过程与优化方案。

### 9.1 泛化问题诊断

模型曾出现严重的过拟合——只学会重建 256² 参考图，对其他内容失效：

```mermaid
graph TD
    subgraph 诊断结果
        T1[256² 参考图<br/>整图重建<br/>PSNR 29 dB ✓]
        T2[256² 程序生成图<br/>PSNR 8-12 dB ✗]
        T3[512² 参考图<br/>整图重建<br/>PSNR 13 dB ✗]
        T4[512² 参考图<br/>裁剪 256² 重建<br/>PSNR 12 dB ✗]
        T5[256² 重建结果<br/>双线性放大到 512²<br/>PSNR 26 dB]
    end

    T1 -.->|同尺寸不同内容| T2
    T1 -.->|同内容不同尺寸| T3
    T3 -.->|裁剪后单独重建| T4
    T1 -.->|先重建再放大| T5

    subgraph 根因
        R1[capacity_stage<br/>只用 capacity_loader<br/>单一参考图多尺寸]
        R2[模型深度过拟合<br/>到参考图像素分布]
        R3[gate 后 fine-tune<br/>学习率低 + rate loss<br/>无法纠正过拟合]
        R1 --> R2
        R2 --> R3
    end

    R1 -.->|导致| T2
    R2 -.->|导致| T3
    R2 -.->|导致| T4

    style T1 fill:#c8e6c9
    style T2 fill:#ffcdd2
    style T3 fill:#ffcdd2
    style T4 fill:#ffcdd2
    style T5 fill:#fff9c4
    style R1 fill:#fce4ec
    style R2 fill:#fce4ec
    style R3 fill:#fce4ec
```

### 9.2 修复后的训练数据策略

```mermaid
graph TD
    subgraph 修复前
        OLD_CAP[capacity_loader<br/>仅参考图多尺寸<br/>32 steps]
        OLD_CAP -->|内容单一| OVERFIT[过拟合参考图<br/>颜色分布窄]
    end

    subgraph 修复后
        NEW_CAP_REF[capacity_loader<br/>参考图多尺寸<br/>32 steps]
        NEW_CAP_PROG[train_loader<br/>程序图多尺度<br/>128+ 样本]
        NEW_CAP_REAL[real_loader<br/>真实高清图<br/>assets/hd_images/<br/>多样色彩与纹理]
        NEW_REH[rehearsal_loader<br/>参考图多尺寸<br/>8 steps]

        NEW_CAP_REF --> MERGE[交错训练步骤<br/>每 10 步: 4 ref + 3 prog + 3 real]
        NEW_CAP_PROG --> MERGE
        NEW_CAP_REAL --> MERGE
        MERGE --> NEW_REH
        NEW_REH -->|保证| GATE[闸门可通过<br/>参考图 PSNR≥26<br/>或 epoch≥40 强制过闸]
        MERGE -->|保证| GEN[真实梯度比例 40/30/30<br/>不再只是指标加权]
    end

    OVERFIT -.->|修复| MERGE

    style OVERFIT fill:#ffcdd2
    style GATE fill:#c8e6c9
    style GEN fill:#c8e6c9
    style MERGE fill:#fff9c4
    style NEW_CAP_REAL fill:#e8f5e9
```

### 9.3 高清大图推理流程

推理时不切分图片、不缩放，整图通过模型（边缘复制 padding 到 16 的倍数；
SCH attention 内部按 window strip 分块限制临时内存）：

```mermaid
flowchart TD
    INPUT([输入图片<br/>任意尺寸 WxH]) --> CHK{尺寸检查}
    CHK -->|W or H > 4096| REJECT[拒绝: 超过 4096px]
    CHK -->|OK| PAD[16 倍数边缘复制对齐<br/>Base / Haar / SCH 共同对齐]

    PAD --> LOAD[加载 architecture v8 checkpoint<br/>严格校验学习参数]
    LOAD --> BASE[完整低频 YCoCg<br/>/8 → 4 channel latent]
    BASE --> BCODE[Base EntropyBottleneck<br/>编码并解码]
    BCODE --> BASE_DEC[Base Synthesis<br/>decoded Base reference]
    LOAD --> DETAIL_SUB[YCoCg - expand decoded Base]
    BASE_DEC --> DETAIL_SUB
    DETAIL_SUB --> DWT_HD[固定 Haar DWT<br/>LL/LH/HL/HH]
    DWT_HD --> ENCODE[Haar-SCH Detail 分析<br/>→ H/4 x W/4 x 8]
    ENCODE --> Y[y = tanh bound ±5]
    Y --> HA[h_a → z]
    HA --> ZCODE[EntropyBottleneck<br/>编码并解码 z]
    ZCODE --> PARAMS[h_s + Attention<br/>mean / scale]
    Y --> YCODE[GaussianConditional<br/>条件编码并解码 y]
    PARAMS --> YCODE
    YCODE --> DECODE[IGDN + SCH<br/>预测 Haar coefficients]
    DECODE --> IDWT_HD[固定 Haar IDWT<br/>有符号 YCoCg Detail]
    IDWT_HD --> CFUSE[expand Base + signed Detail]
    BASE_DEC --> CFUSE

    CFUSE --> CLAMP[clamp 0, 1]
    CLAMP --> CROP[去除 padding<br/>恢复 WxH]
    CROP --> OUT([输出重建图<br/>+ PSNR/SSIM/bpp])

    style INPUT fill:#e1f5fe
    style ENCODE fill:#f3e5f5
    style ZCODE fill:#fff3e0
    style DECODE fill:#e8f5e9
    style OUT fill:#c8e6c9
    style REJECT fill:#ffcdd2
```

### 9.4 多尺度训练覆盖范围

```mermaid
graph LR
    subgraph 训练尺寸分布
        S128[128²<br/>~8%]
        S192[192²<br/>~8%]
        S256[256²<br/>~58%<br/>50% primary + 8% pool]
        S384[384²<br/>~8%]
        S512[512²<br/>~8%]
        S768[768²<br/>~8%]
    end

    subgraph 推理覆盖
        I_SMALL[小图<br/>16-256px<br/>直接处理 ✓]
        I_MED[中图<br/>256-768px<br/>训练分布内 ✓]
        I_LARGE[大图<br/>768-4096px<br/>外推 ⚠️]
    end

    S128 --> I_SMALL
    S192 --> I_SMALL
    S256 --> I_SMALL
    S256 --> I_MED
    S384 --> I_MED
    S512 --> I_MED
    S768 --> I_MED
    S768 -.->|外推| I_LARGE

    subgraph 架构保证
        ARCH1[BlendedInstanceNorm<br/>90% IN + 10% raw 初始值<br/>每通道可学习]
        ARCH2[WeightNorm<br/>权重归一化<br/>尺寸无关]
        ARCH3[全卷积<br/>无全连接层<br/>任意尺寸]
        ARCH4[固定 Haar + PixelUnshuffle<br/>方向子带与空间重排<br/>尺寸无关]
        ARCH5[512 高清 checkpoint 闸门<br/>PSNR/SSIM 回退保护]
        ARCH6[完整低频 YCoCg Base<br/>绕过 InstanceNorm]
        ARCH7[轻量 SCH<br/>增强局部窗口通道建模]
    end

    ARCH1 -.->|支持| I_LARGE
    ARCH2 -.->|支持| I_LARGE
    ARCH3 -.->|支持| I_LARGE
    ARCH4 -.->|支持| I_LARGE
    ARCH5 -.->|保护| I_LARGE
    ARCH6 -.->|保护颜色| I_LARGE
    ARCH7 -.->|窗口通道上下文| I_LARGE

    style S256 fill:#c8e6c9
    style I_SMALL fill:#c8e6c9
    style I_MED fill:#c8e6c9
    style I_LARGE fill:#fff9c4
```

### 9.5 高清大图颜色与亮度问题

#### 当前方案：可学习的 Base / Detail 分解

架构 v8 先把 RGB 可逆转换为 YCoCg，再显式分解成 `/8` Base low-pass 和有符号
Detail high-pass。完整低频 Y/Co/Cg 由不含 InstanceNorm 的 Base Analysis /
Synthesis 编码为 4 通道潜变量；编码端先量化并重建 Base，Detail 编码器再接收
减去这个 decoded Base 后的残差，从而能在 Detail 中补偿 Base 量化误差，
不再为最终会被覆盖的低频重复付费。Detail 边界使用固定 Haar DWT/IDWT，把
LL/LH/HL/HH 方向子带交给网络学习；64 通道瓶颈使用轻量 SCH，把局部多尺度卷积
与 4×4 window-channel attention 融合。Detail 解码器预测 12 个 Haar coefficients，
经 IDWT 生成有符号 YCoCg 残差，不经过 Sigmoid。合成仍使用严格可逆的
Laplacian pyramid 形式 `expand(Base) + Detail`。

熵模型刻意保持为 `p(z)·p(y_detail|z)·p(y_base)`，没有引入联合上下文或串行
自回归依赖。`.kky` v10 使用 `[z,y_detail,y_base]` 三段；编码端 Base-first，解码端仍不需要
像素级串行自回归。项目以实验效果优先，架构版本提升到 8，旧 checkpoint 与 v9 bitstream
明确不兼容。Haar
不含可学习参数，SCH 只放在 H/4 的对称瓶颈，不把论文的 256/320 通道大模型或
5-slice 自回归熵模型照搬进当前项目。

训练数据通过 step mixer 按实际优化步骤执行 40% reference / 30% procedural /
30% real；checkpoint 除 PSNR/SSIM 外还保护高清 chroma MAE。

#### 9.5.1 历史诊断：过拟合导致颜色丢失

模型早期出现严重的过拟合问题——只学会重建 256² 参考图，对其他内容完全失效：

```mermaid
flowchart TD
    subgraph 问题现象
        PHEN1[小图 256px<br/>颜色正常<br/>PSNR 29 dB]
        PHEN2[大图 512px+<br/>颜色丢失或偏色<br/>PSNR 13 dB]
    end

    subgraph 诊断路径
        D1[测试: 512 裁剪 256 单独重建]
        D2[结果: PSNR 12 dB<br/>和整图 512 一样差]
        D3[结论: 不是尺寸问题<br/>是内容泛化问题]
        D4[测试: 256 程序生成图]
        D5[结果: PSNR 8-12 dB]
        D6[结论: 模型只学会<br/>重建参考图]
        D1 --> D2 --> D3
        D4 --> D5 --> D6
    end

    PHEN2 --> D1
    PHEN2 --> D4

    subgraph 根因链
        ROOT1[capacity_stage<br/>30+ epochs<br/>只用参考图训练]
        ROOT2[模型权重过拟合<br/>到参考图特定像素]
        ROOT3[encoder 学到的特征<br/>无法处理新内容]
        ROOT4[decoder 输出<br/>偏离正确颜色分布]
        ROOT1 --> ROOT2
        ROOT2 --> ROOT3
        ROOT3 --> ROOT4
    end

    D6 --> ROOT1
    ROOT4 --> PHEN2

    subgraph 已完成修复
        FIX1[capacity_stage 加入<br/>train_loader 程序图]
        FIX2[每 epoch 混合训练<br/>0.5 ref + 0.5 prog]
        FIX3[rehearsal 全阶段<br/>保持参考图能力]
        FIX4[capacity_stage 开启<br/>挂谷正则 0.001]
        FIX1 --> FIX2
        FIX2 --> FIX3
        FIX3 --> FIX4
    end

    ROOT1 -.->|修复| FIX1
    FIX4 -.->|解决| PHEN2

    style PHEN1 fill:#c8e6c9
    style PHEN2 fill:#ffcdd2
    style D3 fill:#fff9c4
    style D6 fill:#fff9c4
    style ROOT1 fill:#fce4ec
    style ROOT4 fill:#fce4ec
    style FIX4 fill:#c8e6c9
```

#### 9.5.2 失真问题的优化：分阶段损失调度

训练后期出现明显失真，根因是**部分损失项在后期才突然引入**——capacity/transition 阶段关闭 mse/structural，到 finetune 阶段才一次性启用，造成梯度方向突变、模型震荡。

**解决方案：所有损失方向全程参与，各阶段仅权重递增**——让模型从第一轮就在所有损失方向上学习，避免后期冷启动造成的梯度突变，各阶段通过权重递进调整侧重。权重统一由 `config.stage_weights()` 管理（[config.py](../src/kakeya/config.py) `DEFAULT_STAGE_WEIGHTS`），可在 YAML / API 覆盖。

| 阶段 | 损失权重 (config.stage_weights) | kakeya | 设计意图 |
|------|--------------------------------|--------|----------|
| Capacity | mse 3.0 / structural 0.1 / lab 0.02 | 0.002 | 专注方向覆盖，mse/structural/lab 用低权重做"种子" |
| Transition | mse 5.0 / structural 0.25 / lab 0.05 | 0.001 | 权重递增，感知约束逐步强化 |
| Finetune | mse 5.0 / structural 0.5 / lab 0.10 + rate | 0.0005 | 全部权重最高，协同优化码率与画质 |

```mermaid
flowchart TD
    subgraph 问题现象
        P1[训练后期失真明显<br/>细节模糊 / 色块 / 偏色]
        P2[损失曲线震荡<br/>损失项突然加入造成梯度突变]
    end

    subgraph 根因分析
        R1[部分损失项后期才启用]
        R2[capacity/transition 关闭 mse/structural<br/>finetune 一次性启用]
        R3[冷启动梯度方向<br/>与已学方向冲突]
        R1 --> R2 --> R3
    end

    subgraph 解决方案：所有损失全程参与，权重递增
        S1[Capacity: 全部损失参与<br/>mse 3.0 / structural 0.1 / lab 0.02<br/>kakeya 0.002]
        S2[Transition: 权重递增<br/>mse 5.0 / structural 0.25 / lab 0.05<br/>kakeya 0.001]
        S3[Finetune: 权重最高<br/>structural 0.5 / lab 0.10 + rate<br/>kakeya 0.0005]
        S1 --> S2 --> S3
    end

    subgraph 效果
        E1[无冷启动<br/>所有方向从早期学习]
        E2[过渡平滑<br/>权重递进避免梯度突变]
        E3[微调精细<br/>感知质量更优]
        E4[整体失真明显减少]
        E1 --> E2 --> E3 --> E4
    end

    P1 --> R1
    P2 --> R2
    S3 -.->|解决| P1
    S2 -.->|解决| P2

    style P1 fill:#ffcdd2
    style P2 fill:#ffcdd2
    style R1 fill:#fce4ec
    style S3 fill:#c8e6c9
```

#### 9.5.3 经验教训：损失函数 vs 数据质量

**核心结论：对于颜色还原问题，训练数据的多样性和覆盖面远比损失函数的精细调整重要。**

```mermaid
graph LR
    subgraph 损失函数优化路线
        L1[增加更多损失项<br/>LAB / perceptual / detail]
        L2[调权重 / 加自适应机制]
        L3[训练速度下降<br/>调参成本高<br/>边际收益低]
        L1 --> L2 --> L3
    end

    subgraph 数据质量路线
        D1[增加真实多样的训练图片]
        D2[扩展颜色与纹理分布]
        D3[模型自然学到通用映射<br/>效果立竿见影<br/>训练速度不受影响]
        D1 --> D2 --> D3
    end

    subgraph 对比结论
        C1[数据 > 损失函数<br/>好数据胜过精巧的损失设计]
        C2[损失函数是放大器<br/>数据分布决定上限]
    end

    L3 -.->|对比| C1
    D3 -.->|对比| C1
    C1 --> C2

    style L3 fill:#fff9c4
    style D3 fill:#c8e6c9
    style C1 fill:#e1f5fe
    style C2 fill:#e1f5fe
```

### 9.6 码率一致性检查 (rate consistency)

训练结束后验证多尺度码率稳定性：

```mermaid
flowchart LR
    subgraph 测试流程
        SRC[参考图 256²] --> SCALE1[0.5x → 128²]
        SRC --> SCALE2[1.0x → 256²]
        SRC --> SCALE3[1.5x → 384²]
        SRC --> SCALE4[2.0x → 512²]

        SCALE1 --> ENC1[encode + rate_bpp]
        SCALE2 --> ENC2[encode + rate_bpp]
        SCALE3 --> ENC3[encode + rate_bpp]
        SCALE4 --> ENC4[encode + rate_bpp]

        ENC2 --> REF_BPP[reference_bpp]
        ENC1 --> DEV1[bpp_deviation]
        ENC3 --> DEV3[bpp_deviation]
        ENC4 --> DEV4[bpp_deviation]

        REF_BPP --> MAX_DEV[max_deviation]
        DEV1 --> MAX_DEV
        DEV3 --> MAX_DEV
        DEV4 --> MAX_DEV
    end

    subgraph 期望结果
        GOOD[一致性良好<br/>max_deviation < 5%<br/>大图 bpp 稳定]
        BAD[一致性差<br/>max_deviation > 20%<br/>大图码率漂移]
    end

    MAX_DEV --> GOOD
    MAX_DEV -.->|过拟合时| BAD

    style GOOD fill:#c8e6c9
    style BAD fill:#ffcdd2
    style MAX_DEV fill:#fff9c4
```
