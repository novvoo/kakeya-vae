"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

function FormattedDate({ dateString }: { dateString: string }) {
  return (
    <time suppressHydrationWarning>
      {new Date(dateString).toLocaleString("zh-CN")}
    </time>
  );
}

const API_BASE =
  process.env.NEXT_PUBLIC_KAKEYA_API_URL ?? "http://127.0.0.1:8000";

type Status =
  | "queued"
  | "running"
  | "evaluating"
  | "stopping"
  | "completed"
  | "failed"
  | "cancelled";

type MetricMap = Record<string, number>;

type SeriesPoint = {
  epoch: number;
  train: MetricMap;
  validation: MetricMap;
};

type Job = {
  id: string;
  config: ExperimentConfig;
  device: string;
  status: Status;
  epoch: number;
  total_epochs: number;
  progress: number;
  message: string;
  error: string | null;
  run_dir: string | null;
  metrics: MetricMap | null;
  series: SeriesPoint[];
  logs: string[];
  created_at: string;
  has_result: boolean;
};

type Environment = {
  ready: boolean;
  python: string;
  packages: Record<string, { installed: boolean; version: string | null }>;
  install_status: "idle" | "running" | "completed" | "failed";
  install_logs: string[];
  install_error: string | null;
  device: {
    recommended: string;
    mps_available: boolean;
    cuda_available: boolean;
    label: string;
  };
};

type ExperimentConfig = {
  method: string;
  epochs: number;
  latent_dim: number;
  batch_size: number;
  learning_rate: number;
  seed: number;
  num_workers: number;
  train_limit: number;
  test_limit: number;
  num_projections: number;
  k: number;
  lambda_rate: number;
  lambda_kakeya: number;
  stage_weights?: Record<string, Record<string, number>>;
};
type ResultPayload = {
  config: ExperimentConfig;
  history: {
    epoch: number[];
    train: Record<string, number[]>;
    validation: Record<string, number[]>;
  };
  metrics: MetricMap & { variance_spectrum?: number[] };
  latent: Array<{ x: number; y: number; label: number }>;
  runtime?: { device: string };
  image_codec?: {
    image_size: number;
    test_asset: string;
    test_role: string;
    latent_shape?: number[];
    images: {
      original?: string;
      reconstruction?: string;
      error?: string;
      original_hd?: string;
      reconstruction_hd?: string;
      error_hd?: string;
    };
    training?: {
      capacity_gate_passed: boolean;
      capacity_gate_epoch: number | null;
      capacity_gate: { psnr: number; ssim: number };
      selected_checkpoint_epoch: number | null;
      selected_checkpoint_psnr: number | null;
      selected_checkpoint_rate_bpp: number | null;
      target_rate_bpp: number;
      final_stage: "capacity_pretrain" | "compression_finetune";
      compression_finetune_epochs: number;
    };
    bitstream?: {
      path: string;
      filename: string;
      format: string;
      bytes: number;
      payload_bytes: number;
      header_bytes: number;
      bpp: number;
      requires_checkpoint: boolean;
      checkpoint: string;
    };
  };
  codec_baselines?: Array<{
    codec: string;
    settings: string;
    bytes: number;
    mse: number;
    psnr: number;
    ssim: number;
  }>;
};

const DEFAULT_CONFIG: ExperimentConfig = {
  method: "image_codec",
  epochs: 80,
  latent_dim: 8,
  batch_size: 4,
  learning_rate: 0.0005,
  seed: 42,
  num_workers: 0,
  train_limit: 128,
  test_limit: 0,
  num_projections: 32,
  k: 3,
  lambda_rate: 1.0,
  lambda_kakeya: 0.001,
};

const METHOD_LABELS: Record<string, string> = {
  image_codec: "Kakeya VAE (超先验 + 挂谷)",
};

const IMAGE_CODEC_REFERENCES = [
  {
    name: "SAAF",
    venue: "CVPR 2026",
    focus: "稀疏注意力与自适应频率的学习式图像压缩",
    relation: "直接优化率失真，并在 256×256 图像块上训练",
    gap: "当前已有基础因子化熵瓶颈；下一步是超先验、上下文模型和自适应频率建模",
    reported:
      "论文报告：相对 VTM-9.1，Kodak / CLIC / Tecnick BD-rate 为 −17.40% / −17.35% / −20.57%。",
    href: "https://openaccess.thecvf.com/content/CVPR2026/papers/Ma_Learned_Image_Compression_via_Sparse_Attention_and_Adaptive_Frequency_CVPR_2026_paper.pdf",
  },
  {
    name: "Diff-ICMH",
    venue: "NeurIPS 2025",
    focus: "生成先验、视觉质量与语义保真统一",
    relation: "图文中的文字属于必须保持的关键语义信息",
    gap: "需增加 OCR 一致性损失和语义保真指标",
    reported: "论文公开结果用于路线参照；默认不在本机复现其扩散模型训练。",
    href: "https://proceedings.neurips.cc/paper_files/paper/2025/hash/5c33e9aedee21daeda9e03f43ec4865d-Abstract-Conference.html",
  },
  {
    name: "DC-AE",
    venue: "ICLR 2025",
    focus: "高空间压缩比下保持图像重建精度",
    relation: "同样采用空间潜变量，残差自编码缓解高压缩训练困难",
    gap: "当前已有 space-to-depth 路径和两阶段训练；仍需扩大数据与模型容量",
    reported: "官方提供 Diffusers 预训练权重；它主要服务潜空间生成，不等同于可输出 bitstream 的图像编码器。",
    href: "https://openreview.net/forum?id=wH8XXUOUZU",
  },
  {
    name: "Selective Detail + DISTS",
    venue: "CVPRW 2021",
    focus: "分区细节解码与文字区域加权失真",
    relation: "四个参照中最直接面向图文和文字清晰度",
    gap: "需增加文字区域检测、DISTS 与专用细节解码器",
    reported: "论文公开了文字区域加权失真与 DISTS 优化结果；默认直接引用论文数据。",
    href: "https://openaccess.thecvf.com/content/CVPR2021W/CLIC/html/Suzuki_Learned_Image_Compression_With_Super-Resolution_Residual_Modules_and_DISTS_Optimization_CVPRW_2021_paper.html",
  },
  {
    name: "CompressAI Model Zoo",
    venue: "OFFICIAL PRETRAINED",
    focus: "可下载的学习式图像压缩预训练模型与统一评估工具",
    relation: "适合作为后续按需下载的可复现实测基线，无需重新训练",
    gap: "当前使用其基础 EntropyBottleneck；预训练模型仍作为按需下载基线",
    reported: "官方模型库支持 pretrained=True，并提供编码、解码与评估脚本。",
    href: "https://github.com/InterDigitalInc/CompressAI",
  },
] as const;

const STATUS_LABELS: Record<Status, string> = {
  queued: "排队中",
  running: "训练中",
  evaluating: "评估中",
  stopping: "停止中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已停止",
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = payload?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((item) => item.msg).join("；")
          : `请求失败（${response.status}）`;
    throw new Error(message);
  }
  return response.json();
}

export default function Home() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [device, setDevice] = useState("auto");
  const [environment, setEnvironment] = useState<Environment | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [result, setResult] = useState<ResultPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [sandboxOpen, setSandboxOpen] = useState(false);
    const eventSource = useRef<EventSource | null>(null);

  const refreshEnvironment = useCallback(async () => {
    try {
      setEnvironment(await api<Environment>("/api/environment"));
    } catch (reason) {
      setError(humanError(reason, "无法连接训练服务"));
    }
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await api<Job[]>("/api/experiments"));
    } catch {
      // The environment banner already communicates backend connectivity.
    }
  }, []);
  
  const refreshDefaults = useCallback(async () => {
    
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      refreshEnvironment();
      refreshJobs();
      refreshDefaults();
    }, 0);
    const timer = window.setInterval(() => {
      refreshEnvironment();
      refreshJobs();
    }, 3000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refreshEnvironment, refreshJobs, refreshDefaults]);

  useEffect(
    () => () => {
      eventSource.current?.close();
    },
    [],
  );

  const connectToJob = useCallback((jobId: string) => {
    eventSource.current?.close();
    const source = new EventSource(
      `${API_BASE}/api/experiments/${jobId}/events`,
    );
    eventSource.current = source;
    source.addEventListener("snapshot", async (event) => {
      const job = JSON.parse((event as MessageEvent).data) as Job;
      setActiveJob(job);
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      if (job.status === "completed" && job.has_result) {
        source.close();
        try {
          setResult(
            await api<ResultPayload>(`/api/experiments/${job.id}/result`),
          );
        } catch (reason) {
          setError(humanError(reason, "实验完成，但结果读取失败"));
        }
      } else if (job.status === "failed" || job.status === "cancelled") {
        source.close();
      }
    });
    source.onerror = () => {
      if (source.readyState === EventSource.CLOSED) {
        source.close();
      }
    };
  }, []);

  async function startExperiment(event: FormEvent) {
    event.preventDefault();
    setStarting(true);
    setError(null);
    setResult(null);
    try {
      const job = await api<Job>("/api/experiments", {
        method: "POST",
        body: JSON.stringify({ ...config, device }),
      });
      setActiveJob(job);
      setJobs((current) => [job, ...current]);
      connectToJob(job.id);
    } catch (reason) {
      setError(humanError(reason, "无法创建实验"));
    } finally {
      setStarting(false);
    }
  }

  async function stopExperiment() {
    if (!activeJob) return;
    try {
      const job = await api<Job>(`/api/experiments/${activeJob.id}/stop`, {
        method: "POST",
      });
      setActiveJob(job);
    } catch (reason) {
      setError(humanError(reason, "无法停止实验"));
    }
  }

  async function installDependencies() {
    setError(null);
    try {
      await api("/api/environment/install", { method: "POST" });
      await refreshEnvironment();
    } catch (reason) {
      setError(humanError(reason, "无法开始安装依赖"));
    }
  }

  async function selectJob(job: Job) {
    setActiveJob(job);
    setError(null);
    setResult(null);
    if (job.has_result) {
      try {
        setResult(await api<ResultPayload>(`/api/experiments/${job.id}/result`));
      } catch (reason) {
        setError(humanError(reason, "结果读取失败"));
      }
    } else if (["queued", "running", "evaluating", "stopping"].includes(job.status)) {
      connectToJob(job.id);
    }
  }

  const isBusy =
    activeJob &&
    ["queued", "running", "evaluating", "stopping"].includes(activeJob.status);

  function changeMethod(method: string) {
    setConfig({ ...config, method });
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">KAKEYA / EXPERIMENT OPERATIONS</p>
          <h1>图文压缩实验台</h1>
        </div>
        <div className="topbar-actions">
          <button
            type="button"
            className="sandbox-trigger"
            onClick={() => setSandboxOpen(true)}
          >
            模型试玩台
          </button>
          <EnvironmentControl
            environment={environment}
            onInstall={installDependencies}
          />
        </div>
      </header>

      <OperationStatusBoard
        environment={environment}
        activeJob={activeJob}
        starting={starting}
      />

      <ImagePreviewPanel
        selected={config.method === "image_codec"}
        onSelect={() => changeMethod("image_codec")}
      />

      {error && (
        <div className="error-banner" role="alert">
          <span>运行异常</span>
          <p>{error}</p>
          <button type="button" onClick={() => setError(null)}>
            关闭
          </button>
        </div>
      )}

      <section className="workspace">
        <form className="control-panel" onSubmit={startExperiment}>
          <div className="section-heading">
            <div>
              <p className="section-index">01</p>
              <h2>配置实验</h2>
            </div>
            <span className="method-code">{config.method}</span>
          </div>

          <label className="field field-wide">
            <span>训练模型</span>
            <select
              value={config.method}
              onChange={(event) => changeMethod(event.target.value)}
            >
              {Object.entries(METHOD_LABELS).map(([value, label]) => (
                <option value={value} key={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>

          <div className="field-grid">
            <NumberField
              label="训练轮次"
              value={config.epochs}
              min={1}
              max={500}
              step={1}
              onChange={(epochs) => setConfig({ ...config, epochs })}
            />
            <NumberField
              label={
                config.method === "image_codec"
                  ? "微调 Batch size"
                  : "Batch size"
              }
              value={config.batch_size}
              min={config.method === "image_codec" ? 1 : 8}
              max={2048}
              step={config.method === "image_codec" ? 1 : 8}
              onChange={(batch_size) => setConfig({ ...config, batch_size })}
            />
            <NumberField
              label={
                config.method === "image_codec"
                  ? "空间潜在通道"
                  : "潜在维度"
              }
              value={config.latent_dim}
              min={2}
              max={config.method === "image_codec" ? 32 : 256}
              step={1}
              onChange={(latent_dim) => setConfig({ ...config, latent_dim })}
            />
            <NumberField
              label="学习率"
              value={config.learning_rate}
              min={0}
              max={1}
              step={0.0001}
              onChange={(learning_rate) =>
                setConfig({ ...config, learning_rate })
              }
            />
            <NumberField
              label="随机种子"
              value={config.seed}
              min={0}
              max={2147483647}
              step={1}
              onChange={(seed) => setConfig({ ...config, seed })}
            />
            <NumberField
              label={
                config.method === "image_codec"
                  ? "微调训练卡片数"
                  : "训练样本上限（0=全部）"
              }
              value={config.train_limit}
              min={0}
              max={config.method === "image_codec" ? 4096 : 60000}
              step={config.method === "image_codec" ? 32 : 100}
              onChange={(train_limit) => setConfig({ ...config, train_limit })}
            />
            {config.method !== "image_codec" && (
              <NumberField
                label="测试样本上限（0=全部）"
                value={config.test_limit}
                min={0}
                max={10000}
                step={100}
                onChange={(test_limit) => setConfig({ ...config, test_limit })}
              />
            )}
            <label className="field">
              <span>计算设备</span>
              <select value={device} onChange={(event) => setDevice(event.target.value)}>
                <option value="auto">自动选择</option>
                <option value="cpu">CPU</option>
                <option value="mps">Apple MPS</option>
                <option value="cuda">CUDA</option>
              </select>
            </label>
          </div>

          <ObjectiveFields config={config} setConfig={setConfig} />

          <button
            className="primary-action"
            type="submit"
            disabled={Boolean(starting || isBusy || !environment?.ready)}
          >
            {starting ? "正在创建…" : isBusy ? "已有实验运行中" : "开始训练"}
          </button>
          {!environment?.ready && (
            <p className="field-note">环境就绪后才能开始训练。</p>
          )}
        </form>

        <section className="monitor-panel">
          <div className="section-heading">
            <div>
              <p className="section-index">02</p>
              <h2>训练监控</h2>
            </div>
            {activeJob && (
              <span className={`status status-${activeJob.status}`}>
                {STATUS_LABELS[activeJob.status]}
              </span>
            )}
          </div>

          {activeJob ? (
            <>
              <div className="run-meta">
                <div>
                  <span>实验</span>
                  <strong>
                    {METHOD_LABELS[activeJob.config.method] ??
                      activeJob.config.method}
                  </strong>
                </div>
                <div>
                  <span>轮次</span>
                  <strong>
                    {activeJob.epoch} / {activeJob.total_epochs}
                  </strong>
                </div>
                <div>
                  <span>阶段</span>
                  <strong>
                    {activeJob.series.length
                      ? (() => {
                          const last = activeJob.series[activeJob.series.length - 1];
                          const s = getStageFromPoint(last);
                          const map = {
                            capacity: "容量阶段",
                            transition: "过渡阶段",
                            finetune: "微调阶段",
                          };
                          return map[s];
                        })()
                      : "—"}
                  </strong>
                </div>
                <div>
                  <span>进度</span>
                  <strong>{Math.round(activeJob.progress * 100)}%</strong>
                </div>
                <div>
                  <span>设备</span>
                  <strong>{formatDevice(activeJob.device)}</strong>
                </div>
              </div>
              <div
                className="progress-track"
                role="progressbar"
                aria-valuenow={Math.round(activeJob.progress * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <span style={{ width: `${activeJob.progress * 100}%` }} />
              </div>
              <p className="current-message">{activeJob.message}</p>
              <LossChart series={activeJob.series} />
              <div className="console">
                <div className="console-heading">
                  <span>实时日志</span>
                  <code>{activeJob.id}</code>
                </div>
                <div className="console-lines" aria-live="polite">
                  {activeJob.logs.length ? (
                    activeJob.logs.slice(-12).map((line, index) => (
                      <p key={`${index}-${line}`}>
                        <span>{String(index + 1).padStart(2, "0")}</span>
                        {line}
                      </p>
                    ))
                  ) : (
                    <p className="console-empty">等待训练输出…</p>
                  )}
                </div>
              </div>
              {isBusy && (
                <button
                  className="stop-action"
                  type="button"
                  onClick={stopExperiment}
                  disabled={activeJob.status === "stopping"}
                >
                  强制停止训练
                </button>
              )}
              {activeJob.status === "failed" && (
                <div className="inline-error" role="alert">
                  <strong>训练失败</strong>
                  <p>{activeJob.error ?? "训练进程异常退出"}</p>
                </div>
              )}
            </>
          ) : (
            <EmptyMonitor />
          )}
        </section>
      </section>

      <section className="results-panel">
        <div className="section-heading">
          <div>
            <p className="section-index">03</p>
            <h2>实验结果</h2>
          </div>
          {result && <span className="result-ready">RESULT SET READY</span>}
        </div>
        {result ? (
          <Results
            result={result}
            jobId={activeJob?.id ?? ""}
          />
        ) : (
          <EmptyResults />
        )}
      </section>

      <section className="history-panel">
        <div className="section-heading">
          <div>
            <p className="section-index">04</p>
            <h2>本次服务记录</h2>
          </div>
          <span className="record-count">{jobs.length} RUNS</span>
        </div>
        <div className="history-list">
          {jobs.length ? (
            jobs.map((job) => (
              <button
                type="button"
                className={activeJob?.id === job.id ? "history-row active" : "history-row"}
                key={job.id}
                onClick={() => selectJob(job)}
              >
                <span className="history-method">
                  {METHOD_LABELS[job.config.method] ?? job.config.method}
                </span>
                <span>{job.config.epochs} epochs</span>
                <FormattedDate dateString={job.created_at} />
                <span className={`status status-${job.status}`}>
                  {STATUS_LABELS[job.status]}
                </span>
              </button>
            ))
          ) : (
            <p className="history-empty">尚未创建实验。</p>
          )}
        </div>
      </section>
      <SandboxPlayground
        open={sandboxOpen}
        onClose={() => setSandboxOpen(false)}
      />
    </main>
  );
}

function EnvironmentControl({
  environment,
  onInstall,
}: {
  environment: Environment | null;
  onInstall: () => void;
}) {
  const installing = environment?.install_status === "running";
  return (
    <div className="environment-control">
      <span
        className={
          environment?.ready ? "environment-dot ready" : "environment-dot"
        }
      />
      <div>
        <strong>
          {environment?.ready
            ? `Python ${environment.python} · ${environment.device.label}`
            : environment
              ? "依赖未完整安装"
              : "连接训练服务中"}
        </strong>
        <span>
          {installing
            ? environment?.install_logs.at(-1) ?? "正在安装依赖"
            : "本机隔离训练环境"}
        </span>
      </div>
      <button
        type="button"
        onClick={onInstall}
        disabled={installing || Boolean(environment?.ready)}
      >
        {installing
          ? "安装中…"
          : environment?.ready
            ? "安装成功"
            : "检查 / 安装"}
      </button>
    </div>
  );
}

function OperationStatusBoard({
  environment,
  activeJob,
  starting,
}: {
  environment: Environment | null;
  activeJob: Job | null;
  starting: boolean;
}) {
  const install = installState(environment);
  const training = trainingState(activeJob, starting);
  return (
    <section className="operation-status-board" aria-live="polite">
      <StatusCard
        index="01"
        title="环境安装"
        state={install.state}
        label={install.label}
        detail={install.detail}
      />
      <StatusCard
        index="02"
        title="模型训练"
        state={training.state}
        label={training.label}
        detail={training.detail}
        progress={training.progress}
      />
    </section>
  );
}

function StatusCard({
  index,
  title,
  state,
  label,
  detail,
  progress,
}: {
  index: string;
  title: string;
  state: "idle" | "working" | "success" | "error" | "stopped";
  label: string;
  detail: string;
  progress?: number;
}) {
  return (
    <article className={`operation-card operation-${state}`}>
      <span className="operation-index">{index}</span>
      <div className="operation-copy">
        <span>{title}</span>
        <strong>{label}</strong>
        <small>{detail}</small>
      </div>
      <span className="operation-symbol" aria-hidden="true">
        {state === "success"
          ? "✓"
          : state === "error"
            ? "!"
            : state === "stopped"
              ? "■"
              : state === "working"
                ? "●"
                : "○"}
      </span>
      {typeof progress === "number" && (
        <div className="operation-progress">
          <span style={{ width: `${Math.max(0, Math.min(progress, 1)) * 100}%` }} />
        </div>
      )}
    </article>
  );
}

function installState(environment: Environment | null): {
  state: "idle" | "working" | "success" | "error";
  label: string;
  detail: string;
} {
  if (!environment) {
    return { state: "working", label: "正在检查", detail: "连接本地训练服务" };
  }
  if (environment.install_status === "running") {
    return {
      state: "working",
      label: "正在安装",
      detail: environment.install_logs.at(-1) ?? "正在准备 Python 依赖",
    };
  }
  if (environment.install_status === "failed") {
    return {
      state: "error",
      label: "安装失败",
      detail: environment.install_error ?? "请查看错误提示后重试",
    };
  }
  if (environment.ready) {
    return {
      state: "success",
      label:
        environment.install_status === "completed" ? "安装成功" : "环境已安装",
      detail: `Python ${environment.python} · ${environment.device.label}`,
    };
  }
  return { state: "idle", label: "等待安装", detail: "点击右上角“检查 / 安装”" };
}

function trainingState(
  job: Job | null,
  starting: boolean,
): {
  state: "idle" | "working" | "success" | "error" | "stopped";
  label: string;
  detail: string;
  progress?: number;
} {
  if (starting) {
    return {
      state: "working",
      label: "正在创建任务",
      detail: "正在校验参数并启动训练进程",
      progress: 0,
    };
  }
  if (!job) {
    return { state: "idle", label: "尚未开始", detail: "配置参数后开始训练" };
  }
  if (job.status === "completed") {
    return {
      state: "success",
      label: "训练完成",
      detail: `${METHOD_LABELS[job.config.method] ?? job.config.method} · ${job.total_epochs} epochs`,
      progress: 1,
    };
  }
  if (job.status === "failed") {
    return {
      state: "error",
      label: "训练失败",
      detail: job.error ?? job.message,
      progress: job.progress,
    };
  }
  if (job.status === "cancelled") {
    return {
      state: "stopped",
      label: "训练已停止",
      detail: `停止于第 ${job.epoch}/${job.total_epochs} 轮`,
      progress: job.progress,
    };
  }
  const label =
    job.status === "queued"
      ? "等待训练"
      : job.status === "evaluating"
        ? "正在生成结果"
      : job.status === "stopping"
          ? "正在强制停止"
          : "正在训练";
  return {
    state: "working",
    label,
    detail: `${job.message} · ${Math.round(job.progress * 100)}%`,
    progress: job.progress,
  };
}

function ObjectiveFields({
  config,
  setConfig,
}: {
  config: ExperimentConfig;
  setConfig: (config: ExperimentConfig) => void;
}) {
  return (
    <div className="objective-box">
      <p>方法参数</p>
      <div className="field-grid">
        {config.method === "image_codec" && (
          <>
            <NumberField
              label="率失真 λ"
              value={config.lambda_rate}
              min={0.001}
              max={100}
              step={0.001}
              onChange={(lambda_rate) =>
                setConfig({ ...config, lambda_rate })
              }
            />
            <NumberField
              label="挂谷 λₖ"
              value={config.lambda_kakeya}
              min={0}
              max={10}
              step={0.001}
              onChange={(lambda_kakeya) =>
                setConfig({ ...config, lambda_kakeya })
              }
            />
            <NumberField
              label="随机投影数"
              value={config.num_projections}
              min={4}
              max={1024}
              step={4}
              onChange={(num_projections) =>
                setConfig({ ...config, num_projections })
              }
            />
            <NumberField
              label="Top-k 间距"
              value={config.k}
              min={1}
              max={4096}
              step={1}
              onChange={(k) => setConfig({ ...config, k })}
            />
          </>
        )}
      </div>
      {config.method === "image_codec" && (
        <StageWeightsEditor config={config} setConfig={setConfig} />
      )}
    </div>
  );
}

const STAGE_LOSS_KEYS = ["mse", "edge", "structural", "multiscale", "lab", "hue", "saturation", "kakeya"];

const STAGE_DEFAULTS: Record<string, Record<string, number>> = {
  capacity:  { mse: 1.0, edge: 1.0, structural: 0.2, multiscale: 0.2, lab: 0.05, hue: 0.05, saturation: 0.05, kakeya: 0.002 },
  transition: { mse: 2.0, edge: 1.5, structural: 0.4, multiscale: 0.3, lab: 0.08, hue: 0.08, saturation: 0.08, kakeya: 0.001 },
  finetune:   { mse: 3.0, edge: 2.0, structural: 0.6, multiscale: 0.4, lab: 0.12, hue: 0.06, saturation: 0.08, kakeya: 0.0005 },
};
function StageWeightsEditor({
  config, setConfig,
}: {
  config: ExperimentConfig;
  setConfig: (config: ExperimentConfig) => void;
}) {
  const stages = ["capacity", "transition", "finetune"] as const;
  const current = config.stage_weights ?? {};
  const update = (stage: string, key: string, value: number) => {
    const stageObj = { ...(current[stage] ?? {}) };
    stageObj[key] = value;
    setConfig({
      ...config,
      stage_weights: { ...current, [stage]: stageObj },
    });
  };
  const stepForKey = (key: string) =>
    key === "kakeya" ? 0.0005 : key === "mse" ? 0.5 : key === "structural" ? 0.05 : 0.01;
  return (
    <div className="stage-weights-editor">
      <p>分阶段损失权重</p>
      <table className="stage-weights-table">
        <thead>
          <tr>
            <th>阶段</th>
            {STAGE_LOSS_KEYS.map((k) => (
              <th key={k}>{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {stages.map((stage) => {
            const stageObj = { ...STAGE_DEFAULTS[stage], ...(current[stage] ?? {}) };
            return (
              <tr key={stage}>
                <td>{stage}</td>
                {STAGE_LOSS_KEYS.map((key) => (
                  <td key={key}>
                    <input
                      type="number"
                      step={stepForKey(key)}
                      min={0}
                      value={stageObj[key]}
                      onChange={(e) => update(stage, key, parseFloat(e.target.value) || 0)}
                    />
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      <small>所有损失方向从 capacity 阶段即启用，各阶段仅权重递增。留空使用默认值。</small>
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
        required
      />
    </label>
  );
}

function getStageFromPoint(point: SeriesPoint): "capacity" | "transition" | "finetune" {
  const train = point.train;
  if (train.capacity_stage >= 0.5) return "capacity";
  if (!train.capacity_gate_passed) return "capacity";
  const gateEpoch = train.capacity_gate_epoch ?? 0;
  const epoch = point.epoch;
  const epochsSinceGate = epoch - gateEpoch;
  if (epochsSinceGate <= 5) return "transition";
  return "finetune";
}

function getStageRanges(series: SeriesPoint[]): Array<{
  stage: "capacity" | "transition" | "finetune";
  startIndex: number;
  endIndex: number;
}> {
  if (!series.length) return [];
  const ranges: Array<{
    stage: "capacity" | "transition" | "finetune";
    startIndex: number;
    endIndex: number;
  }> = [];
  let currentStage = getStageFromPoint(series[0]);
  let startIndex = 0;
  for (let i = 1; i < series.length; i++) {
    const s = getStageFromPoint(series[i]);
    if (s !== currentStage) {
      ranges.push({ stage: currentStage, startIndex, endIndex: i - 1 });
      currentStage = s;
      startIndex = i;
    }
  }
  ranges.push({ stage: currentStage, startIndex, endIndex: series.length - 1 });
  return ranges;
}

function LossChart({ series }: { series: SeriesPoint[] }) {
  const hasGeneralization = series.some((point) =>
    Number.isFinite(point.validation.generalization_total),
  );
  const values = series.flatMap((point) => [
    point.train.total,
    point.validation.total,
    ...(Number.isFinite(point.validation.generalization_total)
      ? [point.validation.generalization_total]
      : []),
  ]);
  if (!values.length) {
    return <div className="chart-empty">首轮完成后显示训练与验证损失。</div>;
  }
  const width = 700;
  const height = 220;
  const padding = 28;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1e-6);
  const points = (key: "train" | "validation") =>
    series
      .map((point, index) => {
        const x =
          padding +
          (index / Math.max(series.length - 1, 1)) * (width - padding * 2);
        const y =
          height -
          padding -
          ((point[key].total - min) / range) * (height - padding * 2);
        return `${x},${y}`;
      })
      .join(" ");
  const generalizationPoints = series
    .map((point, index) => {
      const value = point.validation.generalization_total;
      if (!Number.isFinite(value)) return null;
      const x =
        padding +
        (index / Math.max(series.length - 1, 1)) * (width - padding * 2);
      const y =
        height -
        padding -
        ((value - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .filter(Boolean)
    .join(" ");
  return (
    <div className="chart-block">
      <div className="chart-heading">
        <span>总损失</span>
        <div className="legend">
          <span><i className="legend-train" />训练</span>
          <span><i className="legend-validation" />同阶段验证</span>
          {hasGeneralization && (
            <span><i className="legend-generalization" />独立图文</span>
          )}
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="训练和验证总损失曲线">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="axis" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="axis" />
        {getStageRanges(series).map((range, i) => {
          const xStart =
            padding +
            (range.startIndex / Math.max(series.length - 1, 1)) * (width - padding * 2);
          const xEnd =
            padding +
            (range.endIndex / Math.max(series.length - 1, 1)) * (width - padding * 2);
          const rectWidth = Math.max(xEnd - xStart, 1);
          const colorMap = {
            capacity: "rgba(187, 222, 251, 0.25)",
            transition: "rgba(255, 224, 178, 0.25)",
            finetune: "rgba(248, 187, 208, 0.25)",
          };
          const labelMap = {
            capacity: "容量阶段",
            transition: "过渡阶段",
            finetune: "微调阶段",
          };
          return (
            <g key={i}>
              <rect
                x={xStart}
                y={padding}
                width={rectWidth}
                height={height - padding * 2}
                fill={colorMap[range.stage]}
                rx={2}
              />
              <text
                x={(xStart + xEnd) / 2}
                y={padding + 14}
                textAnchor="middle"
                fontSize={11}
                fill="#555"
                style={{ pointerEvents: "none" }}
              >
                {labelMap[range.stage]}
              </text>
            </g>
          );
        })}
        <polyline points={points("train")} className="loss-line train-line" />
        <polyline points={points("validation")} className="loss-line validation-line" />
        {hasGeneralization && (
          <polyline
            points={generalizationPoints}
            className="loss-line generalization-line"
          />
        )}
        <text x={padding} y={18}>{max.toFixed(2)}</text>
        <text x={padding} y={height - 7}>{min.toFixed(2)}</text>
        <text x={width - padding} y={height - 7} textAnchor="end">epoch {series.at(-1)?.epoch}</text>
      </svg>
    </div>
  );
}

function Results({
  result,
  jobId,
}: {
  result: ResultPayload;
  jobId: string;
}) {
  if (result.config.method === "image_codec" && result.image_codec) {
    return <ImageCodecResults result={result} jobId={jobId} />;
  }
  return <EmptyResults />;
}

function ImagePreviewPanel({
  selected,
  onSelect,
}: {
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <section className="image-preview-panel">
      <div className="image-preview-heading">
        <p className="section-index">BUILT-IN TEST ASSET / ALWAYS VISIBLE</p>
        <h2>参考图文测试卡</h2>
      </div>
      <figure className="test-image-preview">
        <a
          href={`${API_BASE}/api/test-image`}
          target="_blank"
          rel="noreferrer"
          aria-label="打开测试卡原图"
        >
          {/* Dynamic local API assets should bypass framework image optimization. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`${API_BASE}/api/test-image`}
            alt="内置 Kakeya 参考图文测试卡"
            width={256}
            height={256}
          />
        </a>
        <figcaption>
          <span>256 × 256</span>
          <span>RGB / PNG</span>
          <span>点击查看原图</span>
        </figcaption>
      </figure>
      <div className="image-preview-copy">
        <span className="preview-kicker">Kakeya Codec Test Card v2</span>
        <strong>这张图片会在训练后自动执行压缩还原</strong>
        <p>
          同时覆盖中文、英文、小字号、真实街景、细线、网格、灰阶、色块和高对比边缘，
          用来直观看出文字模糊、纹理丢失与棋盘格伪影。
        </p>
        <div className="preview-tags" aria-label="测试内容">
          <span>中英文</span>
          <span>街景纹理</span>
          <span>1px 细线</span>
          <span>灰阶</span>
          <span>色彩</span>
        </div>
        <ol className="preview-flow">
          <li><span>01</span>选择图文模型</li>
          <li><span>02</span>完成训练</li>
          <li><span>03</span>查看原图 / 还原图 / 误差图</li>
        </ol>
        <button
          type="button"
          className={selected ? "preview-action selected" : "preview-action"}
          onClick={onSelect}
          disabled={selected}
        >
          {selected ? "已选择图文模型 ✓" : "使用这张图片开始图文实验"}
        </button>
      </div>
    </section>
  );
}

function SandboxPlayground({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [checkpointFile, setCheckpointFile] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [recon, setRecon] = useState<{
    original: string;
    reconstruction: string;
    error: string;
    metrics: { psnr: number; ssim: number; bpp: number; bitstream_bytes: number };
  } | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    if (!checkpointFile || !imageFile) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append("checkpoint", checkpointFile);
      form.append("image", imageFile);
      const res = await fetch(`${API_BASE}/api/reconstruct-custom`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "还原失败");
      }
      const data = await res.json();
      setRecon(data);
    } catch (err) {
      alert((err as Error).message || "图片还原失败，请检查模型和图片");
    } finally {
      setLoading(false);
    }
  };
  const panels = recon
    ? [
        ["original", "原图", recon.original],
        ["reconstruction", "还原图", recon.reconstruction],
        ["error", "误差热图", recon.error],
      ]
    : [];
  if (!open) return null;
  return (
    <div className="sandbox-overlay" onClick={onClose}>
      <div className="sandbox-panel" onClick={(e) => e.stopPropagation()}>
        <div className="sandbox-header">
          <div>
            <p className="section-index">SANDBOX · NO TRAINING NEEDED</p>
            <h2>模型试玩台</h2>
          </div>
          <button
            type="button"
            className="sandbox-close"
            onClick={onClose}
            aria-label="关闭"
          >
            ×
          </button>
        </div>
        <div className="sandbox-body">
          <p className="sandbox-intro">
            上传别人训练好的 Kakeya ImageCodecVAE checkpoint（.pt 文件）和一张测试图，
            立即看压缩还原效果。不需要训练，所有计算都在你本机完成。
          </p>
          <div className="sandbox-upload-grid">
            <label className="sandbox-upload-card">
              <span className="sandbox-card-label">模型文件</span>
              <strong>.pt / .pth / .ckpt</strong>
              <input
                type="file"
                accept=".pt,.pth,.ckpt"
                onChange={(e) => {
                  setCheckpointFile(e.target.files?.[0] || null);
                  setRecon(null);
                }}
              />
              <small>
                {checkpointFile ? checkpointFile.name : "点击选择文件"}
              </small>
            </label>
            <label className="sandbox-upload-card">
              <span className="sandbox-card-label">测试图片</span>
              <strong>任意图片格式</strong>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  setImageFile(e.target.files?.[0] || null);
                  setRecon(null);
                }}
              />
              <small>{imageFile ? imageFile.name : "点击选择图片"}</small>
            </label>
          </div>
          <button
            type="button"
            className="sandbox-run-btn-lg"
            onClick={handleRun}
            disabled={loading || !checkpointFile || !imageFile}
          >
            {loading ? "编码解码中…" : "开始压缩还原"}
          </button>
          {recon && (
            <>
              <div className="sandbox-result-header">
                <p className="section-index">RESULT</p>
                <h3>还原结果</h3>
              </div>
              <div className="codec-images sandbox-result-images">
                {panels.map(([kind, label, dataUrl]) => (
                  <figure key={kind}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={`data:image/png;base64,${dataUrl}`} alt={label} />
                    <figcaption>{label}</figcaption>
                  </figure>
                ))}
              </div>
              <div className="sandbox-result-metrics">
                <div className="metric">
                  <span>PSNR</span>
                  <strong>{recon.metrics.psnr.toFixed(2)} dB</strong>
                </div>
                <div className="metric">
                  <span>SSIM</span>
                  <strong>{recon.metrics.ssim.toFixed(4)}</strong>
                </div>
                <div className="metric">
                  <span>码流大小</span>
                  <strong>{formatBytes(recon.metrics.bitstream_bytes)}</strong>
                </div>
                <div className="metric">
                  <span>bpp</span>
                  <strong>{recon.metrics.bpp.toFixed(3)}</strong>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function ImageCodecResults({
  result,
  jobId,
}: {
  result: ResultPayload;
  jobId: string;
}) {
  const [regenerating, setRegenerating] = useState(false);
  const [regenerateKey, setRegenerateKey] = useState(0);
  const [regeneratedMetrics, setRegeneratedMetrics] = useState<Record<string, number> | null>(null);
  const [customRecon, setCustomRecon] = useState<{
    original: string;
    reconstruction: string;
    error: string;
    metrics: { psnr: number; ssim: number; bpp: number; bitstream_bytes: number };
  } | null>(null);
  const [customLoading, setCustomLoading] = useState(false);
  const m = regeneratedMetrics ?? result.metrics;
  const quality = assessImageQuality(
    Number(m.psnr ?? 0),
    Number(m.ssim ?? 0),
  );
  const training = result.image_codec?.training;
  const bitstream = result.image_codec?.bitstream;
  const latentShape = result.image_codec?.latent_shape ?? [
    result.config.latent_dim,
    32,
    32,
  ];
  const metrics = [
    ["当前结论", quality.usable ? "还原基本有效" : "当前还原无效"],
    ["肉眼质量", quality.label],
    ["结构保真", `${(Number(m.ssim ?? 0) * 100).toFixed(1)}%`],
    ["真实码流", bitstream ? formatBytes(bitstream.bytes) : "未生成"],
    ["潜在通道", String(result.config.latent_dim)],
  ] as const;
  const panels = [
    ["original", "原始测试图"],
    ["reconstruction", "模型还原图"],
    ["error", "误差热图"],
  ] as const;
  const hdPanels = [
    ["original_hd", "高清原图"],
    ["reconstruction_hd", "高清还原图"],
    ["error_hd", "高清误差热图"],
  ] as const;
  const hasHdImages = Boolean(
    result.image_codec?.images?.original_hd &&
      result.image_codec?.images?.reconstruction_hd &&
      result.image_codec?.images?.error_hd,
  );
  const hdMetrics = result.metrics as {
    hd_psnr?: number;
    hd_ssim?: number;
    hd_width?: number;
    hd_height?: number;
  };
  const handleRegenerate = async () => {
    setRegenerating(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/experiments/${jobId}/regenerate`,
        { method: "POST" }
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "重新生成失败");
      }
      const data = await res.json();
      setRegeneratedMetrics(data.metrics);
      setRegenerateKey((k) => k + 1);
    } catch (err) {
      alert((err as Error).message || "重新生成失败，请稍后重试");
    } finally {
      setRegenerating(false);
    }
  };

  const handleCustomUpload = async (file: File) => {
    setCustomLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(
        `${API_BASE}/api/experiments/${jobId}/reconstruct`,
        { method: "POST", body: form }
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "还原失败");
      }
      const data = await res.json();
      setCustomRecon(data);
    } catch (err) {
      alert((err as Error).message || "图片还原失败，请检查图片格式");
    } finally {
      setCustomLoading(false);
    }
  };
  const customPanels = customRecon
    ? [
        ["original", "你的原图", customRecon.original],
        ["reconstruction", "模型还原图", customRecon.reconstruction],
        ["error", "误差热图", customRecon.error],
      ]
    : [];
  return (
    <>
      <div className="metric-grid codec-metrics">
        {metrics.map(([label, value]) => (
          <div className="metric" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      {training && (
        <div
          className={`capacity-gate ${training.capacity_gate_passed ? "gate-passed" : "gate-failed"}`}
        >
          <div>
            <span>CAPACITY GATE</span>
            <strong>
              {training.capacity_gate_passed
                ? `第 ${training.capacity_gate_epoch} 轮通过`
                : "容量闸门未通过"}
            </strong>
          </div>
          <div className="capacity-gate-copy">
            <p>
              {training.capacity_gate_passed
                ? `模型先证明可以还原图文，随后进行了 ${training.compression_finetune_epochs} 轮压缩微调；最终采用第 ${training.selected_checkpoint_epoch ?? training.capacity_gate_epoch} 轮的质量优先 checkpoint。`
                : `本轮没有达到 PSNR ${training.capacity_gate.psnr.toFixed(0)}、结构保真 ${(training.capacity_gate.ssim * 100).toFixed(0)}% 的最低要求，因此全程保持确定性容量训练，未启用 KL 和 Kakeya。`}
            </p>
            {training.selected_checkpoint_epoch && (
              <small>
                选择条件：校准 PSNR 最高且估计码率不超过{" "}
                {training.target_rate_bpp.toFixed(1)} bpp；本轮选中
                checkpoint 约{" "}
                {training.selected_checkpoint_rate_bpp?.toFixed(3)} bpp。
              </small>
            )}
          </div>
        </div>
      )}
      {training && training.capacity_gate_passed && (
        <div className="training-stages">
          <p className="section-index">TRAINING STAGES</p>
          <div className="stage-bar">
            {(() => {
              const gateEpoch = training.capacity_gate_epoch ?? 0;
              const transitionEpochs = 5;
              const finetuneEpochs = Math.max(
                0,
                training.compression_finetune_epochs - transitionEpochs,
              );
              const total = gateEpoch + transitionEpochs + finetuneEpochs;
              const stages = [
                {
                  label: "容量阶段",
                  epochs: gateEpoch,
                  color: "#bbdefb",
                  desc: "方向覆盖 + 基础重建",
                },
                {
                  label: "过渡阶段",
                  epochs: transitionEpochs,
                  color: "#ffe0b2",
                  desc: "逐步引入感知损失",
                },
                {
                  label: "微调阶段",
                  epochs: finetuneEpochs,
                  color: "#f8bbd0",
                  desc: "码率 + 感知质量优化",
                },
              ];
              return stages.map((s, i) => {
                const width = total > 0 ? (s.epochs / total) * 100 : 0;
                return (
                  <div
                    key={i}
                    className="stage-segment"
                    style={{
                      width: `${width}%`,
                      backgroundColor: s.color,
                    }}
                  >
                    <span className="stage-label">{s.label}</span>
                    <span className="stage-epochs">{s.epochs} 轮</span>
                    <span className="stage-desc">{s.desc}</span>
                  </div>
                );
              });
            })()}
          </div>
        </div>
      )}
      <div className={`codec-verdict verdict-${quality.tone}`}>
        <strong>{quality.usable ? "本轮还原基本可用" : "本轮还原尚不可用"}</strong>
        <p>
          {quality.description}{" "}
          {bitstream
            ? `报告图由 ${formatBytes(bitstream.bytes)} 的真实量化码流解码得到。`
            : "本次旧结果没有可验证码流。"}{" "}
          这项判断同时参考结构相似度和实际图像，不要求理解 PSNR。
        </p>
      </div>
      <div className="regenerate-bar">
        <button
          type="button"
          className="regenerate-btn"
          onClick={handleRegenerate}
          disabled={regenerating}
        >
          {regenerating ? "重新生成中…" : "重新生成图文还原"}
        </button>
        <span className="regenerate-hint">
          重新用当前 checkpoint 跑 encode → decode，覆盖已保存的结果图和码流
        </span>
      </div>
      <div className="codec-summary">
        <div>
          <p className="section-index">IN-DISTRIBUTION CODEC CALIBRATION</p>
          <h3>图文还原结果 (多尺度)</h3>
        </div>
        <p>
          这是模型容量与复原能力测试，不是陌生图片泛化测试。模型使用{" "}
          {latentShape.join("×")} 空间潜变量。
        </p>
      </div>
      <div className="codec-images">
        {panels.map(([kind, label]) => (
          <figure key={kind}>
            {/* Dynamic experiment artifacts are intentionally loaded from the local API. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${API_BASE}/api/experiments/${jobId}/image/${kind}?t=${regenerateKey}`}
              alt={label}
            />
            <figcaption>{label}</figcaption>
          </figure>
        ))}
      </div>
      {hasHdImages && (
        <>
          <div className="codec-summary codec-summary-hd">
            <div>
              <p className="section-index">HIGH-RES GENERALIZATION</p>
              <h3>
                {hdMetrics.hd_width && hdMetrics.hd_height
                  ? `${hdMetrics.hd_width}×${hdMetrics.hd_height} 图文还原结果`
                  : "高清原图还原结果"}
              </h3>
            </div>
            <p>
              这是模型在 256² 之外的泛化能力测试，直接对 1024² 源图做 encode → decode，
              不经过 resize，反映真实大图压缩表现。
            </p>
          </div>
          <div className="codec-images">
            {hdPanels.map(([kind, label]) => (
              <figure key={kind}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`${API_BASE}/api/experiments/${jobId}/image/${kind}?t=${regenerateKey}`}
                  alt={label}
                />
                <figcaption>{label}</figcaption>
              </figure>
            ))}
          </div>
          {(hdMetrics.hd_psnr !== undefined || hdMetrics.hd_ssim !== undefined) && (
            <div className="custom-recon-metrics">
              {hdMetrics.hd_psnr !== undefined && (
                <span>
                  PSNR{" "}
                  <strong>{hdMetrics.hd_psnr.toFixed(2)} dB</strong>
                </span>
              )}
              {hdMetrics.hd_ssim !== undefined && (
                <span>
                  SSIM{" "}
                  <strong>{hdMetrics.hd_ssim.toFixed(4)}</strong>
                </span>
              )}
            </div>
          )}
        </>
      )}
      <div className="custom-recon-section">
        <div className="custom-recon-header">
          <div>
            <p className="section-index">TRY IT YOURSELF</p>
            <h3>上传你的图测试压缩</h3>
          </div>
          <label className="upload-btn">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleCustomUpload(file);
              }}
              disabled={customLoading}
            />
            {customLoading ? "处理中…" : "选择图片"}
          </label>
        </div>
        <p className="custom-recon-note">
          图片会自动缩放到 256×256，使用本次训练的 final.pt 做真实熵编码，
          结果只在你本机处理。也可以在左侧「模型试玩台」上传别人训练的模型。
        </p>
        {customRecon && (
          <>
            <div className="codec-images">
              {customPanels.map(([kind, label, dataUrl]) => (
                <figure key={kind}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`data:image/png;base64,${dataUrl}`} alt={label} />
                  <figcaption>{label}</figcaption>
                </figure>
              ))}
            </div>
            <div className="custom-recon-metrics">
              <span>
                PSNR <strong>{customRecon.metrics.psnr.toFixed(2)} dB</strong>
              </span>
              <span>
                SSIM <strong>{customRecon.metrics.ssim.toFixed(4)}</strong>
              </span>
              <span>
                码流{" "}
                <strong>
                  {formatBytes(customRecon.metrics.bitstream_bytes)}
                </strong>
              </span>
              <span>
                bpp <strong>{customRecon.metrics.bpp.toFixed(3)}</strong>
              </span>
            </div>
          </>
        )}
      </div>
      {bitstream && (
        <div className="bitstream-callout">
          <div>
            <span>REAL BITSTREAM</span>
            <strong>
              {bitstream.format} · {bitstream.bpp.toFixed(3)} bpp
            </strong>
            <p>
              码流包含格式头和量化潜变量；解码还需要本次训练的模型检查点。
              点击右侧下载 final.pt（可配合 <code>scripts/codec_cli.py</code> 压缩/解压图片）。
            </p>
          </div>
          <div className="bitstream-callout-actions">
            <button
              type="button"
              onClick={async () => {
                try {
                  await fetch(
                    `${API_BASE}/api/experiments/${jobId}/artifact/open-checkpoint-dir`,
                    { method: "POST" }
                  );
                } catch (err) {
                  console.error("打开目录失败", err);
                }
              }}
            >
              打开目录
            </button>
            <a
              href={`${API_BASE}/api/experiments/${jobId}/artifact/checkpoint`}
              download="final.pt"
            >
              下载 final.pt
            </a>
          </div>
        </div>
      )}
      <div className="result-charts codec-history">
        <HistoryChart result={result} />
      </div>
      <CodecBaselineComparison result={result} />
      <ImageCodecFrontier />
      <p className="benchmark-note">
        当前版本验证同分布图文重建质量；陌生图片泛化仍需要真实图文数据集。文件大小来自
        CompressAI EntropyBottleneck 生成的实际码流，不再用潜变量元素数估算。
      </p>
    </>
  );
}

function CodecBaselineComparison({ result }: { result: ResultPayload }) {
  const baselines = result.codec_baselines ?? [];
  const sourceBytes =
    baselines.find((item) => item.codec === "Original PNG")?.bytes ?? 0;
  const candidates = baselines.filter(
    (item) =>
      item.codec !== "Original PNG" &&
      assessImageQuality(item.psnr, item.ssim).usable,
  );
  const bestCandidate = candidates.reduce<(typeof candidates)[number] | null>(
    (best, item) => (!best || item.bytes < best.bytes ? item : best),
    null,
  );
  const modelQuality = assessImageQuality(
    Number(result.metrics.psnr ?? 0),
    Number(result.metrics.ssim ?? 0),
  );
  const bitstream = result.image_codec?.bitstream;
  const modelCompressionEffective = Boolean(
    bitstream && sourceBytes && bitstream.bytes < sourceBytes && modelQuality.usable,
  );
  const modelEffectTone = modelCompressionEffective
    ? "good"
    : modelQuality.tone === "warn"
      ? "warn"
      : "bad";
  const latentShape = result.image_codec?.latent_shape ?? [
    result.config.latent_dim,
    32,
    32,
  ];
  return (
    <div className="benchmark-section">
      <div className="benchmark-title">
        <div>
          <p className="section-index">SAME IMAGE / LOCAL MEASUREMENT</p>
          <h3>同一图文测试卡编码对比</h3>
        </div>
        <span>256×256 RGB · 先看结论，再看技术指标</span>
      </div>
      <div className="comparison-verdicts">
        <article className={modelCompressionEffective ? "is-good" : "is-bad"}>
          <span>当前 Kakeya VAE</span>
          <strong>
            {modelCompressionEffective
              ? "压缩与还原有效"
              : bitstream
                ? "已生成码流，但当前无效"
                : "旧结果没有码流"}
          </strong>
          <p>
            {modelQuality.description}
            {bitstream && sourceBytes
              ? `真实码流 ${formatBytes(bitstream.bytes)}，相对原始 PNG ${spaceSaving(bitstream.bytes, sourceBytes)}。只有画质达标且文件更小才计为有效。`
              : "需要用新版本重新训练后，才能同时判断画质和文件体积。"}
          </p>
        </article>
        <article className="is-good">
          <span>本机可用方案</span>
          <strong>{bestCandidate?.codec ?? "暂无"}</strong>
          <p>
            {bestCandidate && sourceBytes
              ? `${bestCandidate.settings}：${qualityLabel(bestCandidate.psnr, bestCandidate.ssim)}，文件比原始 PNG 小 ${spaceSaving(bestCandidate.bytes, sourceBytes)}。`
              : "当前结果中没有同时满足画质和体积要求的编码。"}
          </p>
        </article>
      </div>
      <div className="benchmark-table-wrap">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>编码方式</th>
              <th>是否有效</th>
              <th>文件大小 ↓</th>
              <th>节省空间</th>
              <th>画质说明</th>
              <th>技术指标</th>
            </tr>
          </thead>
          <tbody>
            {baselines.map((item) => {
              const quality = assessImageQuality(item.psnr, item.ssim);
              const isSource = item.codec === "Original PNG";
              return (
                <tr key={`${item.codec}-${item.settings}`}>
                  <td>
                    {item.codec}
                    <small>{item.settings}</small>
                  </td>
                  <td>
                    <span className={`effect-badge effect-${isSource ? "reference" : quality.tone}`}>
                      {isSource
                        ? "原图参照"
                        : quality.usable && item.bytes < sourceBytes
                          ? "有效"
                          : quality.tone === "warn"
                            ? "有限"
                            : "无效"}
                    </span>
                  </td>
                  <td>{formatBytes(item.bytes)}</td>
                  <td>{isSource ? "—" : spaceSaving(item.bytes, sourceBytes)}</td>
                  <td>{quality.label}</td>
                  <td>
                    {item.psnr >= 99
                      ? "无损"
                      : `PSNR ${item.psnr.toFixed(1)} · SSIM ${(item.ssim * 100).toFixed(1)}%`}
                  </td>
                </tr>
              );
            })}
            <tr className="highlight-row">
              <td>
                图文 Kakeya VAE
                <small>
                  Hyperprior v5 · 条件高斯 · {latentShape.join("×")}
                </small>
              </td>
              <td>
                <span className={`effect-badge effect-${modelEffectTone}`}>
                  {modelCompressionEffective
                    ? "有效"
                    : bitstream
                      ? modelQuality.usable
                        ? "体积无效"
                        : "画质无效"
                      : "无真实码流"}
                </span>
              </td>
              <td>{bitstream ? formatBytes(bitstream.bytes) : "—"}</td>
              <td>
                {bitstream
                  ? spaceSaving(bitstream.bytes, sourceBytes)
                  : "无法计算"}
              </td>
              <td>{modelQuality.label}</td>
              <td>
                PSNR {Number(result.metrics.psnr ?? 0).toFixed(1)} · SSIM{" "}
                {(Number(result.metrics.ssim ?? 0) * 100).toFixed(1)}%
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="benchmark-note">
        通俗理解：PSNR 越高越接近原图，但它不是百分比；大约 30 dB 以上才适合继续观察，
        低于 25 dB 通常已有明显失真。SSIM 已换算为结构保真百分比，越接近 100% 越好。
        图文还必须以原图、码流解码图和误差图的实际文字清晰度为准。.kky 大小包含文件头与
        熵编码负载，但不重复计入双方都需要的编解码器程序；该文件需配合本轮检查点解码。
      </p>
    </div>
  );
}

function ImageCodecFrontier() {
  return (
    <div className="benchmark-section frontier-section">
      <div className="benchmark-title">
        <div>
          <p className="section-index">IMAGE COMPRESSION FRONTIER</p>
          <h3>图文压缩前沿能力参照</h3>
        </div>
        <span>只比较图像编码能力，不再引用 MNIST 潜空间指标</span>
      </div>
      <div className="frontier-grid">
        {IMAGE_CODEC_REFERENCES.map((reference) => (
          <article className="frontier-card" key={reference.name}>
            <div>
              <span>{reference.venue}</span>
              <a href={reference.href} target="_blank" rel="noreferrer">
                {reference.name} ↗
              </a>
            </div>
            <strong>{reference.focus}</strong>
            <p>{reference.relation}</p>
            <em>{reference.reported}</em>
            <small>{reference.gap}</small>
          </article>
        ))}
      </div>
      <p className="benchmark-note">
        “论文报告值”与“本机同图实测”严格分开。外部模型优先使用官方预训练权重或
        论文公开指标，不占用本机训练资源；不同数据集和硬件的数字不参与同一排名。
      </p>
    </div>
  );
}

function HistoryChart({ result }: { result: ResultPayload }) {
  const series: SeriesPoint[] = result.history.epoch.map((epoch, index) => ({
    epoch,
    train: { total: result.history.train.total[index] },
    validation: {
      total: result.history.validation.total[index],
      generalization_total:
        result.history.validation.generalization_total?.[index] ?? Number.NaN,
    },
  }));
  return <LossChart series={series} />;
}

function EmptyMonitor() {
  return (
    <div className="empty-state">
      <span>WAITING FOR RUN</span>
      <strong>配置参数并开始训练</strong>
      <p>训练轮次、损失和进程日志将在这里实时更新。</p>
    </div>
  );
}

function EmptyResults() {
  return (
    <div className="empty-results">
      <div />
      <p>完成训练后，这里会展示原图、还原图、误差图和图文编码对比。</p>
      <div />
    </div>
  );
}

function humanError(reason: unknown, fallback: string) {
  return reason instanceof Error ? reason.message : fallback;
}

function assessImageQuality(psnr: number, ssim: number) {
  if (psnr >= 60 && ssim >= 0.999) {
    return {
      label: "无损，与原图一致",
      description: "像素和结构几乎完全保留。",
      usable: true,
      tone: "good",
    } as const;
  }
  if (psnr >= 35 && ssim >= 0.95) {
    return {
      label: "清晰，肉眼差异很小",
      description: "文字、线条和整体结构保留良好。",
      usable: true,
      tone: "good",
    } as const;
  }
  if (psnr >= 30 && ssim >= 0.9) {
    return {
      label: "基本可用，有轻微损失",
      description: "整体仍可辨认，但细小文字和边缘可能变软。",
      usable: true,
      tone: "good",
    } as const;
  }
  if (psnr >= 25 && ssim >= 0.75) {
    return {
      label: "效果有限，仅适合预览",
      description: "能辨认主要内容，但图文细节已有明显损失。",
      usable: false,
      tone: "warn",
    } as const;
  }
  return {
    label: "失真明显，不适合图文",
    description: "文字、细线或结构与原图差异过大。",
    usable: false,
    tone: "bad",
  } as const;
}

function qualityLabel(psnr: number, ssim: number) {
  return assessImageQuality(psnr, ssim).label;
}

function spaceSaving(bytes: number, sourceBytes: number) {
  if (!sourceBytes) return "无法计算";
  const percent = (1 - bytes / sourceBytes) * 100;
  return percent >= 0
    ? `${percent.toFixed(1)}%`
    : `增大 ${Math.abs(percent).toFixed(1)}%`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function formatDevice(device: string) {
  if (device === "mps" || device === "auto") return "Apple MPS";
  if (device === "cuda") return "CUDA";
  if (device === "saved") return "已保存";
  return device.toUpperCase();
}
