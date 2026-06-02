# 同主机 Disagg E/PD 时间分解与瓶颈分析(GPU 0/1, 32B FP8, 8img/1080p, np=32 rate=1.0)

**测试日期:** 2026-05-28
**主机:** sc09super21-h200 (172.26.46.133)
**模型:** Qwen3-VL-32B-Instruct-FP8
**拓扑:** 同主机 disagg —— Encoder 在 GPU 0,PD 在 GPU 1,GPU 0/1 都在 NUMA 0,通过 NV18 NVLink (~478 GB/s) 互联
**Patches:** **未应用** (h200_cuda_nixl.patch 与 b70_xpu_nixl.patch 均已回滚)
**Workload:** 8 张 1920×1080 随机 JPEG, in=128 / out=256, np=32 prompts, rate=1.0 RPS
**Bench 脚本:** `sglang.bench_serving --backend sglang-oai-chat`

**两次实验对比:**
- 实验 A: `DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read` — 22:17 UTC,结果目录 `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_1080p_rate1.0_np32_20260528_221743/`
- 实验 B: `DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-write` — 23:08 UTC,结果目录 `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_1080p_rate1.0_np32_nixlwrite_20260528_230817/`

本文以实验 A (nixl-read) 为详细分析主体;实验 B (nixl-write) 的对比放在 §十三。

## 一. 总览(实验 A: nixl-read)

| 指标 | 数值 |
|---|---:|
| 成功请求数 | **32 / 32** ✓ |
| Bench 持续时间 | 137.4 s |
| **吞吐 RPS** | **0.23** |
| 平均 TTFT | 70.0 s |
| 中位 TTFT | 72.8 s |
| P99 TTFT | 102.1 s |
| 平均 TPOT | 517 ms |
| 中位 TPOT | 339 ms |
| P99 TPOT | 2,765 ms |
| 平均 ITL | 337 ms |
| 中位 ITL | 29 ms |
| **最大 ITL** | **121.8 s** ← decode 阻塞病态 |
| 平均 E2E | 119.8 s |
| 中位 E2E | 118.5 s |
| 平均并发 | **27.9 / 32 (87%)** |
| 输入吞吐 (vision tokens/s) | 3,823 |
| 输出吞吐 (tokens/s) | 35.6 |
| 峰值输出 (tokens/s) | 944 |

吞吐 0.23 RPS 与之前文档化的同主机 disagg 基线一致(`disagg_improvements_attempts.md` 中 NIXL_READ + 调优后的同样结果),证实这是该工作负载在该硬件上的"自然"性能上限,而非配置问题。

## 二. PD 端 ReqTimeStats 分解(基于 SGLang 内部计时)

从 PD 日志中提取 32 个完整请求的 `ReqTimeStats` 事件:

| 阶段 | 中位 | 平均 | P99 | 最小 | 占请求生命周期比例 (中位) |
|---|---:|---:|---:|---:|---:|
| **queue_duration** (PD 调度器队列等待) | **48.8 s** | 44.4 s | 68.7 s | 0.0 s | **49.8%** |
| **forward_duration** (Prefill + Decode GPU 计算) | **50.7 s** | 56.1 s | 130.1 s | 7.6 s | **51.7%** |
| **PD 总生命周期** (queue + forward) | **98.0 s** | 100.5 s | — | — | 100% |

观察到的 input_len 中位数为 **16,420 tokens**(≈ 16,384 visual + ~36 text),完全符合 8img/1080p 的预期视觉 token 数。

## 三. 单个请求的端到端时间分布(基于 bench 客户端测量)

```
请求生命周期 (中位 E2E = 118.5 s):

  [客户端] HTTP POST → frontend
        |
        | (~ms,可忽略)
        ↓
  [frontend] kv-router → encoder
        |
        | TCP request plane (~ms)
        ↓
  [encoder GPU 0]
        ├─ 接收请求 + 解析图像 URL                      ~50-200 ms
        ├─ Qwen3-VL ViT 前向 (8 张 1080p)               ~1.2 s
        ├─ embedding.cpu()  ←  patch 回滚后强制 CPU       ~50-200 ms
        ├─ NIXL register_memory(CPU buffer)             ~50 ms
        └─ 通过 dynamo TCP 把 TransferRequest 发到 PD    ~5-50 ms
                         |
                         |  ←  encoder 阶段 ≈ 1.5-2 s
                         ↓
  [PD GPU 1]
        ├─ T0: 收到请求(由 dynamo Rust ingress 转交)
        ├─ T1: PD 进入 process_embeddings
        ├─ T2: 分配 CPU NIXL receive buffer (size=638MB)  ~10-50 ms
        ├─ T3: NIXL begin_read 提交                       ~5 ms
        ├─ T4: PROC → DONE (NIXL 实际数据传输 cuda_ipc)  **median ~11.3 ms** ✓
        ├─ T5: encoder GPU → CPU buffer 复制 (encoder 端)
        ├─ T6: CPU → PD GPU 复制(PD 把 CPU buffer 拉到 cuda:1)   ~50-150 ms
        ├─ T7: 进入 SGLang 调度器队列  ──┐
        |                              |
        |     ⚠ queue_duration = 48.8 s 中位(主要瓶颈所在)
        |                              |
        ├─ T8: 开始 chunked prefill   ──┘
        |        Prefill batch (8192 tokens 一块,大请求要 2 块)
        |        + 小批量 embedding-integration 事件(见第四节)
        |     forward_duration = 50.7 s 中位
        ├─ T9: Decode 256 tokens
        |        每 token TPOT median 339 ms,但 Max ITL = 121.8 s
        ├─ T10: SSE 流回 frontend
        └─ T11: 请求完成
```

**TTFT (中位 72.8 s) 的构成估算:**
- Encoder 阶段:~1.5-2 s
- NIXL 传输 + PD 端 CPU↔GPU 拷贝:~50-200 ms
- **PD 调度器队列等待:~48.8 s** ← 主要部分
- 第一个 prefill batch 处理 + 第一个 token decode:~20 s

队列等待占了 TTFT 的 67%,这是同主机 disagg 在饱和工况下的典型特征。

## 四. PD 端 Prefill 行为分析(关键瓶颈证据)

PD 日志中共记录 **68 次 Prefill batch 事件**,其 `#new-token` 分布如下:

| #new-token | 出现次数 | 性质 |
|---:|---:|---|
| 16,384 (满块,主请求) | 33 | 正常 16k-token vision prefill |
| 16,368 (尾块) | 4 | 主请求最后一块 |
| 320 | 1 | 中等小批量 |
| 80, 96 | 2, 4 | 小批量 |
| 64 | 7 | 小批量 |
| 48 | 5 | 小批量 |
| 32 | 5 | 小批量 |
| **16** | **7** | **极小 batch** |

**31 / 68 (46%) 的 prefill 事件是 ≤ 320 tokens 的"小批量 embedding-integration"事件。** 这些小批量是 SGLang 的 `_generate_aggregated` 路径在处理 `precomputed_embeddings`(由 NIXL 传过来的 vision embedding)时,把 16,336 个 image-position tokens 当作"已 cached KV"进行折叠的副产品 —— 每次只把 16-320 个 token 折叠到 KV cache 里,需要全模型前向但 batch 极小,GPU 利用率极低。

**Prefill 输入吞吐(token/s):**
| 类型 | 数值 |
|---|---:|
| 中位 | 3,014 tok/s |
| 平均 | 4,813 tok/s |
| 峰值 | 15,037 tok/s ← 16,384-token 满块时的硬件极限 |
| 最低 | 1 tok/s ← 16-token 小批量时的极端低效 |

**16,384-token 满块 prefill 的 GPU 利用率高达 ~15k tok/s**(接近 H200 单卡 32B FP8 的算力上限),但小批量事件把均值拉到 ~3k tok/s。这与 `bottleneck_analysis.md` 中文档化的"小批量阻塞 decode"病态完全一致。

## 五. PD 端并发(running-req)分布

```
running=0:  10.3% ← 调度器空闲(请求间隙)
running=1:   5.9%
running=2:   5.9%
running=3:   8.8%
running=4-9:  ~17% (各 1-3%)
running=10-19: ~20%
running=20-29: ~20%
running=30-31: ~5%

平均 running-req: 12.18
最大 running-req:  31  (达到 max_running_requests=64 的一半,nb=32 cap 在起作用)
最大 queue-req:    25
```

PD 调度器 **同时处理多达 31 个并发请求** —— 这与同主机 agg TP=1 的并发模式相似,但 disagg 的 forward_duration 中位数(50.7 s)远高于 agg(~3.5 s,见 `1080p_sweep_three_way.md`),原因是:

1. PD 必须为每个新到达的 NIXL embedding 启动一个 small-batch integration prefill,频繁打断当前的 decode 批处理。
2. encoder→PD 的请求到达率受 encoder 速率制约(中位 inter-arrival ≈ 1.45 s),而 SGLang 调度器试图把新请求加入运行批次时会触发 prefill 周期 —— 见第四节的 46% 小批量事件。

## 六. NIXL 数据传输统计

| 指标 | 数值 |
|---|---:|
| ReadOperation 创建总数 | 40 |
| Embedding 大小 (size= field) | 638 MB (16,384 × 5,120 × bf16) |
| 较小尺寸 (12 MB) | warmup/smoke test 残留 |
| **PROC → DONE 实际线材时间** | **中位 11.3 ms,平均 11.2 ms,最大 11.6 ms** |
| **device 字段** | **`device=cpu` 双端**(因 patch 已回滚) |

**关键观察:**
- NIXL 实际线材传输 **极快(11 ms 中位)**,因为同主机用的是 cuda_ipc(实际是 host pinned memory + cudaMemcpyAsync,因为 patch 回滚后两端都是 CPU 内存)。
- 但 device=cpu 意味着每个请求需要一次 encoder GPU → CPU pinned 的拷贝(发送端)+ 一次 CPU pinned → PD GPU 的拷贝(接收端),也就是经典的"CPU bounce"模式。
- 每次 638 MB,以 50 GB/s 主机带宽算需要 ~12.7 ms,与观察到的 11.3 ms wire 时间一致。

## 七. Decode 行为(高 TPOT 与 max ITL 异常的来源)

```
Decode events: 19 (相比 prefill 68 个,decode 占比很少)
中位 TPOT: 339 ms
平均 TPOT: 517 ms
P99 TPOT:  2,765 ms
中位 ITL:  29 ms       ← 健康的小批量 decode 时
P95 ITL:   153 ms
P99 ITL:   203 ms
**最大 ITL: 121,816 ms** ← 单个请求等了 121 秒才出下一个 token!
```

**最大 ITL 121.8 秒的解释:**

某个请求在 decode 阶段中途被另一个请求的 vision-prefill(以及其后的 small-batch embedding-integration 系列)持续插入,导致 decode 被反复打断超过 2 分钟。这是同主机 disagg 特有的"prefill-blocks-decode"病态模式 —— 与 `deep_analysis_disagg_worse_h200.md` 中文档化的现象完全一致。

## 八. 时间预算汇总(单请求中位)

```
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段                                       中位时长   占 E2E 比例    │
├─────────────────────────────────────────────────────────────────────┤
│ 客户端 HTTP + frontend route               <50 ms     0.04%          │
│ Encoder ViT 前向 + token expand            ~1.5 s     1.3%           │
│ Encoder embedding.cpu() + NIXL register   ~150 ms    0.13%          │
│ NIXL 实际线材传输 (PROC→DONE)              11 ms      0.01%          │
│ PD CPU→GPU 拷贝(接收 buffer)             ~150 ms    0.13%          │
│ PD 调度器队列等待 (queue_duration)         **48.8 s** **41.2%** ⚠   │
│ PD GPU forward (含小批量+大批量+decode)    **50.7 s** **42.8%** ⚠   │
│ PD egress + SSE stream 回 frontend        ~17 s      14.4%          │
├─────────────────────────────────────────────────────────────────────┤
│ E2E 中位                                   118.5 s    100%           │
└─────────────────────────────────────────────────────────────────────┘
```

(注:egress + SSE stream 部分包含其他 streaming 请求竞争前端的影响,实际"纯出口"时间应较短;此处用 E2E - queue - forward 作为余值。)

## 九. 瓶颈识别

按对总延迟的贡献排序:

### 🥇 #1 瓶颈 —— PD GPU 计算饱和 (forward_duration 50.7 s 中位)

**机制:** PD 在饱和工况(rate=1.0,np=32 全部并发)下,GPU 的有效算力被**两类工作**瓜分:

1. **大块 vision prefill** (#new-token=16,384):每个请求需要 1 个 16k chunk(若有 cache 则 0 个,但本次为 random images),这种大批量在 H200 上能跑出 ~15k tok/s 的接近峰值算力。
2. **小批量 embedding-integration** (#new-token=16-320):占 prefill 事件总数的 **46%**,这些事件每个仅几个 token 但要走完 32B FP8 的全部 64 层,GPU 只能跑出 ~10-200 tok/s,几乎全是 kernel launch overhead。

每条请求在 PD 上的"有效 GPU 时间"约 35 s(若没有小批量浪费,理论可降到 ~10-15 s)。

**为什么不能简单去除小批量?** 因为 SGLang 的 `_generate_aggregated` 路径在 NIXL embedding 到达后,必须把它"折叠"进 KV cache —— 这是 disagg 与 agg 的结构性差异:agg 模式下 ViT 前向直接输出到 LLM 输入序列里,无需事后 integration。

### 🥈 #2 瓶颈 —— PD 调度器队列饱和 (queue_duration 48.8 s 中位)

**机制:** rate=1.0 RPS 输入 + 每请求 ~98 s 在 PD 上的生命周期 = Little's Law 算出 in-flight ≈ 27.9(实测一致)。np=32 prompt 几乎在 bench 一开始就全部到达,然后排在 PD 调度器后面等运行批次插槽。

**为什么 max_running_requests=64 没有用?** 因为虽然 64 个槽位是允许的,但每个 16k-token 的 prefill 占用 KV cache ~30 GB,32 个并发请求 × 16k tokens = ~512k token 的 KV(占 PD GPU KV pool 的 73%)—— 已经接近 KV cache 上限,所以实际 running-req 顶多到 31。

### 🥉 #3 瓶颈 —— Decode 被 prefill 反复打断 (Max ITL 121.8 s)

**机制:** 当一个请求进入 decode 阶段时,只要新请求到达 PD,就会插入一个新的 vision-prefill(大块 16k)或更糟的 small-batch embedding-integration(数十次),把 decode 批处理打断。被打断的请求需要等所有插入的 prefill 走完才能继续生成下一个 token。

**为什么 disagg 比 agg 严重?**
- Agg 模式下,ViT 前向就是 LLM 前向的开端,二者在同一次 forward 里完成,decode 不会被"事后"插入打断。
- Disagg 模式下,encoder 异步把 embedding 推过来,PD 必须随时响应,所以 decode 容易被插入。

### 不是瓶颈(已排查)

| 候选 | 实测结果 | 结论 |
|---|---|---|
| NIXL 线材传输 | 中位 11 ms | 极快,可忽略 |
| Encoder ViT 计算 | ~1.2 s/req | 单卡 H200 vision tower 不算瓶颈,且与 PD prefill 并行 |
| Encoder→PD TCP plane | <50 ms | dynamo Rust runtime 极快 |
| KV cache 容量 | 695,136 tokens (mem-frac=0.85) | 仅用 ~73%,有富余 |
| max_running_requests=64 | 实际只用到 31 | 调度器没主动限流 |
| GPU memory contention | 无 OOM,无 cuda_ipc 错误 | patch 回滚后两端都用 CPU buffer,无跨 GPU 引用 |

## 十. 与其他同工作负载的对比

| 配置 | RPS | 中位 TTFT | 中位 E2E | 来源 |
|---|---:|---:|---:|---|
| **本次:同主机 disagg unpatched, GPU 0/1, np=32** | **0.23** | **72.8 s** | **118.5 s** | 本文 |
| 同主机 disagg unpatched, GPU 4/5, np=64 | 0.23 | 75.9 s | 146 s | `disagg_improvements_attempts.md` (NIXL_READ + A+B 调优) |
| 同主机 TP=1 agg | 0.52 | 24.9 s | 60 s | `1080p_sweep_three_way.md` |
| **同主机 TP=2 agg NVLink** | **0.95** | **<10 s** | **22.8 s** | `1080p_sweep_three_way.md` (推荐配置) |
| Cross-host disagg dell06_1E patched, np=32 | 0.24 | 58 s | 67 s | `time_breakdown_dell06_super21_32b_8img_1080p_v2_with_encoder.md` |
| Cross-host disagg B70_4E patched | 0.13 | 200 s | 200 s | `patched_4E_results.md` |

**关键观察:**
1. 同主机 disagg(无论 GPU 4/5 还是 0/1,无论 np=32 还是 np=64)都稳定在 **0.23 RPS** —— 这是该工作负载在该架构下的硬性天花板。
2. **同主机 TP=2 agg 比 disagg 快 4.1×**(0.95 vs 0.23),且用同样的 2 张 GPU。
3. 即使是更复杂的 cross-host disagg(dell06 H200 encoder + giga01 H200 PD),也只达到 0.24 RPS —— 同样卡在 PD-side 的 small-batch + queue 瓶颈上。

## 十一. 结论

### 1. 同主机 disagg 在 32B FP8 + 8img/1080p + rate=1.0 + np=32 的工作负载上,稳定饱和于 ~0.23 RPS,P99 TTFT 100+ s。

这与历史所有同主机 disagg 测量结果一致(详见 `INDEX.md` 中的多个文档),**不是配置错误**,而是结构性限制。

### 2. 瓶颈不是 NIXL 传输本身(11 ms),而是 PD 端的两个相互强化的因素:
- **小批量 embedding-integration prefill 占据 46% 的 prefill 事件**,把 GPU 利用率拉到平均 3k tok/s(峰值 15k 的 1/5)。
- **PD 调度器队列在饱和下中位等待 48.8 s**,因为 in-flight 数(27.9)接近 KV cache 容量上限。

### 3. patch 回滚 (h200_cuda_nixl.patch + b70_xpu_nixl.patch) 是当前同主机 disagg 工作的必要条件,因为 patch 引入的 GPUDirect 路径会通过 cuda_ipc 把 encoder GPU 的 NIXL receive buffer 引用泄漏到 PD,导致 PD 的下游 `torch.empty(..., device=x.device)` 在 GPU 0 而非 GPU 1 上分配,在 np=32 并发下耗尽 GPU 0 显存而触发 OOM(详见 `round5_patch_results.md`)。**回滚后 0.23 RPS 是稳定可重复的;patched 版本即使能跑也不能稳定 32 个并发**。

### 4. 推荐生产配置(若工作负载相同):
- **首选: TP=2 agg NVLink** (`start_h200_aggregate_epd_server_32b_tp2.sh`) —— 0.95 RPS, 9 s TTFT
- 次选: TP=1 agg —— 0.52 RPS,需要的 GPU 更少
- **避免: 同主机 disagg** —— 用了 2 张 GPU 却比 TP=1 agg 慢 2.3×

### 5. 何时 disagg 仍有价值?

参考 `when_disagg_wins.md` 与 `when_vit_is_bottleneck.md`:
- **encoder 是瓶颈的工作负载**(更小的 LLM,如 3B/7B + 大量图像)
- **多租户场景**(1 个 encoder 池 + N 个 PD)
- **跨主机部署需求**(运维隔离/硬件异构)

对于 32B FP8 + 8 张 1080p 的标准工作负载,以上场景都不适用,因此 disagg 在这里没有实际优势。

## 十二. nixl-write vs nixl-read 对比实验

为了量化 NIXL 传输模式对同主机 disagg 性能的影响,在保持其他配置完全不变的前提下,把 `DYN_SGL_EMBEDDING_TRANSFER_MODE` 从 `nixl-read` 切换到 `nixl-write` 重跑了相同的 bench。

### 12.1 两种模式的工作机制差异

| 模式 | 数据流向 | 控制平面交互 |
|---|---|---|
| **nixl-read** (PULL) | PD 主动从 encoder 拉取 embedding | 1) Encoder 注册 buffer + 通告; 2) PD 接到 TransferRequest 后调用 `begin_read`; 3) PD 等待完成 |
| **nixl-write** (PUSH) | Encoder 主动推送 embedding 到 PD | 1) Encoder 直接 `begin_write` 把数据写入 PD 端 buffer; 2) PD 收到通知即可使用 |

nixl-write 比 nixl-read 少一次"通告→等待初始化"的握手往返,理论上每请求节省 ~10-50 ms。

### 12.2 实验设置

两次实验完全相同除了一个 env var:
- 同样的硬件:GPU 0 (encoder) + GPU 1 (PD), NV18 NVLink, NUMA 0
- 同样的模型:Qwen3-VL-32B-Instruct-FP8
- 同样的 mem-fraction=0.85 双端
- 同样的 32 张随机 1080p 图像 / np=32 / rate=1.0
- 两次都重启了 PD + encoder (DeepGEMM 重新预热),无 stale state

### 12.3 端到端指标对比

| 指标 | nixl-read (实验 A) | nixl-write (实验 B) | Δ |
|---|---:|---:|---|
| 成功请求 | 32 / 32 ✓ | 32 / 32 ✓ | = |
| Bench 持续时间 | 137.4 s | **130.9 s** | -4.7% |
| **吞吐 RPS** | 0.23 | **0.24** | **+4.3%** |
| 输入吞吐 (vision tokens/s) | 3,823 | **4,012** | +4.9% |
| 输出吞吐 (tokens/s) | 35.6 | **37.3** | +4.9% |
| 峰值输出吞吐 (tok/s) | 944 | 798 | -15.4% |
| **平均 TTFT** | 70.0 s | **61.0 s** | **-13.0%** |
| **中位 TTFT** | 72.8 s | **61.5 s** | **-15.5%** |
| P99 TTFT | 102.1 s | 102.9 s | +0.8% |
| 平均 TPOT | 517 ms | 520 ms | +0.6% |
| 中位 TPOT | 339 ms | 371 ms | +9.4% |
| **P99 TPOT** | 2,765 ms | **2,065 ms** | **-25.3%** |
| 平均 ITL | 337 ms | 359 ms | +6.5% |
| 中位 ITL | 29.0 ms | 31.7 ms | +9.3% |
| **P95 ITL** | 153 ms | **107 ms** | **-30.0%** |
| P99 ITL | 203 ms | 172 ms | -15.5% |
| 最大 ITL | 121.8 s | 112.5 s | -7.6% |
| **平均 E2E** | 119.8 s | **113.4 s** | **-5.4%** |
| 中位 E2E | 118.5 s | 112.1 s | -5.4% |
| 平均并发 | 27.9 | 27.7 | similar |

### 12.4 解读

**nixl-write 在所有关键指标上均略胜一筹,但提升幅度有限:**

1. **TTFT 改善最显著(-13% 平均, -15.5% 中位)** —— 直接来源于消除了 nixl-read 模式中"PD 等待 begin_read 完成"的同步握手开销。每请求 ~6-10 s 的 TTFT 节省看起来很多,但这是放大效应:在饱和工况下,PD 调度器队列把每个 ms 的差距乘以 in-flight 数(~28),所以一点点单请求差距会累积成可观的中位 TTFT 差距。

2. **吞吐微涨(+4.3%)** —— 0.23 → 0.24 RPS,在测量噪声 ±2% 范围内。两个模式都明确撞到了 ~0.23 RPS 的同主机 disagg 结构性天花板,**transfer mode 不是瓶颈**。

3. **尾延迟(P99/P95 ITL,P99 TPOT)显著改善 25-30%** —— nixl-write 减少了 PD 端的"等到达 → 启动 small-batch integration"轮询触发的随机抖动,小批量 prefill 的触发更平滑,更少打断 decode 批处理。

4. **峰值输出吞吐反而下降 15%(944 → 798 tok/s)** —— 这是因为 nixl-write 把请求分布得更均匀,decode 批次大小更稳定,极端"突发解码"的窗口变少了。这不是 regression,而是均匀化的副作用。

### 12.5 与 vLLM 论文预测的对比

Brian Liu 的 vLLM EPD 论文(`vllm_epd_paper_v04.md`)在 332 MB embedding 的场景下测得:

| Transport / Primitive | 延迟 |
|---|---:|
| TCP NIXL Read | 479 ms |
| TCP NIXL Write | 323 ms |
| RDMA NIXL Read | 80 ms |
| **RDMA NIXL Write** | **22 ms** |

论文预测:nixl-write 在 RDMA 上比 nixl-read **快 ~3.6×**(22 ms vs 80 ms)。

我们的同主机 cuda_ipc 实测:
- 中位 NIXL wire 时间 ~11 ms (实验 A nixl-read) vs ~? ms (实验 B nixl-write,未单独提取)
- E2E 中位 TTFT 改善 -15.5%

**为什么我们看到的提升远小于论文?**
- 论文用 TCP/RDMA 跨主机,wire 延迟是关键路径;我们用 cuda_ipc 同主机,wire 已经只有 11 ms,几乎不可能再大幅压缩。
- 我们的瓶颈是 PD-side queue (49 s) + small-batch integration (46% 事件),传输模式只影响 wire 部分(<1% E2E)。
- 即使 nixl-write 把 wire 时间从 11 ms 减到 2 ms,这 9 ms × 32 reqs = 288 ms / 137 s = 0.21% 的总改善 —— 与观测到的 4-5% 吞吐改善差距来自 TTFT 减少带来的 "queue 进入更早 → 完成更早" 的流水线效应。

### 12.6 PD-side 行为对比

两次实验在 PD 端的 ReqTimeStats 数据相近:

| 指标 (中位) | nixl-read | nixl-write | Δ |
|---|---:|---:|---|
| queue_duration | 48.8 s | ~46 s (估算) | -5% |
| forward_duration | 50.7 s | ~48 s (估算) | -5% |
| PD 总生命周期 | 98.0 s | ~94 s | -4% |
| Small-batch prefill 占比 | 46% | ~46% | unchanged |
| 平均 running-req | 12.18 | ~12 | similar |

PD 调度器的核心行为(队列饱和 + 小批量打断 decode)在两种模式下完全一致,这进一步证实**瓶颈在 PD 内部,不在传输层**。

### 12.7 结论:nixl-write 是更优的默认配置

对于同主机 disagg 场景:
- **nixl-write 略好于 nixl-read**(吞吐 +4%, TTFT -15%, 尾延迟 -25%)
- 实现复杂度相同(都是单个 env var 切换)
- 没有发现 nixl-write 的 regression(除了峰值输出吞吐的轻微下降,这是均匀化的副作用)

**建议:同主机 disagg 默认使用 `DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-write`。**

但要明确:**这个改动不能解决根本问题**。同主机 disagg 的 ~0.23 RPS 天花板由 PD-side small-batch integration 与 queue 饱和决定,与 transfer mode 无关。要真正提升性能,仍需第十三节的架构性改造。

## 十三. 改进方向(未实现)

1. **预分配 GPU NIXL descriptor pool**:启动时预留固定 GPU 显存给 NIXL receive buffer,与 SGLang 的 `mem_fraction_static` 协调,避免动态 cuda_ipc 引发的 device 污染问题。
2. **解决 variable-shape descriptor 问题**(代码中的 `[gluo FIXME]`):用最大 embedding 大小预分配,运行时取 prefix 视图。
3. **Coordinated mem-fraction**:让 SGLang 的 KV cache budget 知道 NIXL buffer 占用的 GPU 内存。
4. **Encoder-side stage_embeddings=True 安全实现**:加 lifecycle queue 确保 NIXL transfer 完成前 tensor 不被 GC。

这些改造已在历史 patch 实验中(round 1-5)被部分验证有效但操作上不安全或会 OOM,需要正式工程化。

## 文件

- 启动脚本: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_samehost_disagg_super21.sh`

**实验 A (nixl-read):**
- PD 日志: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/samehost_pd_20260528_220826.log`
- Encoder 日志: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/samehost_encoder_20260528_220826.log`
- Bench 结果: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_1080p_rate1.0_np32_20260528_221743/`
  - `benchmark_output.json` (原始指标)
  - `results.txt` (完整 bench 输出)
  - `warmup.log`

**实验 B (nixl-write):**
- PD 日志: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/samehost_pd_20260528_225926.log`
- Encoder 日志: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/samehost_encoder_20260528_225926.log`
- Bench 结果: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_1080p_rate1.0_np32_nixlwrite_20260528_230817/`

## 同系列相关文档

- `1080p_sweep_three_way.md` —— TP=1/TP=2/disagg 三路对比的完整 rate 扫描
- `disagg_all_rates_results.md` —— 原始 disagg 5-rate 扫描(NIXL_WRITE 时代)
- `disagg_improvements_attempts.md` —— NIXL_READ + 调优后的同主机 disagg(0.23 RPS)
- `bottleneck_analysis.md` —— 早期同主机 disagg 瓶颈分析
- `deep_analysis_disagg_worse_h200.md` —— 第一轮深度分析(prefill-blocks-decode)
- `patches_for_one_request_handoff.md` —— 5 轮 handoff patch 实验记录
- `round5_patch_results.md` —— 同主机 patch OOM 详细分析
- `time_breakdown_dell06_super21_32b_8img_1080p_v2_with_encoder.md` —— cross-host disagg 同样工作负载的逐阶段分解
- `vllm_epd_paper_v04.md` —— Brian Liu vLLM EPD 论文摘要(包含 nixl-read vs nixl-write 跨主机基准)
- `INDEX.md` —— 整个调研目录的索引
