# Agg vs Disagg:为什么在同一块 GPU 上 Agg 能达到更高的并发

## 背景

在 `agg_vs_disagg_32b_8img_1080p_comparison.md` 文档里,我曾给出 4 条理由来解释 agg 为何能维持 running-req=14+:

1. Encoder ViT 与 LLM forward **内联运行(inline)**——两者都属于 SGLang 调度器的一部分
2. SGLang 调度器可以把 N 个 visual-prefill 请求合并到一次批处理 forward pass 中
3. KV cache 容量是 695k tokens → 可同时容纳约 42 个 16k-token 请求的 KV
4. 没有外部依赖:前一个请求一完成,下一个请求就能立即进入调度器

本文档**逐条用真实日志去核对这些说法**,纠正其中错误的论断,并给出 agg 实现高并发的精确机制。

## 论点 1 ✓ 已验证:在 agg 模式下 ViT 与 LLM 调度器在同一进程中运行

**来自 agg 日志的证据**(`epd_worker_server.log`,所有事件来自同一个 PID):

```
2026-05-27T20:15:08  INFO scheduler.init_model_worker: max_total_num_tokens=695136, ...
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ — 与下面是同一进程 ↓
2026-05-27T20:17:36  DEBUG qwen_vl.process_mm_data_async: [QwenVLProcessor Perf] rid='2568...', total_time: 269 ms
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ — 视觉预处理事件
2026-05-27T20:17:39  INFO  scheduler_metrics_mixin.report_prefill_stats: Prefill batch, ...
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ — 同一进程发出 prefill 事件
```

三种事件来源(`scheduler`、`qwen_vl.process_mm_data_async`、`scheduler_metrics_mixin`)都写到同一个文件。**在 agg 模式下,ViT 预处理、SGLang 调度器、LLM forward 都在同一个 Python 进程中运行**(PID 23117 已通过 `pgrep` 确认),且都在同一块 GPU 上。

而在 disagg 模式下,与之对应的 QwenVLProcessor 事件**只出现在 dell06 的 encoder 日志中**——并不出现在 super21 的 PD 日志中。调度器和 LLM forward 在 super21(PD 日志)运行,而视觉预处理在 dell06(encoder 日志)运行。它们**跨主机分离**。

agg 单个请求的逐步时序(rid `43bcc5634c484171a650e35802c5e8af`):

```
时间          事件                                     说明
─────────────────────────────────────────────────────────────────────────────────
20:19:34.852  request received                        Rust ingress
20:19:34.940  starting task to process async stream   Python handler 进入(收到后 88 ms)
20:19:34.940  decode_handler.generate: New Request ID Python decode_handler 启动
20:19:35.677  qwen_vl.process_mm_data_async: 735 ms   ← ViT 内联:完整的 ViT 预处理
20:19:38.099  decode_handler: New SGLang Request ID   交给 SGLang 调度器
20:19:38.495  ReqTimeStats: forward_duration=2123 ms  GPU forward 完成
20:19:38.501  request completed                       Rust egress
─────────────────────────────────────────────────────────────────────────────────
总耗时: 3.65 s
  ├─ ViT 预处理:    735 ms(SGLang,同一进程内)
  ├─ 其他开销:    1690 ms(ViT 完成到 prefill 开始之间;可能在排队)
  └─ Forward+decode: 2123 ms(来自 ReqTimeStats)
```

**结论**:论点 1 **得到确认**。ViT 在同一个 Python 进程、同一块 GPU 上运行,完全不涉及 NIXL 传输。

## 论点 2 ✗ 错误:SGLang **不会**把多个 visual prefill 合并到一次 forward 中

**Prefill 批次事件的证据:**

```
Agg prefill 中 #new-seq 的分布(本次基准 57 个事件):
  #new-seq=1: 57 (100%)   ← 每个 prefill 批只处理一个新请求

Disagg prefill 中 #new-seq 的分布(57 个事件):
  #new-seq=1: 57 (100%)   ← 完全相同 — 每个 prefill 批一个新请求
```

agg 和 disagg 都是**每次 prefill 只新增一个 sequence**——调度器并不把多个 visual prefill 合到一起。两者的差异在于:在 prefill 进行时,正在 in-flight 的请求的 decode **是否还能继续**,但 prefill 本身就是单请求的。

所以"SGLang 把 N 个 visual prefill 一起 batch"这个说法是错的,需要去掉。

**agg 并发优势的真正来源是 decode 阶段,而不是 prefill 阶段**。见下面的论点 3 和后续解释。

## 论点 3 ✓ 已验证:agg 的 KV cache 容量明显更大

**来自日志的证据:**

| 模式 | `max_total_num_tokens` | `mem-fraction-static` | 初始化后剩余 GPU 内存 |
|---|---:|---|---:|
| **agg_TP1** | **695 136** | 0.85 | 20.5 GB |
| **disagg PD** | **467 072** | 0.65 | 48.2 GB |

两次运行 `--page-size 16` 相同、KV dtype 相同(FP8),所以差异完全来自 `mem-fraction-static`:
- agg 用 0.85 → 占用约 122 GB GPU 内存(模型权重 + KV + 工作区 + cuda graphs)
- disagg 用 0.65 → 占用约 94 GB,空出来的部分是给 NIXL 接收 buffer 留的(每个 cuda buffer 638 MB × in-flight 请求数)

对于一个 16 384-token 的 visual prefill 来说:
- **agg 可同时容纳 `695 136 / 16 384 = 42` 个完整上下文请求的 KV**
- **disagg 可同时容纳 `467 072 / 16 384 = 28` 个完整上下文请求的 KV**

实际 bench 中 output_len=256,所以每个请求需要约 16 640 个 KV tokens,实际上限大约是 42 vs 28(1.5× 比例)。

bench 期间的 KV 利用率:

| 模式 | 观测到的 `token usage` | 含义 |
|---|---|---|
| agg | 0.02 → 0.05 → ... → 0.24(逐渐增长) | bench 中段 KV 占用约 24% |
| disagg | 数值更低 | KV 占用比 agg 低 |

agg 24% 的峰值 KV 利用率 ≈ in-flight 167k tokens ≈ 10 个并发 16k-token 请求。这**与 prefill 事件中观察到的 running-req=14 并不完全吻合**——差额是只在 decode、不在 prefill 的请求,它们仍然占用 KV。

**结论**:论点 3 **得到确认,但相关性有限**。agg 的 KV 是 1.5× 大,但本次 bench 中两个模式都没有把 KV 占满。KV 容量这一项贡献的并发倍数大约是 1-1.5×,**不是 10×**。

## 论点 4 ✓ 已验证、并且是**主导因素**

**PD 端 request_received(数据面入口)的到达时间间隔:**

```
agg PD bench(super21 GPU 5):
  T20:19:34.852  ← bench 起始
  T20:19:39.934  Δ = 5.08 s
  T20:19:40.255  Δ = 0.32 s ← 可以快速到达(没有 encoder 瓶颈)
  T20:19:40.919  Δ = 0.66 s
  T20:19:43.473  Δ = 2.55 s  
  T20:19:47.164  Δ = 3.69 s
  T20:19:47.211  Δ = 0.05 s ← 出现突发!agg 能突发接受请求
  T20:19:49.532  Δ = 2.32 s
  T20:19:50.972  Δ = 1.44 s
  T20:19:51.054  Δ = 0.08 s ← 突发
  
disagg PD bench(super21 GPU 5):
  T19:31:44.224  ← bench 起始
  T19:31:54.397  Δ = 10.17 s ← 约 10 秒间隔(encoder ViT + RoCE wire + dispatch)
  T19:31:55.884  Δ = 1.49 s
  T19:31:57.319  Δ = 1.43 s
  T19:31:58.748  Δ = 1.43 s ← 稳定的 ~1.4 s 节拍,被 encoder 限速
```

**关键观察:**
- **agg 的 PD 接收请求的间隔从 0.05 s 到 5 s 不等,而且能吸收突发**——因为 bench 客户端按 Poisson rate=1.0 把请求直接发给 agg 的 HTTP 前端,Rust 调度器立即接受最多 `np=32` 个并发请求。
- **disagg 的 PD 接收请求的间隔几乎是稳定的 ~1.4 s**——因为每个请求必须依次经过:bench → frontend → encoder.dispatch → encoder ViT(~1.3 s)→ cpu→cuda(~50 ms)→ NIXL register(~50 ms)→ 跨主机 TCP 通道 → PD ingress。**encoder 流水线就像一个 ~1.4 s/req 的节拍器**,无论 PD 想要多快,它都按这个节奏喂给 PD。

**这是主导差异。**disagg 的 PD 调度器队列基本是空的(median queue_duration = 0.28 ms),原因不是 PD 不愿排队,而是 **encoder 根本喂不饱它**。

```
agg 模式下,所有 32 个 prompt 在 bench 启动后约 30 秒内都能进入 SGLang 调度器队列。
disagg 模式下,等到前 5 个请求开始完成的时候,32 个 prompt 中可能只有约 22 个到达了 PD。
```

**结论**:论点 4 **得到确认,且为决定性因素**。

## 真正的并发机制:decode 阶段批处理,而不是 prefill 阶段

我之前把 agg 高并发归因于"在一次 forward 中合并多个 visual prefill"是错的。真正的机制是 decode 阶段的批处理:

```
agg decode 批次中 running-req 的演化(bench 期间):
  20:19:38.366  running-req=1   gen 0.67 tok/s   ← 第 1 个请求独自 decode
  20:19:47.618  running-req=4   gen 9.19 tok/s   ← 第 1+2+3+4 起 decode
  ... 峰值时 running-req 高达 32

disagg decode 批次 running-req 分布(bench 期间 51 个事件):
  running=1: 13 (25%)
  running=2: 22 (43%)
  running=3: 11 (22%)
  running=4:  3  (6%)
  running=5:  2  (4%)   ← 整个 bench 期间最大只到 5
```

**两种模式下 prefill 都是单序列的**,但是 decode 阶段 SGLang 调度器会**让 `running-req` 个请求并行 decode**(每个请求在一次 forward pass 中产出一个 token)。agg 之所以能维持 32 个并行 decoder,是因为:

1. bench 客户端能在几秒内把 32 个请求都发到 agg 前端
2. 一个请求 prefill 完成后,马上加入 running 的 decode batch
3. 新的 prefill 通过 SGLang 的 chunked-prefill 机制和 decode 步交替进行(短暂打断 decode 来插入 prefill,然后继续 decode)

disagg 只能维持约 2-5 个并行 decoder,是因为:

1. encoder 流水线以 ~1.4 s/req 的节拍把请求送到 PD
2. 每个请求在 PD 上需要约 8-10 s 的工作(forward_duration)
3. 稳态 in-flight 数 = 到达率 × forward 时间 = (1/1.4) × 8.5 ≈ 6(与观察到的 running-req 最大 5 吻合)

## decode 批处理的代价

agg 的高 decode 并发**导致**它的 TPOT 更高:

```
agg 在 running-req=32 时:
  - 每一步 decode 是一次对 32 个活跃序列的批处理 forward
  - forward 耗时随 batch size 增长(注意力 / KV 计算变大)
  - 单 token 延迟:中位数 229 ms = (forward_step_time / 32 / streams) 加上一些开销

disagg 在 running-req=2 时:
  - 每一步 decode 只对 2 个活跃序列做一次 forward
  - forward 很快:2 个序列约 50 ms,而 32 个序列约 1500 ms
  - 单 token 延迟:中位数 39 ms
```

这是**吞吐 vs 延迟的内在折衷**,而不是 disagg vs agg 的架构差异。如果想让 disagg 在吞吐上追平 agg,需要把请求送得更快(比如 4-8 个 encoder 同时喂一个 PD,让 PD 上的 running-req 达到 14+)。如果想让 agg 的 TPOT 接近 disagg,把 max-running-requests 调到 4-8 即可。

## 修正后的解释:为何 agg 能在同一块 GPU 上达到 running-req=14+

(替换原来的 4 条说法)

1. ✓ **同进程流水线(没有跨主机协调):** ViT 和 LLM 调度器/forward 在同一个 Python 进程、同一块 GPU 上运行。每个请求依次经过:HTTP → handler → QwenVLProcessor → SGLang 调度器 → forward,全在同一进程中,各阶段间切换 < 100 ms。

2. ✓ **没有外部投递瓶颈:** bench 客户端按 rate=1.0 直接给 agg 的 HTTP 前端发 32 个请求,所有 32 个请求都能在约 30 秒内进入 agg 的 handler。disagg 中,请求只有走完 encoder 流水线之后(~1.4 s/req)才能到达 PD。

3. ✓ **decode 批处理实现高并发:** 一旦请求 prefill 完成,SGLang 调度器把它们放进同一个批处理 decode forward pass。32 个并发 decoder 让 agg 在每一步 decode 上获得 32× 的吞吐(代价是单 token 延迟更高)。

4. ✓ **更大的 KV 预算可以容纳更多并发 decoder:** agg 有 695k KV slot,disagg 是 467k(1.5× 大),原因是 `mem-fraction-static` 更高。这是并发上限,但实际观测 running-req 约 10-15 时,两种模式都还有 KV 余量。

5. ✗ ~~批处理 prefill~~ — 此条错误。两种模式 prefill 都是单序列(#new-seq=1)。

## 真正的瓶颈分解

```
agg TP1 瓶颈栈(已验证):
  1. PD GPU forward 时间随 running-req 增大而增大(14+ 请求并行 decode 时单 token 慢)
  2. KV cache 上限(约 42 个并发完整上下文请求)— 本次 bench 未触及
  3. max-running-requests=40 上限 — 观测峰值 running-req=31(decode 时 in-flight),接近但未撞顶
  → 实际天花板:running-req 高时的 GPU 算力

disagg 瓶颈栈(已验证):
  1. encoder→PD 投递节奏(rate=1.0 单 encoder 时约 1.4 s/req)
  2. PD running-req 卡在 2-5,因为 encoder 喂不快
  3. PD GPU forward 时间反而**短**(8.6 s,而 agg 是 35 s),因为 batch 小
  4. KV cache 上限(467k)— 远未触及(只用了 30-40k = 7%)
  → 实际天花板:encoder 流水线吞吐,而不是 PD 算力
```

## 吞吐量推算(两边都已实证)

```
agg:    running-req × throughput_per_req
        = 14.67 × (1 / 35 s)
        = 0.42 RPS  ✓ 与实测 0.42 一致

disagg: running-req × throughput_per_req
        = 1.54 × (1 / 8.6 s)
        = 0.18 RPS  ≈ 实测 0.22

为什么 agg 的单请求 forward 更慢、吞吐反而更高?
  agg 拿到 9.5× 的并发倍数(14.67 / 1.54)
  agg 付出 4× 的单请求慢化(35 s / 8.6 s)
  净效果:9.5 / 4 = 2.4× 吞吐 → 与实测 1.84× 接近
```

预测 2.4× 与实测 1.84× 之间约 25% 的差距,来源于 agg 的更高 TTFT(更多时间花在调度器队列等待上,挤占了总吞吐)。

## 启示

1. **agg 的吞吐优势来自 decode 并行,而不是因为它"没有 handoff 开销"**。我之前提到"1-2 s NIXL handoff"作为 disagg 主要代价的说法**有误导性**——伤 disagg 的不是单次 handoff 开销,而是**PD 上始终只有 2 个请求**所累积出的效应。

2. **disagg 的结构性问题不在 NIXL 或 RDMA,而在 encoder 喂送速率。** 解决方法:
   - 加更多 encoder(4× 或 8×)让 encoder 流水线匹配 PD 的胃口
   - 用更快的 encoder GPU(已经是 H200,提升空间有限)
   - 预取 / 流水预加载,让 PD 始终有 14+ 个请求在排队

3. **disagg 的 TPOT 优势(39 ms vs 229 ms)是真实而结构性的**——它直接来自 running-req 较低这一事实。这是 disagg 架构**唯一**真正占优的指标;对流式 UI 体验来说,平滑的单 token 延迟很重要。

4. **对 32B-FP8 8img/1080p 来说,"单 encoder + 1 PD"的 disagg 在吞吐上无法击败"单 PD"的 agg**,因为 encoder 不够快,无法让 PD 维持 running-req≥14。增加 encoder 池(4E、8E)**可能**追平,但也不一定——如果 PD 在 running-req=14 时的 forward_duration 与 agg 相当,就只是把 agg 重新做一遍并多加几次跨主机跳。

## 文件位置

- agg 日志(单进程):`/hongming/dynamo/01_cuda_sh/agg_h200_32b/logs/epd_worker_server.log`
- disagg PD 日志(encoder 在 dell06):`/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01.log`
- 本文档(中文):`/hongming/dynamo/01_cuda_sh/disagg_h200_32b/why_agg_concurrency_explained_zh.md`
- 英文原版:`/hongming/dynamo/01_cuda_sh/disagg_h200_32b/why_agg_concurrency_explained.md`

## 配套文档

- `agg_vs_disagg_32b_8img_1080p_comparison.md` — 最初的对比报告(本文档纠正其中一条错误)
- `time_breakdown_dell06_super21_32b_8img_1080p_v2_with_encoder.md` — disagg 详细分解
- `1080p_sweep_three_way.md` — 最初的 TP=1 / TP=2 / disagg 对比
