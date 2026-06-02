# Same-Host Disagg E/PD —— 四种工作负载 RPS 对比 (rate=1.0, np=32)

**测试日期:** 2026-05-30 / 2026-05-31(16img 补测)
**主机:** sc09super21-h200 (172.26.46.133)
**模型:** Qwen3-VL-32B-Instruct-FP8
**拓扑:** 同主机 disagg —— Encoder GPU 0 + PD GPU 1, NUMA 0, NV18 NVLink
**Patches:** **未应用** (h200_cuda_nixl.patch + b70_xpu_nixl.patch 均已回滚)
**NIXL 模式:** `DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-write`
**Mem fraction:** encoder=0.85, PD=0.85
**测试参数:** in=128, out=256, np=32, rate=1.0 RPS
**Bench 客户端:** `sglang.bench_serving --backend sglang-oai-chat --seed 0`

## 四种工作负载 RPS 对比

| Workload | RPS | Bench 持续时间 | 成功请求 | 中位 TTFT | 中位 E2E | 中位 TPOT | 平均并发 |
|---|---:|---:|:---:|---:|---:|---:|---:|
| **8 img / 1080p** (rate=1.0/np=32) | **0.24** | 130.9 s | 32/32 | 61.5 s | 112.1 s | 371 ms | 27.7 |
| **16 img / 768p** (rate=1.0/np=32) | **0.35** | 91.7 s  | 32/32 | 40.6 s | 71.8 s  | 225 ms | 25.9 |
| **8 img / 768p** (rate=1.0/np=32)  | **0.64** | 49.8 s  | 32/32 | 23.2 s | 37.6 s  | 113 ms | 24.9 |
| **8 img / 768p** (rate=1.0/np=128) ★长时间饱和★ | **0.70** | 181.9 s | 128/128 | 39.9 s | 84.9 s  | 489 ms | 64.6 |
| **8 img / 768p** (rate=2.0/np=64) ★加压实测真实饱和★ | **0.72** | 89.3 s  | 64/64 | 36.2 s | 66.8 s  | 267 ms | 49.0 |
| **8 img / 768p** (rate=2.0/np=256) ⚠ goodput 崩溃 | **0.65** | 231.3 s | **151/256** | 77.2 s | 136.0 s | 475 ms | 93.0 |
| **4 img / 768p** (rate=1.0/np=32)  | **0.84** | 37.9 s  | 32/32 | **2.4 s** | **8.1 s** | **25 ms** | 7.0 |

## 完整指标对比

| 指标 | 8img/1080p | 16img/768p | 8img/768p | 4img/768p |
|---|---:|---:|---:|---:|
| **吞吐 RPS** | **0.24** | **0.35** | **0.64** | **0.84** |
| 输入吞吐 (vision tokens/s) | 4012 | 4327 | 4006 | 2662 |
| 输出吞吐 (tokens/s) | 37.3 | 53.3 | 98.0 | 128.7 |
| 峰值输出吞吐 (tok/s) | 798 | 984 | 1347 | 1095 |
| Vision tokens/req(实测中位 input_len) | 16,420 | 12,401 | 6,238 | 3,158 |
| 平均 TTFT (ms) | 60,962 | 39,085 | 22,004 | 2,922 |
| 中位 TTFT (ms) | 61,474 | 40,649 | 23,246 | 2,393 |
| P99 TTFT (ms) | 102,919 | 61,787 | 26,550 | 5,995 |
| 平均 TPOT (ms) | 520 | 349 | 165 | 38 |
| 中位 TPOT (ms) | 371 | 225 | 113 | 25 |
| P99 TPOT (ms) | 2,065 | 1,892 | 779 | 116 |
| 平均 ITL (ms) | 359 | 234 | 113 | 40 |
| 中位 ITL (ms) | 32 | 24 | 18 | 13 |
| 最大 ITL (ms) | 112,527 | 76,391 | 35,691 | 6,727 |
| 平均 E2E (ms) | 113,367 | 74,207 | 38,798 | 8,265 |
| 中位 E2E (ms) | 112,078 | 71,751 | 37,595 | 8,081 |
| P99 E2E (ms) | 129,796 | 89,371 | 49,114 | 16,307 |
| 平均并发 | 27.7 | 25.9 | 24.9 | 7.0 |
| 峰值并发 | 32 | 32 | 32 | 20 |

## 关键观察

### 1. 吞吐与每 prefill batch 容纳的请求数挂钩(不是简单的"input_len 反比")

**深度分析见 `same_host_768p_problem_analysis_zh.md`**:RPS 不是简单的 `1/input_len`,而是受 input_len 与 chunked_prefill_size(16,384)的比值决定:

| 工作负载 | input_len | chunked_prefill_size 占比 | 每 batch 容纳请求数 | 实测 RPS |
|---|---:|---:|:---:|---:|
| 8img/1080p | 16,420 | 100%(切分) | 1(+小尾巴) | 0.24 |
| 16img/768p | 12,401 | 76% | **1**(2 个超 budget) | **0.35** |
| 8img/768p  |  6,238 | 38% | **2** | 0.64 |
| 4img/768p  |  3,158 | 19% | **5** | 0.84 |

`RPS ≈ Total_token_throughput / per_req_tokens`,实测精度极高:
- 16img: 4,381 / (12,401+153) = 0.349 ≈ **0.35** ✓
- 8img:  4,104 / ( 6,238+153) = 0.642 ≈ **0.64** ✓
- 4img: input-rate-limited(未饱和)

### 2. 8img/768p 加压实测真实饱和点 0.70-0.72 RPS,过载会发生 goodput 崩溃

rate=1.0/np=32 时实测 0.64 RPS,但 in-flight 峰值仅 30 — 显然受输入速率限制。三组独立加压:

| 指标 | rate=1.0/np=32 | **rate=1.0/np=128** | **rate=2.0/np=64** | **rate=2.0/np=256(过载)** |
|---|---:|---:|---:|---:|
| **RPS** | 0.64 | **0.70** | **0.72** | **0.65 ↓** |
| **成功请求** | 32/32 | 128/128 | 64/64 | **151/256**(105 失败) |
| Total token throughput (tok/s) | 4,104 | 4,470 | 4,552 | 4,152 |
| **PD running-req 峰值** | 30 | **63** | **63** | **63** |
| Queue-req 峰值 | 20 | **33** | 30 | 33 |
| 中位 queue_duration | 13.3 s | 31.4 s | 23.1 s | 33.8 s |
| 中位 forward_duration | 18.3 s | 38.0 s | 33.2 s | **58.0 s** |
| P99 forward_duration | — | — | 86 s | **198 s** |
| 中位 E2E(成功) | 37.6 s | 84.9 s | 66.8 s | **136.0 s** |
| 中位 TTFT | 23.2 s | 39.9 s | 36.2 s | **77.2 s** |
| 平均并发 | 24.9 | 64.6 | 49.0 | 93.0 |
| Bench 持续时间 | 50 s | 182 s | 89 s | **231 s** |
| 失败模式 | — | — | — | **NIXL embedding buffer 池耗尽** |

**两组未崩溃的加压(np=128/r=1, np=64/r=2)的实测 RPS 高度收敛(0.70-0.72)**,两者 PD running-req 峰值都精确停在 63/64,total token throughput 都接近 4,500 tok/s — **强证据表明 8img/768p 的真实饱和上限是 0.70-0.72 RPS**。

**真实瓶颈构成:**
1. **`--max-running-requests=64`** 是硬性上限:三次加压实测 in-flight 总是停在 63
2. **GPU compute throughput** 接近饱和:in-flight 从 30 → 63 翻倍,total tput 只升 11%
3. **Prefill batch 在长时间饱和下退化为单请求 batch**:np=128 持续 3 分钟饱和时,44% 的 prefill batch 是 6,238 token(单请求),np=256 过载时 51%,因为 running batch 占用了 KV slot 让 `rem_chunk_tokens` 缩小
4. **过载时 NIXL embedding buffer 池耗尽**:`NIXL_MAX_BUFFER_SIZE=805306368`(768 MB)≈ 12 个并发 embedding(每个 ~63 MB),input rate=2.0 ≫ PD admit 0.72 时迅速被 stuck embedding 占满,新请求 60s 内拿不到 buffer → `Timeout while waiting for available buffer`

**生产红线:** 8img/768p 安全工作区 RPS ≤ 0.65,允许少量突发到 0.7,**绝对不能持续超过 0.72 否则会失败**。需要前端 admission control。

详见 `same_host_768p_problem_analysis_zh.md` §3.5-3.7。

### 3. 4img/768p 在 rate=1.0 下未饱和

- **平均并发仅 7.0**(其他三个工作负载是 25-49),说明 PD 还有富余容量
- **中位 TTFT 仅 2.4 s**(其他三个是 23-61 s)
- 真正饱和应该在 rate=2.0 或 3.0 才会出现

### 4. 16img/768p 与 1080p 的对比验证"1-请求-per-batch"假设

16img/768p 的 input_len(12,401)虽然小于 chunked_prefill_size(16,384),不会切分,**但两个请求的合并需求(24,802)远超 budget**,所以 scheduler 每次只 admit 1 个请求。其行为质上类似 1080p:

- KV pool 触顶(in-flight 卡 31)
- 中位 queue_duration 25 s(1080p 是 49 s,8img/768p rate=1.0 是 13 s)
- GPU compute bound:RPS 0.35 = total tput 4,381 tok/s ÷ 12,554 tokens/req

### 5. 子饱和 (rate=0.1) 单请求延迟

| Workload | 单请求 TTFT (warmup, rate=0.1) |
|---|---:|
| 4img/768p  |  1.5 s |
| 8img/768p  |  3.2 s |
| 16img/768p |  ~7 s(估算) |
| 8img/1080p | 12.3 s |

这是同主机 disagg 在每种工作负载下"单请求最优"的下限。

### 6. 与既有文档的对照

| Workload | 同主机 disagg(本次) | Cross-host dell06_1E | 同主机 TP=1 agg |
|---|:---:|:---:|:---:|
| 8img/1080p | 0.24 | 0.24 | 0.45 |
| 16img/768p | 0.35 | (未测) | (未测) |
| 8img/768p (rate=1.0/np=32)  | 0.64 | 0.62 | 1.26 |
| 8img/768p (rate=1.0/np=128,饱和) | **0.70** | (未测) | (未测) |
| 8img/768p (rate=2.0/np=64,饱和) | **0.72** | (未测) | (未测) |
| 8img/768p (rate=2.0/np=256,**过载崩溃**) | **0.65 + 105/256 失败** | (未测) | (未测) |
| 4img/768p  | 0.84 | 0.78 | 0.87 |

(cross-host 数据来自 `agg_vs_disagg_full_sweep_np32.md`,TP=1 agg 数据来自 `1080p_sweep_three_way.md`)

**结论:**
- **同主机 disagg ≈ cross-host dell06_1E** —— 同主机不比 cross-host 强,因为瓶颈是 PD-side scheduler + KV cache,不是 NIXL 传输
- **同主机 disagg < TP=1 agg** —— 在 8img 工作负载上 agg 快 1.4-2×,因为 agg 没有 NIXL handoff 也没有小批量尾巴 prefill
- **4img/768p 是唯一同主机 disagg 接近 agg 的工作负载** —— 因为 input_len 3.1k 远小于 chunked_prefill_size 16,384,5 个请求合并到一个 prefill batch,没有被 admission 速率限制
- **8img/768p 加压实测 0.70-0.72 RPS,只达 TP=1 agg 1.26 的 56-57%** —— disagg 的 NIXL handoff overhead + max_running_requests + chunked-prefill admission 三个因素共同压制吞吐

## 配置详情

**启动脚本:** `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_samehost_disagg_super21.sh`

**关键参数:**
```bash
CUDA_DEVICE_PD=1
CUDA_DEVICE_ENCODER=0
MEM_FRAC_PD=0.85
MEM_FRAC_ENC=0.85
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-write
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy
UCX_NET_DEVICES=mlx5_0:1
ENABLE_ENCODER_CACHE=0
DYN_TCP_MAX_MESSAGE_SIZE=268435456
DYN_HTTP_BODY_LIMIT_MB=256
```

**PD worker:**
```bash
python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal --enable-mm-global-cache --multimodal-worker \
    --dtype auto --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 64 --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 --page-size 16 \
    --chunked-prefill-size 16384 \
    --enable-request-time-stats-logging --show-time-cost
```

**Encoder worker:**
```bash
python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal --multimodal-encode-worker \
    --multimodal-embedding-cache-capacity-gb 16 \
    --chat-template qwen2-vl \
    --dtype auto --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 --page-size 16 \
    --enable-request-time-stats-logging --show-time-cost
```

## 结果文件

- 8img/1080p: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_1080p_rate1.0_np32_nixlwrite_20260528_230817/`
- 16img/768p: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/16img_768p_rate1.0_np32_nixlwrite_20260531_154933/`
- 8img/768p (rate=1.0/np=32):  `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_768p_rate1.0_np32_nixlwrite_20260531_145314/`
- 8img/768p (rate=1.0/np=128): `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_768p_rate1.0_np128_nixlwrite_20260531_164750/`
- 8img/768p (rate=2.0/np=64):  `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_768p_rate2.0_np64_nixlwrite_20260531_163936/`
- 8img/768p (rate=2.0/np=256,**过载**): `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_768p_rate2.0_np256_nixlwrite_20260531_170805/`
- 4img/768p:  `/hongming/res_samehost_disagg_32b_gpu01_unpatched/4img_768p_rate1.0_np32_nixlwrite_20260531_145548/`

每个目录包含:
- `benchmark_output.json` — 原始指标
- `results.txt` — 完整 bench stdout
- `warmup.log` — 3-5 prompt 预热结果

## 同系列相关文档

- `same_host_disagg_time_zh.md` — 8img/1080p 单一工作负载的完整时间分解(含 nixl-read vs nixl-write 对比)
- `same_host_problem_analysis_zh.md` — 8img/1080p 两大瓶颈(46% 小批量 + 队列等待 48.8 s)的源码级根因分析
- `same_host_768p_problem_analysis_zh.md` — 16img / 8img / 4img 在 768p 下的瓶颈对比与 input_len-vs-chunked_prefill_size 比值理论
- `agg_vs_disagg_full_sweep_np32.md` — TP=1 agg / cross-host disagg / 同主机 disagg 的横向对比
- `1080p_sweep_three_way.md` — TP=1 agg / TP=2 agg / 同主机 disagg 在 1080p 下的 5-rate 扫描
- `INDEX.md` — 整个调研目录的索引
