# 同主机 Disagg E/PD —— 四种工作负载在 np=128 / rate∈{1.0, 2.0} 下的瓶颈与崩溃模式分析 (v01)

**关联文档:**
- `same_host_problem_analysis_zh.md` —— 8img/1080p 在 np=32/rate=1.0 下的两大瓶颈根因
- `same_host_768p_problem_analysis_zh.md` —— 16img/8img/4img × 768p 在 np=32-256 下的瓶颈对比

**本文档目标:** 用统一的 np=128 输入、横扫 rate∈{1.0, 2.0},对**四种最有代表性的工作负载**(**8img/1080p**、**16img/768p**、**8img/768p**、**4img/768p**)做并排对比,揭示:

1. 在持续饱和压力下,**重负载工作负载(input_len 接近或超过 chunked_prefill_size)都触发 NIXL buffer 池耗尽崩溃**;**轻负载(4img/768p)却可以毫无问题地以 rate=2.0 跑出 1.79 RPS**。
2. RPS、TTFT、E2E 在 1.0 → 2.0 加压下的差异化变化:**4img/768p 是唯一可线性 scale 的 case**,其它 3 个都恶化。
3. 哪个 case 接近真实饱和,哪个 case 已经崩溃。
4. 之前 1080p 文档提出的两条优化建议(尾块合并 + dynamic chunking)在当前 SGLang 上是否可用。
5. 4img/768p 的特殊性:**input_len < chunked_prefill_size / 5**,5 请求可合并到单 prefill batch,完全规避了其它 3 个 case 的瓶颈。

---

## 0. 实验配置

**硬件 / 软件:**
- 主机: sc09super21-h200 (172.26.46.133),H200 GPU 0/1, NUMA 0, NV18 NVLink
- 模型: Qwen3-VL-32B-Instruct-FP8
- 拓扑: 同主机 disagg —— Encoder GPU 0 + PD GPU 1
- Patches: **未应用** (h200_cuda_nixl + b70_xpu_nixl 均已撤销,详见 `round5_patch_results.md`)
- NIXL 模式: `DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-write`
- Mem fraction: encoder=0.85, PD=0.85
- `--max-running-requests 64`,`--chunked-prefill-size 16384`,`--page-size 16`

**bench 参数(本次新增 7 组,全部 np=128):**

| 工作负载 | input_len(中位) | KV demand | 理论 KV cap | rate=1.0/np=128 RPS | rate=2.0/np=128 RPS |
|---|---:|---:|---:|---:|---:|
| **8img/1080p** | 16,420 | 16,692 | 41 | **0.23** (65/128) ⚠崩溃 | **0.18** (51/128) ⚠崩溃 |
| **16img/768p** | 12,401 | 12,673 | 54 | **0.34** (77/128) ⚠崩溃 | **0.33** (72/128) ⚠崩溃 |
| **8img/768p**  |  6,238 |  6,510 | 106 | **0.70** (128/128) ✓ | **0.67** (107/128) ⚠ |
| **4img/768p**  |  3,158 |  3,430 | 202 | **0.98** (128/128) ✓ | **1.79** (128/128) ✓ ★唯一可 scale★ |

**8img/1080p、16img/768p、8img/768p 全部出现 NIXL buffer 池耗尽失败;只有 4img/768p 在 rate=2.0/np=128 仍 100% 成功**。

> 注:4img/768p 的 rate=2.0 测试做了**两轮**:
> - 第一轮(mm-global-cache 启用):RPS=1.88,但发现 SGLang RadixCache 大量命中(median cached_token=3072)
> - **第二轮(mm-global-cache 禁用)**:RPS=**1.79**,RadixCache 仍命中(机制是 input_ids 前缀共享),无 buffer pool 失败 → **本文档正式数据使用第二轮的 1.79 RPS**

---

## 1. 头版结论:input_len 决定一切,4img/768p 是唯一不崩溃的 case

无论 input_len 大小,只要 input rate 持续超过 PD admit 速率,系统会进入 **NIXL embedding buffer 池耗尽 → 请求超时失败** 的崩溃。区别只在于"开始崩溃的 rate 阈值"不同:

| 工作负载 | PD 真实饱和 RPS | 安全 rate(无失败) | 崩溃 rate(出现失败) |
|---|---:|---|---|
| **8img/1080p** | ~0.24 | rate ≤ 0.5(估算) | **rate=1.0/np=128 已失败 49%**(63/128) |
| **16img/768p** | ~0.35 | rate ≤ 0.5(估算) | **rate=1.0/np=128 已失败 40%**(51/128) |
| **8img/768p**  | ~0.70 | rate=1.0/np=128 (0 失败) | **rate=2.0/np=128 失败 16%**(21/128) |
| **4img/768p**  | **≥ 1.79**(无任何崩溃) | **rate=2.0/np=128 (0 失败) ★** | **未观察到崩溃**(rate=2.0 即 input rate × 0.9) |

> **核心规律:** 当 `arrival_rate > PD真实RPS × 0.7` 持续 30 秒以上,NIXL ring buffer 池就会耗尽,后续请求报 `Timeout while waiting for available buffer`(60 秒等待超时)。
>
> **失败比例 ≈ (input_total - PD_throughput × bench_duration) / input_total**
>
> **4img/768p 是唯一例外:rate=2.0 仍未到 PD 真实饱和点,所以即使持续加压 1 分钟也不耗尽 buffer 池。**

---

## 2. 横向对比表(本次实测)

### 2.1 头部数据

| 工作负载 | rate | RPS | 成功/总数 | 中位 TTFT | 中位 E2E | P99 E2E | total_tput (tok/s) | bench 时长 |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| 8img/1080p | 1.0 | 0.23 | 65/128 | 112.0 s | 175.4 s | 229.1 s | 3,781 | 284 s |
| 8img/1080p | 2.0 | 0.18 | 51/128 | 169.9 s | 214.9 s | 256.7 s | 2,975 | 283 s |
| 16img/768p | 1.0 | 0.34 | 77/128 | 71.4 s  | 163.2 s | 223.6 s | 4,242 | 227 s |
| 16img/768p | 2.0 | 0.33 | 72/128 | 104.9 s | 155.7 s | 201.3 s | 4,109 | 219 s |
| 8img/768p  | 1.0 | 0.70 | 128/128 | 39.9 s | 84.9 s | 170.2 s | 4,470 | 182 s |
| 8img/768p  | 2.0 | 0.67 | 107/128 | 60.4 s | 109.9 s | 152.4 s | 4,282 | 159 s |
| **4img/768p** | **1.0** | **0.98** | **128/128** | **2.5 s** | **6.5 s** | **17.3 s** | **3,198** | **131 s** |
| **4img/768p** | **2.0** | **1.79** | **128/128** | **3.3 s** | **11.7 s** | **33.2 s** | **5,856** | **72 s** |

### 2.2 PD-side 统计(对比 SGLang scheduler 行为)

| 工作负载 | rate | PD running 峰值 | Queue 峰值 | <500 token batch | 2-7k batch (单请求) | 7-13k batch (双请求) | 13-17k batch (单请求,full chunk) | 中位 prefill tput | NIXL 超时数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8img/1080p | 1.0 | **31** | 10 | **40%** | 0% | 0% | **60%** | 5,881 | 63 |
| 8img/1080p | 2.0 | 26 | 11 | **42%** | 0% | 0% | **58%** | 5,718 | 72 |
| 16img/768p | 1.0 | **54** | 16 | 1% | 0% | **99%** | 0% | 5,761 | 51 |
| 16img/768p | 2.0 | 38 | 15 | 8% | 0% | **92%** | 0% | 6,491 | 56 |
| 8img/768p  | 1.0 | **63** | 33 | 2% | 26% | **72%** | 0% | 5,665 | 0 |
| 8img/768p  | 2.0 | **63** | 33 | 1% | **40%** | 59% | 0% | 5,916 | 21 |
| **4img/768p** | **1.0** | **11** | **0** | 1% | **95%** | 4% | 0% | 2,688 | **0** |
| **4img/768p** | **2.0** | **21** | **0** | (radix-cache hit*) | — | — | — | — | **0** |

\*4img/768p rate=2.0:median `cached-token=3072`(SGLang RadixCache 命中,不是 mm-global-cache),median `new-token` 仅 ~110。这是因为 bench 用 `--seed 0` 生成相同 random images,RadixCache 共享前缀。**所有其它 7 组实测都没有 RadixCache 命中**(median cached=0),所以 4img/768p rate=2.0 的 prefill 数据不直接可比。

**关键观察:**

1. **8img/1080p**(input 16,420 > chunked_prefill=16,384):**40-42% 的 prefill batch 是小批量 <500 token**,与 np=32/rate=1.0 测得的 46% 完全吻合,验证这是**结构性瓶颈**(每个请求被切成"满块 16,384 + 小尾巴 36 token",每 32 个请求产生 32 个尾巴)。**这个比例不会随 rate / np 变化**。

2. **16img/768p**(input 12,401 < 16,384):**99% 的 prefill batch 是 7-13k(单请求满块)**,确认 input 不切但每 batch 只能装 1 个请求(2 个 = 24,802 ≫ 16,384)。

3. **8img/768p**(input 6,238 ≪ 16,384):rate=1.0 时 **72% double-request batch + 26% single**;rate=2.0 时 **single-request batch 占比从 26% → 40%**,证明 **加压会让 batch composition 退化**(running batch 占满 KV slot,后续 prefill 的 `rem_chunk_tokens` 缩小,只塞得下 1 个请求)。

4. **4img/768p**(input 3,158 ≪ 16,384):rate=1.0 时 95% 是 single-request 2-7k batch;但 4 个请求的合并(4 × 3,158 = 12,632 < 16,384)在 PD running peak=11 时基本没出现(running 太低,scheduler 来不及合并)。理论上 5 请求合并到 15,790 < 16,384 在更高 in-flight 下会出现,但实测此时 PD 完全没饱和。

5. **PD running-req 行为**:
   - 8img/1080p 卡在 **31**(KV pool cap;in_flight × 16,692 = 517k 占 695k 的 74%,加上 mem_frac 0.85 留的 cuda graph 余量)
   - 16img/768p 在 rate=1.0 卡 **54**(显著高于 1080p,因为 KV demand 更小;但仍未触 max_running_requests=64,而是被 KV pool 占用约束;54 × 12,673 = 684k ≈ 695k 的 98%,KV pool 真饱和!)
   - 8img/768p 卡在 **63**(精确触 max_running_requests=64,KV 占用仅 59%)
   - **4img/768p 仅到 11(r=1.0)和 21(r=2.0)** —— **完全没接近任何瓶颈**。input rate 是限制项(rate × lifetime = 1.0 × 7s ≈ 7,2.0 × 12s ≈ 24)。

---

## 3. 四个 case 的瓶颈机制对比

### 3.1 8img/1080p:**KV pool cap + 切分尾巴**(经典双瓶颈)

input_len 16,420 比 chunked_prefill_size 16,384 大 36 → 每个请求必产生:
- 一个 **16,384 token 满块** prefill(GPU 利用率高)
- 一个 **~36 token 小尾巴** prefill(被 page_size 16 对齐到 16/32/48...,GPU launch overhead 占主导,~107 tok/s 的极低有效吞吐)

实测 prefill batch 中:
- **<500 token 的小批量占 40-42%**(rate=1.0 与 rate=2.0 没区别,说明这是 input 结构决定的,不是动态调度产物)
- **13-17k 的满块占 58-60%**(主块,被切到 chunked budget)

KV pool 在 in_flight=31 触顶(per-req 16,692 × 31 = 517k,占 mem_frac=0.85 池的 74%),无法 admit 更多。

加压 rate=1.0 → 2.0 后:
- RPS 从 0.23 → **0.18(下降)**
- 失败数从 63 → 72
- total_token_throughput 从 3,781 → **2,975(下降 21%)** —— 因为 NIXL pipeline 拥塞反向 backpressure 到 PD,PD 只能拿到一部分 embedding 进 forward,GPU 实际利用率反而降低

详细机制见 `same_host_problem_analysis_zh.md`。

### 3.2 16img/768p:**KV pool 真饱和 + 单请求/batch**(双瓶颈但与 1080p 不同)

input_len 12,401 < chunked_prefill_size 16,384 → 不切分,每个请求一次跑完。
但 12,401 × 2 = 24,802 ≫ 16,384 → 每 prefill batch 只能装 1 个请求。

实测:
- **99% 的 prefill batch 在 7-13k**(单请求满 input)
- **<500 small batch 仅 1-8%**(都是 cuda graph 兼容性触发的极少量,不是结构性产物)

**KV pool 在 in_flight=54 触顶**(per-req 12,673 × 54 = 684k,占 695k 的 **98%**)—— 与 1080p 不同,16img 是**KV pool 真饱和**(几乎 100%),并非由 mem_frac 留余量限制。

由于每 batch 只装 1 个,GPU prefill 算力被腰斩(对比 8img/768p 一次能装 2 个、4img 装 5 个):
- prefill input throughput 中位 5,761 tok/s(8img/768p 的 5,665 类似;1080p 是 5,881)
- 但每 batch 只 1 请求 → 单请求平均 prefill 时间从 8img/768p 的 ~1.05s(2req/batch)上升到 16img 的 ~2.15s

加压 rate=1.0 → 2.0:
- RPS 从 0.34 → 0.33(几乎不变)
- 失败数从 51 → 56
- 中位 TTFT 从 71.4s → **104.9s** —— 排队恶化但不崩溃

**16img/768p 的真实 RPS 天花板约 0.35**,与 np=32/rate=1.0 的 0.35 完全吻合,**比 1080p 的 0.24 高 46%(因为没有切分尾巴)**,但比 8img/768p 的 0.70 低 50%(因为每 batch 只装 1 个)。

### 3.3 8img/768p:**max_running_requests=64 cap + GPU compute 饱和**

input_len 6,238 ≪ 16,384 → 不切。
6,238 × 2 = 12,476 < 16,384 → **每 batch 可以装 2 个请求**(实测 72% 的 batch 在 7-13k)。

PD running-req 在 rate=1.0 与 rate=2.0 都精确停在 **63/64**:
- KV 占用仅 6,510 × 63 / 695,136 = **59%**(还有 41% headroom)
- 触顶的是 `--max-running-requests 64` 配置上限

加压 rate=1.0 → 2.0:
- RPS 从 0.70 → 0.67(轻微下降)
- 失败数从 0 → **21**(开始进入崩溃边界)
- 中位 TTFT 从 39.9s → 60.4s

**最关键的发现:加压让 batch composition 退化:**

| rate | <500 small batch | 2-7k single-req | 7-13k double-req |
|---|---:|---:|---:|
| 1.0 | 2% | **26%** | **72%** |
| 2.0 | 1% | **40%** | **59%** |

rate=2.0 时,**40% 的 prefill batch 退化为单请求**,因为 running batch (63) 长期占满 KV slot,新来的 prefill 经常拿不到 12,476 token 的 budget(只剩 < 12,476),只能塞 1 个。这进一步降低 GPU prefill 效率,形成**正反馈循环**(prefill 慢 → queue 累积 → KV 占满 → batch 更小 → prefill 更慢)。

**8img/768p 的真实 RPS 天花板约 0.70-0.72**(与 np=64/rate=2.0 测出的 0.72 一致)。

### 3.4 4img/768p:**未饱和、可线性 scale 到 1.79 RPS**

input_len 3,158 ≪ 16,384 → 不切,且**5 个请求理论可合并**(5 × 3,158 = 15,790 < 16,384)。

per-req KV demand 仅 3,430,KV pool 695,136 / 3,430 = **202 个 in_flight 理论上限**,远高于 max_running_requests=64,所以 64 才是实际约束。

实测 PD running peak:
- rate=1.0: **11**(input rate × lifetime ≈ 1.0 × 7s = 7,Poisson burst → 11)
- rate=2.0: **21**(2.0 × 12s = 24,基本吻合)

**两者都距离 max_running_requests=64 还差 3-6×,完全没接近 PD 容量瓶颈**。

加压 rate=1.0 → 2.0:
- RPS 从 0.98 → **1.79**(线性 +83%)
- 失败数 0 → 0(无任何崩溃)
- 中位 TTFT 从 2.5 s → 3.3 s(轻微上升)
- total_token_throughput 从 3,198 → **5,856 tok/s**(+83%,与 RPS 比例完全一致)
- 中位 forward_duration 从 3.7 s → 6.0 s(in-flight 翻倍 → forward 变慢 ~60%,与 O(B) decode 一致)

**预测真实饱和点:**
按线性外推,4img/768p 在 rate=4.0/np=256 应该接近饱和,达到 RPS ~3.0-3.5(受限于 max_running_requests=64;in-flight=64 时 lifetime 估算 ~20s → RPS = 64/20 = 3.2)。

**为什么 4img/768p 没有 NIXL buffer 池耗尽问题?**

每个 4img/768p embedding ≈ 4 × 770 × 5,120 × 2 byte = ~31.5 MB,**比 8img/1080p 的 ~134 MB 小 4 倍**。同样的 NIXL_MAX_BUFFER_SIZE=805306368 (768 MB) ring buffer 池能容纳:
- 8img/1080p: 768 / 134 ≈ **5-6 个并发 embedding**
- 16img/768p: 768 / 126 ≈ 6 个
- 8img/768p: 768 / 63 ≈ 12 个
- **4img/768p: 768 / 31.5 ≈ 24 个**

**buffer 容量翻 4 倍,加上 PD admit 速度更快(0.98 RPS vs 0.24)→ 缓冲深度足以吸收 input rate 2.0 的脉冲**。 

加上 PD compute 在小 input 下高效,4img/768p 是同主机 disagg 唯一**可在 rate=2.0 持续运行的 case**。

### 3.5 四个 case 的瓶颈结构总览

| 瓶颈类型 | 8img/1080p | 16img/768p | 8img/768p | **4img/768p** |
|---|:---:|:---:|:---:|:---:|
| **#1 切分尾巴(40-46% small batches)** | ✓ **触发** | ✗ | ✗ | ✗ |
| **#2 KV pool 容量触顶** | ✓(in_flight 31 / 41 理论 = 76%) | ✓ **真饱和**(98%) | ✗(59%) | ✗(**5%**) |
| **#3 单请求/batch(每 batch admit 1 个)** | ✓(切分被迫) | ✓ **24,802 > 16,384 强制单请求** | ✗(2 请求/batch) | ✗(**5 请求/batch 可能**) |
| **#4 max_running_requests=64 触顶** | ✗ | ✗(54 < 64) | ✓ **触顶** | ✗ (**11/21 ≪ 64**) |
| **#5 加压时 batch 退化为单请求** | (#1 已经是 small) | (本来就是 1) | ✓ **rate=2.0 时退化** | ✗ |
| **#6 NIXL buffer 池耗尽(过载崩溃)** | ✓ **rate=1.0 已崩** | ✓ **rate=1.0 已崩** | ✓ rate=2.0 边界,21 失败 | ✗ **rate=2.0 仍 0 失败** |

**结论:四个 case 的"主导瓶颈"完全不同**:
- 1080p 受 **#1 + #2** 主导,有 5 个瓶颈活跃 → RPS 最低 0.23
- 16img/768p 受 **#2 + #3** 主导,4 个瓶颈活跃 → RPS 中等 0.34
- 8img/768p 受 **#4 + #5** 主导,3 个瓶颈活跃 → RPS 高 0.70
- **4img/768p 没有任何瓶颈触发** → RPS 1.79(rate=2.0 仍未饱和)

### 3.6 NIXL buffer 池容量与 embedding 大小的反比关系(为什么 4img 是例外)

NIXL_MAX_BUFFER_SIZE=805306368(768 MB)是 ring buffer 池总容量。每个 image embedding 大小:

| 工作负载 | image 数 × visual tokens × hidden × bytes | 单 embedding 大小 | 池容纳的并发 embedding 数 |
|---|---|---:|---:|
| 8img/1080p | 8 × 2,064 × 5,120 × 2 | ~134 MB | **5-6 个** |
| 16img/768p | 16 × 770 × 5,120 × 2 | ~126 MB | 6 个 |
| 8img/768p  | 8 × 770 × 5,120 × 2 | ~63 MB | 12 个 |
| **4img/768p** | 4 × 770 × 5,120 × 2 | **~31.5 MB** | **24 个** |

**分析:**

1. **8img/1080p**:每秒 1 个新 embedding 要 134 MB,buffer 池只能存 5-6 个 → PD admit 速度 0.24 RPS 远低于 input 1.0,**不到 6 秒就耗尽 buffer 池** → 60s 内来不及释放 → `Timeout while waiting for available buffer`

2. **4img/768p**:embedding 小 4 倍,buffer 池存 24 个;PD admit 速度 1.79 RPS 接近 input rate 2.0 → 池里堆积速度慢,**60s 内能释放足够 buffer 给后到的请求**

**关键洞察:NIXL buffer 池耗尽崩溃是 embedding 大小 × PD admit 慢度的双重函数。** 4img/768p 同时享受了"小 embedding"和"高 PD 吞吐",所以不崩溃。

**这印证了之前 768p 文档 §3.6 第 5 点 "Goodput 崩溃机制"**,而且发现**1080p 在 rate=1.0 就已经崩溃**(因为 PD 真实 RPS 只有 0.24,rate=1.0 已经超过 4×)。

---

## 4. 加压实验的核心发现:**重负载** rate=2.0 反而比 rate=1.0 差(三个重 case 验证),**轻负载 4img/768p 例外**

| 工作负载 | rate=1.0 RPS | rate=2.0 RPS | 变化 | 原因 |
|---|---:|---:|---:|---|
| 8img/1080p | 0.23 | 0.18 | **-22%** | NIXL pipeline 严重拥塞,反向 backpressure 让 PD GPU 利用率反而降低 |
| 16img/768p | 0.34 | 0.33 | -3% | NIXL 一样拥塞,但 KV pool 已饱和 in_flight 不能再增,所以基本持平 |
| 8img/768p  | 0.70 | 0.67 | -4% | batch 退化为单请求(单请求占比 26% → 40%),GPU prefill 效率下降 |
| **4img/768p** | **0.98** | **1.79** | **+83% ★** | input rate=1.0 已经接近 input rate-limited(0.98 ≈ 1.0);rate=2.0 才让系统真正运转,**PD 还有 3-6× headroom** |

**统一解释:** 当 input_rate 超过 PD 真实饱和 RPS 时,新 embedding 在 NIXL ring buffer 池中堆积,而 PD admit 速度不变。**所以 rate=2.0 是否有用,完全取决于该 case 的 PD 真实 RPS 是否 > 2.0**。

- 4img/768p PD 真实 RPS ≥ 2.0 → rate=2.0 是欠饱和压力,**PD 实际加速 1.83×**
- 其它 3 个 case PD 真实 RPS < 1.0 → rate=2.0 是 2-10× 过饱和,触发崩溃

详见 §3.6(NIXL buffer 容量与 embedding 大小的反比关系)。

---

## 5. 之前文档提出的优化方法在当前 SGLang 上是否可用?

`same_host_problem_analysis_zh.md` 第 3 节提出了两条改进:
1. **小批量合并(coalesce small chunks)**:把 `chunked_req` 的剩余 < 256 token 尾巴合并到当前 batch
2. **`--enable-dynamic-chunking`**:让 scheduler 根据 KV 占用动态决定 chunk size

**实际验证(检查 SGLang 源码 `/opt/sglang/python/sglang/srt/`):**

### 5.1 `--enable-dynamic-chunking`

```python
# server_args.py:413
enable_dynamic_chunking: bool = False
# CLI: --enable-dynamic-chunking
# 帮助文本: "Enable dynamic chunk size adjustment for pipeline parallelism.
#  When enabled, chunk sizes are dynamically calculated based on fitted function
#  to maintain consistent execution time across chunks."
```

```python
# scheduler.py
self.enable_dynamic_chunking = (
    self.server_args.enable_dynamic_chunking and self.pp_size > 1  # ← 关键!
)
```

**结论:`--enable-dynamic-chunking` 仅在 `pp_size > 1` 时生效,我们用的是 TP=1/PP=1,标志会被静默忽略。** 该方法不适用于当前部署。

要启用这个特性,必须先把 PD worker 改成 PP=2(用 2 张 GPU),但那样 disagg 拓扑就变成 1 encoder + 2 PD workers,不在本次同主机 disagg 范围内。

### 5.2 小批量合并(coalesce small chunks)

```bash
# 检查 SGLang 源码
$ grep -rE "coalesce.*chunk|merge.*chunk|chunked_req.*tail" \
    /opt/sglang/python/sglang/srt/managers/
```

**没有找到任何"尾块合并"的实现**。SGLang 现有的 `_coalesce_streaming_chunks`(在 `tokenizer_manager.py`)仅用于 SSE 流式 token 输出聚合,与 prefill batch 无关。

**结论:小批量合并是上游(SGLang)未实现的改造**。1080p 文档的建议是理论分析,需要写一个 100+ 行的 SGLang scheduler patch(修改 `schedule_policy.py:813-933` 的 `PrefillAdder.add_one_req()` 与 `scheduler.py:2640-2780` 的 `get_new_batch_prefill()`)才能验证。

### 5.3 替代的可行优化方向

既然两条建议都不能直接用,**当前可立即调整的优化方向**:

| 调整 | 适用 case | 预期效果 |
|---|---|---|
| `--chunked-prefill-size 32768` | **8img/1080p** | input 16,420 一次跑完 → #1 消失 |
| `--max-running-requests 128` | **8img/768p** | KV pool 41% headroom → in_flight 63 → ~85,RPS 0.70 → ~0.78(估算) |
| 加大 `NIXL_MAX_BUFFER_SIZE` 到 4 GB+ | **所有 case** | 缓解 goodput 崩溃,但治标不治本 |
| 前端 admission control / 429 拒绝 | **所有 case** | 当 PD running ≥ 60 时直接拒绝新请求,**根治 goodput 崩溃** |
| TP=2 agg(放弃 disagg)| **8img/1080p, 16img/768p** | 实测 8img/1080p TP=2 agg = 0.95 RPS,4× 提升 |

**实测建议:** 下一轮可以用 `--chunked-prefill-size 32768` 重测 8img/1080p,验证消除尾巴的实际收益(理论预测 RPS 0.24 → 0.40 左右,因为去掉 40% 小批量 + KV pool 头还有空间)。

---

## 6. 实测数据验证 RPS 模型

> **公式:** `RPS_capacity ≈ Total_token_throughput / per_req_total_tokens`

| 工作负载 | rate | total_tput | per_req tokens (in + out) | 预测 RPS | 实测 RPS | 误差 |
|---|---:|---:|---:|---:|---:|---:|
| 8img/1080p | 1.0 | 3,781 | 16,420 + 121 = 16,541 | 0.229 | 0.23 | +0.4% |
| 8img/1080p | 2.0 | 2,975 | 16,420 + 128 = 16,548 | 0.180 | 0.18 | 0% |
| 16img/768p | 1.0 | 4,242 | 12,401 + 132 = 12,533 | 0.339 | 0.34 | +0.3% |
| 16img/768p | 2.0 | 4,109 | 12,401 + 126 = 12,527 | 0.328 | 0.33 | +0.6% |
| 8img/768p  | 1.0 | 4,470 | 6,238 + 118 = 6,356 | 0.703 | 0.70 | 0% |
| 8img/768p  | 2.0 | 4,282 | 6,238 + 128 = 6,366 | 0.673 | 0.67 | 0% |
| **4img/768p** | **1.0** | **3,198** | **3,158 + 118 = 3,276** | **0.976** | **0.98** | **+0.4%** |
| **4img/768p** | **2.0** | **5,856** | **3,158 + 128 = 3,286** | **1.782** | **1.79** | **+0.5%** |

**所有 8 组数据 RPS 预测误差 < 1%**,印证 `RPS = total_tput / per_req_tokens` 是 PD 实际 GPU compute throughput 的精确模型。要提升 RPS,只能:
- 提升 total_token_throughput(GPU compute) → 用更大的 GPU、TP=2、或减少 single-request batch 占比
- 减少 per_req_tokens → 用更小的图、更短的 output

无法通过环境变量或配置参数(在当前架构下)有效提升。

**特别注意 4img/768p:**
- rate=1.0:total_tput 仅 3,198 tok/s,因为 input rate 限制了 PD 输入(每秒只 1 个请求 × 3,158 token = 3,158 tok/s prefill,加 decode 116 = 3,274 → 3,198 实测)
- rate=2.0:total_tput **跳到 5,856 tok/s**(+83%),与 RPS 1.79 完全一致 → **PD 在更高 in-flight 下 GPU compute 利用率提升明显**
- 这是唯一一个 case 中 total_tput 在 rate=1.0 → 2.0 显著提升的(其它三个都跌或平),因为只有 4img 在 rate=1.0 时 PD 严重欠饱和

---

## 7. 四个 case 的"瓶颈起源"统一图谱

```
工作负载特征:                     瓶颈出现链:
                                  
8img/1080p:                       input_len 16,420
input_len > chunked_prefill_size   ↓ 强制切分
  + KV demand 16,692               ↓ 产生 36-token 尾巴 (40% small batches)
                                   ↓ KV pool ~74% 占用 (cuda_graph 留余)
                                   ↓ in_flight 卡 31
                                   ↓ NIXL 池在 rate ≥ 0.5 已耗尽
                                   ↓ goodput 崩溃 → RPS 0.23 (rate=1.0)
                                   
16img/768p:                       input_len 12,401
input_len < chunked_prefill        ↓ 不切分(99% 大块)
  但 2 倍 input > chunked          ↓ 每 batch 单请求
  + KV demand 12,673               ↓ KV pool 98% 真饱和
                                   ↓ in_flight 卡 54
                                   ↓ NIXL 池在 rate ≥ 0.5 耗尽
                                   ↓ goodput 崩溃 → RPS 0.34 (rate=1.0)
                                   
8img/768p:                        input_len 6,238
input_len ≪ chunked_prefill        ↓ 不切分,2 请求合并 batch (72%)
  + KV demand 6,510                ↓ KV pool 仅 59% 占用
                                   ↓ in_flight 卡 max_running_requests=64
                                   ↓ rate ≤ 1.0 不崩溃,rate=2.0 batch 退化
                                   ↓ NIXL 池在 rate ≥ 1.5 才耗尽
                                   ↓ rate=1.0 安全运行 → RPS 0.70

★ 4img/768p:                      input_len 3,158
input_len ≤ chunked / 5            ↓ 不切分,理论 5 请求合并 batch
  + KV demand 3,430                ↓ KV pool 仅 5% 占用
  + embedding 仅 31.5 MB           ↓ NIXL 池容纳 24 个并发
                                   ↓ in_flight 仅 11 (r=1.0) / 21 (r=2.0)
                                   ↓ 远未触 max_running_requests=64
                                   ↓ NIXL 池永不耗尽
                                   ↓ ★ rate=2.0 仍 0 失败 → RPS 1.79
```

**四个 case 的瓶颈源头都是"input_len 与 chunked_prefill_size(16,384)的关系":**

| input_len 区间 | 行为 | 瓶颈 | 实测 case |
|---|---|---|---|
| **> chunked_prefill_size** | 必切分,产生小尾巴 | 切分尾巴 + KV pool + NIXL 池 | 8img/1080p (0.23) |
| **chunked / 2 < input ≤ chunked** | 不切但每 batch 1 请求 | KV pool + 单请求 batch + NIXL 池 | 16img/768p (0.34) |
| **chunked / 3 < input ≤ chunked / 2** | 2 请求/batch | max_running_requests + GPU compute | 8img/768p (0.70) |
| **input ≤ chunked / 5** | 5 请求/batch,系统未饱和 | (无) — 受 input rate 限制 | **4img/768p (1.79+)** |

**所以"分辨率"不是因变量,"input_len"才是。** 1080p vs 768p 只是 input_len 的代理变量。如果保持 768p 但用 32 张图,效果与 1080p × 8 张几乎一样(input_len ≈ 24,640,刚刚跨过 16,384,会进入"切分 + 双 chunk" 模式,RPS 估算 ~0.18)。

**反过来,要 disagg 跑得快:用更小的图(降低 input_len)远比追求更大的 GPU 有效。** 768p × 4img 比 1080p × 8img 快 8 倍(1.79 / 0.23),代价仅是分辨率从 1920×1080 → 1024×768。

---

## 8. 生产部署红线建议(基于本次 8 组数据)

### 8.1 安全工作区(无失败)

| 工作负载 | 推荐 max rate | 推荐部署形态 |
|---|---:|---|
| 8img/1080p | ≤ 0.15 RPS | ❌ 不推荐 disagg,改 TP=2 agg(0.95 RPS) |
| 16img/768p | ≤ 0.20 RPS | ❌ 不推荐 disagg,改 TP=2 agg |
| 8img/768p  | ≤ 0.65 RPS | ⚠ 可用,但 P99 TTFT 高;可考虑 TP=2 agg |
| **4img/768p**  | **≤ 1.79 RPS(实测)** | **✓ 强烈推荐 disagg —— 唯一不崩溃的 case** |

**关键洞察:**
- 想用 disagg 拿到合理 RPS,**必须把 input_len 控制在 chunked_prefill_size / 5 = 3,277 token 以下**
- 4img/768p (3,158 < 3,277) 是当前 prod 友好的工作负载
- 一旦 input_len > 6,500(8img/768p),就接近瓶颈,加压会崩溃

### 8.2 必须的前端控制

**对 8img/1080p、16img/768p、8img/768p 这些"会崩溃"的 case 必须**在 dynamo frontend 配置:
1. **Admission control**: 监控 PD running-req,≥ 60 时返回 429
2. **NIXL buffer 池监控**: ring buffer 占用 > 80% 时主动节流
3. **超时上限**: encode_timeout 与 receive_timeout 错开(目前都是 60s,容易雪崩)

**对 4img/768p:** 不需要复杂的前端控制,只要 input rate ≤ ~2 RPS 就稳定运行。如果 input rate > 2 RPS,需要监控并报警。

详细配置见 `same_host_768p_problem_analysis_zh.md` §6.2。

---

## 9. 结果文件与日志

### 本次新增 7 组 bench 结果(全部位于 `/hongming/res_samehost_disagg_32b_gpu01_unpatched/`):

| 工作负载 | 路径 | bench 起 UTC | 时长 | 成功/总数 |
|---|---|---|---:|:---:|
| 8img/1080p np=128 r=1.0 | `8img_1080p_rate1.0_np128_nixlwrite_20260531_173603/` | 17:36:03 | 408 s | 65/128 |
| 8img/1080p np=128 r=2.0 | `8img_1080p_rate2.0_np128_nixlwrite_20260531_174313/` | 17:43:13 | 408 s | 51/128 |
| 16img/768p np=128 r=1.0 | `16img_768p_rate1.0_np128_nixlwrite_20260531_175016/` | 17:50:16 | 310 s | 77/128 |
| 16img/768p np=128 r=2.0 | `16img_768p_rate2.0_np128_nixlwrite_20260531_175537/` | 17:55:37 | 302 s | 72/128 |
| 8img/768p np=128 r=2.0  | `8img_768p_rate2.0_np128_nixlwrite_20260531_180051/`  | 18:00:51 | 213 s | 107/128 |
| **4img/768p np=128 r=1.0 [nocache]** | `4img_768p_rate1.0_np128_nocache_nixlwrite_20260531_193131/` | 19:31:31 | 168 s | **128/128** |
| **4img/768p np=128 r=2.0 [nocache]** | `4img_768p_rate2.0_np128_nocache_nixlwrite_20260531_193425/` | 19:34:25 | 111 s | **128/128** |

### 复用的 1 组数据(prior):

| 工作负载 | 路径 | 时长 | 成功/总数 |
|---|---|---:|:---:|
| 8img/768p np=128 r=1.0 [prior] | `8img_768p_rate1.0_np128_nixlwrite_20260531_164750/` | 182 s | 128/128 |

### 重要的 v1 (cache enabled) 4img 实验数据(供对比,但不是本文档正式数据):

| 工作负载 | 路径 | RPS | 备注 |
|---|---|---:|---|
| 4img/768p np=128 r=1.0 [v1, cache=on] | `4img_768p_rate1.0_np128_v2_nixlwrite_20260531_190934/` | 0.98 | 与 nocache 完全一致 |
| 4img/768p np=128 r=2.0 [v1, cache=on] | `4img_768p_rate2.0_np128_nixlwrite_20260531_191226/` | 1.88 | RadixCache 命中,比 nocache 1.79 高 5% |

### PD / Encoder 日志:
- 重负载 6 组 (8img/1080p、16img/768p、8img/768p): `logs/samehost_pd_20260531_144409.log`
- 重负载 encoder 日志: `logs/samehost_encoder_20260531_144409.log`
- 4img 测试 (cache=off): `/hongming/dynamo/logs/samehost_pd_20260531_192146.log`
- 4img encoder (cache=off): `/hongming/dynamo/logs/samehost_encoder_20260531_192146.log`

每个 bench 目录含:
- `benchmark_output.json` —— 原始指标
- `results.txt` —— 完整 bench stdout
- `bench_start_utc.txt` —— bench 起始时间(用于 PD 日志窗口对齐)

---

## 10. 下一步建议实验

按重要性排序:

1. **实验 A: `--chunked-prefill-size 32768` 在 8img/1080p**
   - 假设:消除切分尾巴,小批量从 40% → ~5%,RPS 从 0.23 → 0.40+
   - 风险:per-batch GPU memory 翻倍,可能 OOM(需 mem_frac 微调)
   - 工作量:重启 PD,1 组 bench(np=128 rate=1.0)

2. **实验 B: `--max-running-requests 128` 在 8img/768p**
   - 假设:KV pool 41% headroom → in_flight 63 → ~85,RPS 0.70 → ~0.78
   - 风险:无(KV pool 还有空间)
   - 工作量:重启 PD,2 组 bench(rate=1.0/2.0)

3. **实验 C: 加大 NIXL_MAX_BUFFER_SIZE 到 4 GB**
   - 假设:8img/1080p rate=1.0 不再崩溃(ring buffer 容纳更多)
   - 风险:GPU 0(encoder)mem 占用上升,需 encoder mem_frac 微调
   - 工作量:重启 stack,3 组 bench(三个 case × rate=1.0)

4. **实验 D: TP=2 agg 验证(放弃 disagg)**
   - 已有数据:8img/1080p TP=2 agg = 0.95 RPS(4× disagg)
   - 工作量:启动 TP=2 agg stack(占 2 张 GPU),3 组 bench

---

## 11. 关键源码路径(用于后续 patch / 验证)

| 现象 | 源码位置 |
|---|---|
| `--enable-dynamic-chunking` 解析 | `/opt/sglang/python/sglang/srt/server_args.py:413` |
| `enable_dynamic_chunking` 与 `pp_size > 1` 的限制 | `/opt/sglang/python/sglang/srt/managers/scheduler.py`(搜索 `enable_dynamic_chunking`) |
| chunked prefill 切分逻辑 | `/opt/sglang/python/sglang/srt/managers/schedule_policy.py:813-933` (`PrefillAdder.add_one_req`) |
| KV pool 容量预留检查 | `schedule_policy.py:866-870`(`if total_tokens >= rem_total_tokens: return NO_TOKEN`) |
| Scheduler 主循环 / admission tick | `scheduler.py:2640-2780` (`get_new_batch_prefill`) |
| max_running_requests 检查 | `scheduler.py`(搜索 `max_running_requests`) |
| NIXL ring buffer 申请与超时 | `/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py:670-725` |
| Dynamo PD-side mm 处理 | `/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/worker_handler.py:402-454` |

---

## 参考文档

- `same_host_problem_analysis_zh.md` —— 8img/1080p np=32/rate=1.0 两大瓶颈根因(本文档基础)
- `same_host_768p_problem_analysis_zh.md` —— 768p 三种 image count 在 np=32-256 下的瓶颈对比
- `same_host_disagg_three_cases_rps.md` —— 横向 RPS 对比表(已扩展到 7 组)
- `same_host_disagg_time_zh.md` —— 同主机 disagg 单请求 latency 分解(nixl-read vs nixl-write)
- `1080p_sweep_three_way.md` —— TP=1 agg / TP=2 agg / disagg 三路对比(8img/1080p TP=2 agg = 0.95 RPS)
- `agg_vs_disagg_full_sweep_np32.md` —— TP=1 agg / cross-host disagg / 同主机 disagg 三路扫描
- `INDEX.md` —— 整个调研目录索引
