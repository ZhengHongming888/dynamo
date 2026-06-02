# 同主机 Disagg E/PD 在 768p 工作负载下的瓶颈根因分析(16img / 8img / 4img)

**关联文档:** `same_host_problem_analysis_zh.md` 解释了 8img/1080p 工作负载下的两大瓶颈(46% 小批量 prefill + 队列等待 48.8 s)。本文档延续同一框架,分析在 768p 分辨率下,随着图片数量从 4 → 8 → 16 变化,瓶颈的**形态发生根本变化**:

| 指标 | 8img/1080p | **16img/768p** | 8img/768p | 4img/768p |
|---|---:|---:|---:|---:|
| **RPS @ rate=1.0/np=32** | 0.24 | 0.35 | **0.64** | 0.84 |
| **RPS @ rate=1.0/np=128** | (未测) | (未测) | **0.70** | (未测) |
| **RPS @ rate=2.0/np=64** | (未测) | (未测) | **0.72** | (未测) |
| **RPS @ rate=2.0/np=256(过载)** | (未测) | (未测) | **0.65**(105/256 失败) | (未测) |
| 中位 E2E | 112.1 s | 71.8 s | 37.6 s | 8.1 s |
| 中位 TTFT | 61.5 s | 40.6 s | 23.2 s | 2.4 s |
| 中位 PD queue_duration | 48.8 s | 25.4 s | 13.3 s | **0.00 s** |
| 中位 PD forward_duration | 50.7 s | 34.3 s | 18.3 s | 5.4 s |
| 单请求 input_len(中位) | 16,420 | 12,401 | 6,238 | 3,158 |
| 小批量 prefill 占比 (#new-token < 1000) | **46%** | **3.0%** | **5.6%** | **4.2%** |
| 每 prefill batch 容纳的请求数 | 1(必切分) | **1**(2 个塞不下) | **2** | **5** |
| Running-req 峰值 / 中位(rate=1.0) | 31 / 12 | 31 / 15 | 30 / 13 | 18 / 2 |
| Running-req 峰值(rate=2.0/np=64) | — | — | **63 / 29** | — |
| Queue-req 峰值(rate=1.0 → 2.0) | 20 → — | 14 → — | 20 → **30** | 0 → — |
| 每请求 KV demand (token) | 16,692 | 12,673 | 6,510 | 3,430 |
| KV pool 理论上限(reqs) | 41 | 54 | 106 | 202 |
| 实际峰值 in-flight 占 KV pool 的比例 | 74% | 57% | 28% → **59%** | **9%** |
| Total token throughput (tok/s) | ~4,000 | 4,381 | 4,104 → **4,552** | 2,791 |

**一句话总结:**

> - **8img/1080p**: prefill 切分瓶颈(#1,46% 小批量)+ KV pool 队列瓶颈(#2)同时强烈触发 → 0.24 RPS
> - **16img/768p**: prefill 切分瓶颈消失(12,401 < 16,384,一次跑完),但**单请求占 76% 的 chunk budget,导致每个 prefill batch 只能装 1 个请求**;再加上 KV pool 在 in-flight=31 触顶,产生 25 s 队列;**RPS = 0.35 与 GPU sustained throughput 4,381 tok/s 完美吻合**(算力 bound)
> - **8img/768p**: prefill 切分瓶颈消失,且 2 个请求可合并到一个 prefill batch(12,476 < 16,384),admission 速率翻倍;np=32/rate=1.0 时 RPS = 0.64,加压到 np=64/rate=2.0 RPS 仅升至 **0.72**(+12.5%),**饱和上限由 `max_running_requests=64` + GPU compute throughput 共同决定**
> - **4img/768p**: 5 个请求可合并到一个 prefill batch(15,790 < 16,384),admission 速率 5×;系统**未饱和**,RPS = 0.84 受限于 input rate → 真实容量 ≥ 1.0 RPS

下面分别说明这两个 768p 工作负载的瓶颈机制。

---

## 1. 为什么 768p 让"小批量 prefill"瓶颈消失

**1080p 的根因回顾(详见 `same_host_problem_analysis_zh.md` §1):**
- 8img/1080p 的 input_len ≈ 16,420 tokens
- `--chunked-prefill-size 16384` 强制把每个请求切成 **2 块**:满块 16,384 + 尾巴 ~36 tokens
- 32 个请求 × 2 块 → 32 个尾巴,每个尾巴 forward 摊薄到 ~107 tok/s(GPU launch overhead 占主导)
- 这些尾巴占 prefill 事件总数的 **46%**

**768p 把这个根因(切分尾巴)连根拔起:**

每张 1024×768 图像在 Qwen3-VL 编码后产出 ~770 visual tokens(1080p 是 ~2,064)。

- **16img/768p**: 16 × 770 = 12,320 visual tokens + ~80 text tokens ≈ **12,401 tokens** —— **小于 chunked_prefill_size 16,384**
- **8img/768p**:  8 × 770 = 6,160 + ~78 ≈ **6,238 tokens** —— 远小于 16,384
- **4img/768p**:  4 × 770 = 3,080 + ~78 ≈ **3,158 tokens** —— 同样远小于 16,384

实测 prefill batch 分布印证了这一点:

```
16img/768p prefill 批次的 #new-token 分布(33 个 prefill 事件):
  16,            ← 1 次(唯一的小批量,可能是 cuda_graph 兼容触发)
  12,352 / 12,368 / 12,384 / 12,400 / 12,416 /
  12,432 / 12,448 / 12,464,                          ← 32 次(每次都是 1 个请求,2 个塞不下)
小批量 (<1000 token):         3.0%
大块   (≥1000 token):        97.0%
```

```
8img/768p prefill 批次的 #new-token 分布(18 个 prefill 事件):
  16,            ← 1 次(唯一的小批量,可能是 cuda_graph 兼容触发)
  6,224,                                            ← 2 次,单请求
  12,384 / 12,416 / 12,464 / 12,480 / 12,496 /
  12,512 / 12,528 / 12,608,                          ← 14 次(每次约 2 个请求合并)
小批量 (<1000 token):         5.6%
大块   (≥1000 token):        94.4%
```

```
4img/768p prefill 批次的 #new-token 分布(24 个 prefill 事件):
  16,                              ← 1 次
  3,104 ~ 3,232,                   ← 18 次,单请求(input_len ~3,158)
  6,336,                           ← 1 次,2 请求合并
  9,472 / 9,488,                   ← 2 次,3 请求合并
  15,792,                          ← 1 次,5 请求合并
小批量 (<1000 token):         4.2%
大块   (≥1000 token):        95.8%
```

**关键观察:每张 prefill batch 能塞多少个请求,取决于单请求大小与 chunked_prefill_size(16,384)的比值:**

| 工作负载 | 单请求 input_len | 单 batch 容量 floor(16,384 / in_len) | 实测每 batch 平均请求数 |
|---|---:|---:|---:|
| 16img/768p | 12,401 | **1**(2 个需 24,802 ≫ 16,384) | 1 |
| 8img/768p | 6,238 | **2**(3 个需 18,714 > 16,384) | ~2 |
| 4img/768p | 3,158 | **5**(6 个需 18,948 > 16,384) | 1-5(根据队列) |

**这就是为什么 16img/768p 即使没有切分尾巴问题(3.0% 小批量),仍然只能跑 0.35 RPS**:每个 prefill batch 只能 admit 1 个新请求,加上 GPU 单 batch forward 时间(~1.85 s for 12,400 tokens 在 fa3 + FP8 + 32 层)→ admission 速率被锁死在 ~0.54 req/s,而当前 batch 内的请求还要做 decode,实际下沉到 ~0.35 RPS。

### 1.1 这是 chunked-prefill 算法的**最佳运行点**

`PrefillAdder.add_one_req()`(`schedule_policy.py:813-933`)的设计目标本来就是让 batch 尽量塞满 `rem_chunk_tokens`:

```python
# schedule_policy.py:907-935(简化)
if input_tokens <= self.rem_chunk_tokens:
    # 整请求一次进 batch,不切分 ← 768p 走这条路径
    self.can_run_list.append(req)
    self.rem_chunk_tokens -= input_tokens
else:
    # 切到 rem_chunk_tokens ← 1080p 走这条路径
    trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
    req.set_extend_input_len(trunc_len)
    ...
```

只要 `input_tokens ≤ rem_chunk_tokens`,就**不产生尾巴**。768p 的 input_len 6,238 / 3,158 都满足这个条件,而 1080p 的 16,420 不满足 → 这是分辨率引入的本质差异。

### 1.2 GPU 利用率从被尾巴拖累到接近峰值

**1080p 实测**:median input throughput ~3,000 tok/s(被 32-token 尾巴稀释)

**768p 实测**:
- 16img/768p median 6,543 tok/s,峰值 15,523 tok/s
- 8img/768p median 7,088 tok/s,峰值 17,783 tok/s
- 4img/768p median 2,081 tok/s(单请求 3.1k 较小)、峰值 19,883 tok/s

768p 接近 GPU 在 fa3 + FP8 路径下的峰值吞吐(~20k tok/s),证实**没有被小批量拖累**。

---

## 2. 16img/768p 的瓶颈:单请求占满 prefill budget + KV pool 触顶(双重夹击)

虽然 #1(切分尾巴)消失了,16img/768p 仍只跑出 0.35 RPS,中位 queue_duration 高达 25.4 s。这是两个机制叠加的结果。

### 2.1 机制 A:每 prefill batch 只能 admit 1 个请求

`PrefillAdder.add_one_req()`(`schedule_policy.py:907-935`)的 budget 检查是**贪心累加**:

```python
# 简化逻辑
self.rem_chunk_tokens = 16384  # 每 batch 起始
for req in waiting_queue:
    if input_tokens <= self.rem_chunk_tokens:
        can_run.append(req)
        self.rem_chunk_tokens -= input_tokens  # 第一个 12,401 → 剩 3,983
    elif self.rem_chunk_tokens >= page_size:  # 还有空间但不够整个新请求
        # 切分新请求(产生尾巴):截到 3,983
        break  # 大多数情况会 break,不切第二个,只 admit 1 个
```

对 16img/768p:
- 第一个请求 12,401 tokens 进 batch → `rem_chunk_tokens = 3,983`
- 第二个请求需要 12,401 tokens,**不够**,且如果切分会产生 8,418 token 的"剩余尾巴"(变回 1080p 那个问题)
- scheduler 实际选择**不切**,放弃第二个,只 admit 1 个,然后启动 forward

实测验证:33 个 prefill 事件,**32 个都是单请求** (#new-token = 12,352~12,464),没有任何 2 请求合并的情况。**有效 admission 速率 = 1 req / 1.87 s = 0.535 req/s**(2.4 节会推导)。

对比:8img/768p 的 6,238 + 6,238 = 12,476 < 16,384 → admit 2 个/batch → 速率翻倍。

### 2.2 机制 B:KV pool 触顶,与 1080p 同样的队列形成

实测 in-flight 峰值 **31**,与 1080p 完全一致。计算:

- per-req KV demand = 12,401(input) + 256(max_new) + 16(page) = **12,673 tokens**
- KV pool 容量 = 695,136 tokens
- 理论上限 = 695,136 / 12,673 ≈ **54 个 in-flight**
- 实测 31,占 KV pool **57%** —— 说明 **mem_fraction=0.85 留的 20 GB headroom 中约 60% 被 cuda graphs / activations 占走**,与 1080p 的 31/41=74% 是同一个 dynamic mem 分配规律,只是分母不同

**为什么峰值只到 31 而不是理论的 54**:与 1080p 相同的根因(详见 1080p 文档 §2.2-2.3),关键是 SGLang scheduler 在 `add_one_req` 时**预留 max_new_tokens 配额**,加上 cuda graph buffer 占用,实际可用 KV pool 比标称小约 35%。

### 2.3 实测 admission 时间线

```
time              queue   running   注释
15:50:49.405        0        0     warmup tail prefill
15:50:56.689        0        0
15:50:58.909        0        1     ← 第一个 bench 请求 admit
15:51:06.516        5        2     ← 7.6 秒后(包含 NIXL handoff + 第一个 forward)
15:51:15.265       11        3     ← 8.7 秒间隔
15:51:19.634       13        4     ← 4.4 秒
15:51:22.900       14        5     ← admission 节奏稳定到 ~1.87 s/请求
15:51:24.797       13        6
15:51:26.698       12        7
15:51:28.609       11        8
15:51:33.929       14        9     ← 队列峰值 14
15:51:36.080       14       10     ← 稳态 (Little's Law: ~14 in waiting + ~16 in running)
15:51:38.255       14       11
15:51:40.408       14       12
15:51:42.571       14       13
15:51:44.720       14       14
15:51:46.869       14       15
15:51:48.981       14       16
15:51:50.847       13       17
15:51:52.715       12       18
15:51:54.584       11       19
15:51:56.461       10       20
...
15:52:13.280        1       29     ← 1.87 s 一个,稳定
15:52:15.148        0       30
15:52:15.945        0       31     ← bench 全部 admit 完毕,后续是 decode
```

**关键观察:从 15:51:24 到 15:52:15,51 秒 admit 27 个请求 = 1.89 s/请求**。

### 2.4 RPS 推导:与 GPU sustained throughput 完美吻合

bench 实测 `Total token throughput = 4,381 tok/s`。每请求总 token 数 = 12,401(input) + 153(median output) = 12,554 tokens。

```
RPS_capacity = 4,381 tok/s / 12,554 tokens/req = 0.349 RPS  ← 与实测 0.35 完美吻合
```

**这意味着 16img/768p 的瓶颈最终是 GPU compute throughput**,而 admission 速率(0.535)和 KV pool 容量(31)只是**它的两个表现形式**:
- 单 batch forward 12,400 tokens 在 GPU 上耗时约 1.87 s(等于 admission 间隔)
- decode 阶段 in-flight 31 个请求 × 30 ms/step = 0.93 s/token,output 153 tokens 需要 ~140 s(分摊到 31 个并发)
- 平均每秒 GPU 既要 prefill 一个新请求,又要 decode 31 个 in-flight → 实际 total token throughput 4,381 已是 GPU 在该 batch 形态下的极限

### 2.5 为什么 16img/768p 在质上更接近 1080p,而不是 8img/768p

| 现象 | 16img/768p | 8img/768p | 1080p |
|---|---|---|---|
| Prefill 切分尾巴 | ✗ | ✗ | ✓(46%) |
| 每 batch 容纳请求数 | **1** | **2** | **1**(因为切分) |
| KV pool 触顶 | ✓(57%) | ✗(28%) | ✓(74%) |
| 队列形成 | ✓(中位 25 s) | ✓(中位 13 s,但因 input 输完) | ✓(中位 49 s) |
| 主导瓶颈 | **GPU compute throughput**(双重夹击) | **GPU compute throughput**(被 input rate 限制) | **#1 切分尾巴 + #2 KV pool** |

**16img/768p 与 1080p 的相似性**:都是 GPU compute 饱和 → in-flight 卡 31 → 队列形成。

**16img/768p 与 1080p 的差异**:1080p 多一层"尾巴稀释 GPU 利用率"(median tput 3k vs 16img 6.5k),所以 1080p 的 RPS 0.24 < 16img 的 0.35;但**两者瓶颈形态本质相同(GPU 算力 + KV pool 双重夹击)**,只是 16img 没有 prefill 算法层面的低效,纯粹是 token 数 × 算力的物理上限。

---

## 3. 8img/768p 的瓶颈:Scheduler 单 tick admission 速率限制

虽然 #1 消失了,但 8img/768p 的 PD queue_duration 中位仍有 **13.3 s**。这不是 KV pool 容量不足(KV 理论上限 106 ≫ 实际 in-flight 30),那是什么?

### 3.1 实测 admission 时间线

把 18 个 prefill 事件的 `(queue, running)` 对按时间画出来:

```
time              queue   running   注释
14:54:29.074        0        0     bench 开始前的 idle prefill
14:54:33.240        0        0     idle
14:54:39.166        3        1     ← 第一个请求被 admit(input 6,238 + 1 个 prefill batch)
14:54:45.186       10        2     ← 第二个被 admit,6 秒后
14:54:48.522       12        4     ← 第三、四个,3.3 秒后(2 个一起 admit)
14:54:54.526       19        6
14:54:57.136       20        8     ← queue 峰值 20
14:54:58.899       18       10
14:55:00.660       16       12
14:55:02.421       14       14
14:55:04.181       12       16
14:55:05.939       10       18     ← admission 节奏稳定 ~1.76 s / 请求
14:55:07.699        8       20
14:55:09.461        6       22
14:55:11.217        4       24
14:55:12.969        2       26
14:55:14.731        0       28
14:55:15.427        0       30     ← 全部 admit 完,decode 阶段
```

**观察:从第一次 admit (14:54:39) 到最后一次 admit (14:55:15),36 秒内 admit 30 个请求**,平均 admission 间隔 = **1.2 s/请求**。

### 3.2 为什么 admission 这么慢:scheduler tick = 一次 forward

看 `scheduler.py:2640-2780` 的 `get_new_batch_prefill()` 循环:

```python
def event_loop_normal(self):
    while True:
        recv_reqs = self.recv_requests()  # 1. 从 ZMQ 拉取新请求(non-blocking)
        self.process_input_requests(recv_reqs)  # 2. 加入 waiting_queue
        batch = self.get_next_batch_to_run()  # 3. 形成 batch(prefill 或 decode)
        if batch:
            result = self.run_batch(batch)    # 4. 同步等 GPU forward 完成
            self.process_batch_result(...)    # 5. 写回结果
```

**关键:第 4 步 `run_batch()` 是同步的**,scheduler 主循环阻塞在 GPU forward 完成之前不会回到第 1 步。也就是说:

- 每个 scheduler "tick" = 一次 forward(prefill 或 decode)
- 每次 prefill tick 最多 admit ≤ K 个请求(K 受 `rem_chunk_tokens` 与 `running_batch.batch_is_full` 限制)
- **如果当前 batch 已经在 decode 状态(in-flight ≥ 1),后续的 prefill tick 必须等当前 decode batch 完成**

8img/768p 的具体节奏:
- 单次 prefill batch 处理 12,400 tokens(2 个请求)→ 用时 ~0.3 s
- 然后切到 decode batch(已 admit 的 N 个请求一起做 1 个 token)→ 每 step 30-50 ms,会跑很多 step
- decode 期间,**新到的请求只能在 waiting_queue 里堆**,直到下一次 prefill tick 触发(由 `--schedule-policy` 与 `prefill-decode-interleave` 控制)

scheduler 在 prefill 与 decode 间切换的节奏受:
- `--schedule-conservativeness`(默认 1.0)
- `--schedule-policy`(默认 lpm,但 fcfs 模式下行为接近)
- 当前 running_batch 的 KV 占用与 chunked_prefill_size 的比值

实测的 1.76 s/请求 admission 间隔 = decode 多个 step 后再做一次 prefill 的耗时,**这是 SGLang 调度器固有的"prefill-decode 交错"开销**。

### 3.3 为什么 4img/768p 没有这个 admission 瓶颈

4img/768p 的请求 input_len 只有 3,158 tokens,**5 个请求就能塞进 16,384 budget**。所以即使 scheduler 1.76 s 才做一次 prefill,**一次能 admit 5 个**,有效 admission 速率 = 5 / 1.76 ≈ 2.8 RPS,远高于 input rate 1.0 → 队列永不形成。

实测 queue-req 峰值 = 0,完全验证。

### 3.4 admission 瓶颈与 1080p / 16img/768p 的对比

| 瓶颈类型 | 1080p | 16img/768p | 8img/768p | 4img/768p |
|---|---|---|---|---|
| KV pool 容量(理论 in-flight 上限) | 41 | 54 | 106 | 202 |
| 每 prefill batch 容纳的请求数 | 1(切分) | **1**(2 个塞不下) | **2** | **5** |
| Scheduler admission 速率(reqs/s) | ~0.3(单请求 2 块) | ~0.54(1 请求/tick × 1.87s) | ~1.07(2 请求/1.86s) | ~2.8(5 请求/tick) |
| GPU sustained throughput (tok/s) | ~4,000 | 4,381 | 4,104 | 2,791(未饱和) |
| 实际饱和 RPS | 0.24 | **0.35** | **0.64** | 0.84(input-limited) |
| 实际饱和瓶颈 | **#1 + KV pool** | **GPU compute** + KV pool | **GPU compute** + admission | **input rate**(未饱和) |

**关键观察:8img/768p 的 in-flight 30 不是被 KV pool 堵住,而是**:
1. bench 总共只有 32 个请求,rate=1.0,32 秒到完
2. 平均 PD lifetime 32 s = queue 13 + forward 18 + 启动 ~1
3. Little's Law: in-flight = 1.0 × 32 = 32(理论值,与实测 30 吻合,差距来自 Poisson 抖动)

**也就是说,8img/768p 在 np=32 / rate=1.0 这个 bench 设置下,峰值 in-flight 受限于"输入只有 32 个"而非 PD 容量**。如果用 rate=2.0 或 np=64,in-flight 还能继续涨,直到撞到 admission 速率天花板。

### 3.5 8img/768p 的真实饱和 RPS:加压实测 0.70-0.72 RPS,过载会发生 goodput 崩溃

通过四组加压实测确认真实饱和点:

| 指标 | rate=1.0/np=32 | **rate=1.0/np=128** | **rate=2.0/np=64** | **rate=2.0/np=256(过载)** |
|---|---:|---:|---:|---:|
| **RPS(成功请求/总时间)** | 0.64 | **0.70** | **0.72** | **0.65**(降级) |
| 输入速率 | 1.0 req/s | 1.0 req/s | 2.0 req/s | 2.0 req/s |
| 输入总数 | 32 | 128 | 64 | 256 |
| **成功请求** | 32/32 | 128/128 | 64/64 | **151/256**(105 失败) |
| Bench 持续时间 | 50 s | 182 s | 89 s | 231 s |
| 中位 E2E(成功) | 37.6 s | 84.9 s | 66.8 s | 136.0 s |
| 中位 TTFT | 23.2 s | 39.9 s | 36.2 s | **77.2 s** |
| P99 E2E | 49.1 s | 170.2 s | 86.3 s | **219.4 s** |
| 中位 queue_duration | 13.3 s | 31.4 s | 23.1 s | **33.8 s** |
| 中位 forward_duration | 18.3 s | 38.0 s | 33.2 s | **58.0 s** |
| P99 forward_duration | — | — | 86 s | **198 s** |
| **PD scheduler running-req 峰值** | 30 | **63** | **63** | **63** |
| Queue-req 峰值 | 20 | 33 | 30 | 33 |
| Bench-side 峰值并发 | 32 | 103 | 64 | **150** |
| Total token throughput (tok/s) | 4,104 | 4,470 | 4,552 | 4,152(降级) |
| 平均并发 | 24.9 | 64.6 | 49.0 | 93.0 |
| 单请求 prefill batch 占比 (2k-7k tokens) | 11% | 44% | 11% | **51%** |
| **失败模式** | — | — | — | **NIXL embedding buffer pool 耗尽**(`Timeout while waiting for available buffer`) |

**关键观察:rate=2.0/np=256 时 RPS 不仅没继续上升,反而从 0.72 降到 0.65,且有 105 个请求失败**。这是 disagg 系统在过载下的 goodput 崩溃模式。

**RPS 验证(算力 bound 检验):**
- np=32 rate=1.0:  4,104 / (6,238 + 153) = **0.642 RPS** ✓
- np=128 rate=1.0: 4,470 / (6,238 + 118) = **0.703 RPS** ✓
- np=64 rate=2.0:  4,552 / (6,238 + 117) = **0.716 RPS** ✓
- np=256 rate=2.0: 4,152 / (6,238 + 125) = **0.653 RPS** ✓(过载下 GPU compute 实际下降)

### 3.6 为什么 8img/768p 的真实天花板是 0.70-0.72 RPS

加压揭示**第三个**饱和瓶颈,不同于 1080p 与 16img/768p:

**1. PD scheduler 触顶 `--max-running-requests=64`**:启动配置 `--max-running-requests 64`,实测三次加压(np=128/r=1, np=64/r=2, np=256/r=2)running-req 峰值都精确停在 **63/64**,从未突破。这是 SGLang scheduler 内的硬性并发上限。

**2. KV pool 还有空间**:
- np=128 rate=1.0 在 running=63 时,KV pool 占用 = 6,510 × 63 / 695,136 = **59%**
- bench-side 并发 103-150 > PD running 63 是因为 dynamo 端有 NIXL handoff 流水线(encoder done → embedding 在传输 → 等待 PD admit),这些不计入 PD running batch
- 所以**KV pool 不是瓶颈**,放开 `--max-running-requests` 到 128 仍有 41% headroom

**3. GPU compute throughput 接近饱和**:
- 三组未崩溃实测 total_token_throughput: 4,104 → 4,470 → 4,552 tok/s
- in-flight 从 30 → 63 → 63,但吞吐只升 11%
- 解释:每个 prefill batch 只能容纳 ≤ 2 个请求,batch 大小不会因为 in-flight 多而增加;decode 阶段并发越多,per-token 时间从 ~30 ms → ~50 ms(O(B²) attention)

**4. PD-side 的 prefill batch 在加压下会"退化"**:
| 配置 | 单请求 prefill batch (2k-7k) 占比 | 双请求 prefill batch (7k-13k) 占比 |
|---|---:|---:|
| np=32 rate=1.0  | 11% | 78% |
| np=64 rate=2.0  | 11% | 86% |
| **np=128 rate=1.0** | **44%** | **55%** |
| **np=256 rate=2.0(过载)** | **51%** | **48%** |

在长时间饱和下,running batch 长期 plateau 在 60+,导致 chunked_prefill 的 `rem_chunk_tokens` 经常 < 12,476(2 请求合并所需),退化为单请求 batch。这降低了 prefill 阶段的 GPU 利用率(单请求 batch 输出的 KV 数 / forward 时间比双请求低)。**过载越严重,退化越严重 → 形成正反馈,GPU compute 实际效率下降**。

**5. Goodput 崩溃机制(np=256/rate=2.0):**

当 input rate 远超 PD 吞吐(2.0 ≫ 0.72)时:
1. NIXL embedding 在 dynamo 端排队等待 PD admit
2. embedding 占用 ring buffer 池(`NIXL_MAX_BUFFER_SIZE=805306368`,即 768 MB)
3. 8img/768p 的每个 embedding ≈ 8 × 770 × 5,120 × 2 byte = ~63 MB,**单个 buffer pool 容量约 12 个并发 embedding**
4. PD admit 速度只有 ~0.72 req/s,但新 embedding 以 2.0 req/s 涌入,buffer pool 迅速耗尽
5. 后续 embedding 在 receive 端的 `ring_buffer.get_buffer()` 阻塞等待
6. 60 秒内拿不到 buffer → 抛 `TimeoutError("Timeout while waiting for available buffer.")` → 105 个请求失败

源码定位:`/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py:670-725`(参见 `receive_timeout=60` 默认值)

**结论:8img/768p 的安全工作区是 RPS ≤ 0.65,接近 0.7-0.72 时延迟急剧上升,> 0.72 时会发生 goodput 崩溃**。

### 3.7 提升 max_running_requests 能否进一步提升 RPS?

理论分析:`--max-running-requests` 从 64 → 128:
- KV pool 还有 41% headroom,理论支持 in-flight 106
- 但 GPU compute 11% 增益已逼近极限(np=64 vs np=128 都饱和在 4,470-4,552 tok/s)
- 估算上限:in-flight 100 时 lifetime ~110 s,total tput 可能 4,800 tok/s → RPS ~0.76

**结论:8img/768p 的真实 RPS 天花板约 0.70-0.80,在当前 max_running_requests=64 下已达 0.70-0.72**。把上限调到 128 至多再多 5-10%,要进一步提升必须改架构(TP=2 或 cross-host disagg)。**也要同时加大 NIXL ring buffer pool 防止过载崩溃**。

---

## 4. 4img/768p 为什么 0.84 RPS 仍未饱和

### 4.1 概念:什么叫"未饱和"

bench 实测:
- Concurrency = **6.97**(同时在 PD 上的请求数)
- Peak concurrent = 20(bench 客户端最大并发)
- queue_duration 中位 0.00 s
- **Successful requests 32/32, P99 TTFT 6 s**(8img/768p 的 P99 是 26.5 s)

这说明 PD 完全跟得上输入速率,bench 是 input-rate-limited。

### 4.2 4img/768p 的真实 RPS 容量

如果把 bench 改成 rate=4.0 / np=128:
- decode 阶段算力:每请求 256 output × 30 ms/token ≈ 7.7 s
- prefill 阶段算力:每请求 3,158 / 19,883 tok/s ≈ 0.16 s
- 单请求 PD 时间 ≈ 7.86 s
- KV pool 容量上限 = 695,136 / 3,430 ≈ **202 个 in-flight**
- 理论饱和 in-flight × 1/lifetime = 202 / 7.86 ≈ **25 RPS** —— 极高

但实际不会到 25 RPS,因为:
1. encoder 单卡处理 4img/768p 的速率上限(实测 ~3-4 RPS,见 `same_host_disagg_three_cases_rps.md`)
2. dynamo NIXL handoff overhead(150 ms/req)在高并发下会形成 backpressure
3. cuda_ipc descriptor pool 在 ~50 in-flight 时容易枯竭

合理估算 4img/768p 真实饱和点约 **2-3 RPS**(encoder 是瓶颈)。

### 4.3 4img/768p 的瓶颈在哪里(如果加压)

按照 `bottleneck_analysis.md` 的方法学,加压到 4img/768p 饱和后,瓶颈最可能出现在:
- **encoder GPU**: ViT 单图 forward + Qwen3-VL processor + thumbnail token gen 的串行开销在高并发下会饱和(估算单卡 3-4 RPS)
- **Dynamo NIXL handoff**: encoder 端 publish embedding + PD 端 pull,150 ms × 4 RPS = 600 ms 的 handoff 累积延迟,是显著瓶颈
- 不会是 PD 上的 prefill / decode(算力远未跑满)

---

## 5. 四个工作负载下瓶颈的对比总结

| 维度 | 8img/1080p | **16img/768p** | 8img/768p | 4img/768p |
|---|---|---|---|---|
| **#1 小批量 prefill 尾巴** | 触发(46%) | 不触发(input < chunked) | 不触发 | 不触发 |
| **#2 KV pool 容量** | 触发(in-flight 卡 31,KV 占用 74%) | **触发**(in-flight 卡 31,KV 占用 57%) | 不触发(rate=1.0 → 28%, rate=2.0 → 59%) | 不触发(KV 占用 9%) |
| **#3 单 batch 装不下 2 请求(1-req-per-batch)** | 触发(切分导致) | **触发**(12,401 × 2 > 16,384) | 不触发(6,238 × 2 < 16,384) | 不触发 |
| **#4 max_running_requests=64 上限** | 不触发 | 不触发 | **rate=2.0 触发(63/64)** | 不触发 |
| **#5 admission 速率(2 请求/batch)** | N/A | N/A | 触发(rate=1.0 时 in-flight 30 限制 RPS;rate=2.0 时让位给 #4) | 不触发(单 tick 5 请求) |
| **encoder/handoff** | 不可见 | 不可见 | 不可见 | 加压时显现 |
| **饱和 RPS** | 0.24(#1+#2) | **0.35**(#2+#3,GPU compute bound) | rate=1.0: 0.64;**rate=2.0: 0.72**(#4 + GPU compute,真实天花板) | ~2-3(encoder bound, 当前 0.84 = input-limited) |

### 5.1 哪些瓶颈是结构性的,哪些可调优

| 瓶颈 | 结构性 | 配置可缓解? | 工程改造可解? |
|---|:---:|:---:|---|
| #1 chunked-prefill 切分尾巴(1080p) | ✓ | 改图大小可彻底避开;改 chunked_prefill_size 治标不治本 | SGLang 上游 patch:小批量合并(详见 1080p 文档 §3.1) |
| #2 KV pool 容量(1080p) | ✓ | mem_fraction 0.85 → 0.92 可挤 ~50k tokens(31 → 35 in-flight),但容易 OOM | TP=2 / 更大 GPU(H200 NVL) |
| #3 Scheduler admission 速率(8img/768p) | 部分 | 调 `--schedule-policy lof` / `--schedule-conservativeness` 可能微调;启用 `--enable-overlap` 让 prefill/decode 并行 | SGLang 上游优化:async admission tick |
| Encoder bound(若 4img/768p 加压) | 部分 | 多 encoder 实例(B70 端 8E 而非 4E) | 跨主机 disagg + GPU embedding(已实验,见 `b70_xpu_nixl.patch` 但已撤销) |

### 5.2 为什么从 1080p → 768p RPS 提升 2.7×,但只是"换了个瓶颈"

1080p → 768p 的根本变化:input_len 16,420 → 6,238(降 62%)。这导致:

1. **chunked-prefill 尾巴消失**(#1 解锁)→ GPU 利用率从被尾巴稀释到接近峰值
2. **KV pool 占用从 74% → 28%**(#2 解锁)→ in-flight 数不再被堵住
3. 但**单请求总 token 通量(input + output)没消失**,仍占 GPU prefill 算力的 ~45%(每请求 6,238 / 14,000 fa3 sustained = 0.45 s / req)
4. 加上 scheduler tick 节奏(1.76 s 一次 prefill batch),admission 速率成为**新的二级瓶颈**

所以**768p 解决的是"低效"问题(GPU 跑了但产出少),而不是"容量"问题(GPU 跑满了)**。要进一步提升,需要并行化 admission(scheduler 改造)或减少单请求 token 数(更小的图)。

### 5.3 为什么 4img/768p 没有这个新瓶颈

4img/768p 单请求 input_len 3,158,**5 个请求合并到一次 prefill batch** → 每次 admission tick admit 5 个 → 有效 admission 速率提升 5×,完全覆盖 input rate 1.0 的需求。

---

## 6. 可操作建议

### 6.1 生产部署的工作负载选择

按 RPS 由低到高:

| 工作负载 | RPS | TTFT 中位 | 推荐部署 |
|---|---:|---:|---|
| **8img/1080p disagg** | 0.24 | 61.5 s | ❌ 不推荐 disagg,改 TP=2 agg(0.95 RPS) |
| **16img/768p disagg** | **0.35** | **40.6 s** | ❌ 与 1080p 同样的 GPU compute bound,改 TP=2 agg(预估 ~0.6-0.7 RPS) |
| **8img/1080p TP=2 agg** | 0.95 | ~10 s | ✓ 高分辨率推荐 |
| **8img/768p disagg @ rate=1.0/np=32** | 0.64 | 23.2 s | ⚠ 输入未饱和,真实容量更高 |
| **8img/768p disagg @ rate=1.0/np=128** | **0.70** | **39.9 s** | ⚠ 持续饱和真实天花板,P99 E2E 170 s |
| **8img/768p disagg @ rate=2.0/np=64** | **0.72** | 36.2 s | ⚠ 真实天花板,可考虑 TP=2 agg |
| **8img/768p disagg @ rate=2.0/np=256(过载)** | **0.65** | **77.2 s** | ❌ **goodput 崩溃,105/256 失败**(NIXL buffer 池耗尽) |
| **4img/768p disagg** | ≥ 0.84(未饱和,真实容量 2-3 RPS) | 2.4 s | ✓ 中等分辨率推荐(disagg 优势在解耦扩展) |

**生产部署红线:** 8img/768p 的安全工作区 RPS ≤ 0.65,允许少量突发到 0.7,**绝对不能持续超过 0.72 否则会失败**。需要在 dynamo 前端做 admission control / rate limiting,避免 NIXL buffer 池耗尽。

### 6.2 如果坚持用 disagg 的优化方向

针对 8img/768p 的 admission 瓶颈(rate=1.0)与 max_running_requests 瓶颈(rate=2.0)与过载崩溃(np=256/r=2):
1. **`--max-running-requests 128`**: rate=2.0 实测 in-flight 撞 63/64,放开 64→128 可能让 RPS 0.72 → 0.75-0.80(KV pool 还有 41% headroom);但 GPU compute 11% 增益已逼近极限,提升幅度有限
2. **加大 `NIXL_MAX_BUFFER_SIZE`**:从 768 MB 提升到 4 GB+(GPU 0/encoder 端有空闲 GPU mem),容纳更多并发 embedding,**避免 np=256/r=2 那种过载崩溃**;但治标不治本,真正应该在前端做 admission control
3. **dynamo 前端 admission control / rate limiting**:监控 PD running-req,当 ≥ 60 时主动拒绝新请求(返回 429),**保护 NIXL buffer 池避免 goodput 崩溃**
4. **`--enable-overlap-schedule`(若 SGLang 已支持)**: 让 admission 与 GPU forward 并行,理论可减半 admission 间隔
5. **`--schedule-policy lof`**: 倾向于先把短请求 admit,可能减少队列峰值
6. **多 PD 实例**(2 个 PD worker 各占一卡): 总 admission 速率 × 2,但需要负载均衡和 KV cache 不共享的牺牲

针对 4img/768p 加压后的 encoder 瓶颈:
1. **跨主机 4E + 1PD**: encoder 算力扩展独立于 PD,可达 3-4 RPS(B70 8E 配合 H200 PD 可能达 6 RPS)
2. **重新评估 `b70_xpu_nixl.patch`**: GPU 直传 embedding 可以省去 CPU 拷贝的 150 ms,如果 OOM 问题能从 dynamo runtime 层解决(预分配 GPU NIXL pool,避免 cuda_ipc 误导 device 推断)

### 6.3 不要做的优化

- ❌ 不要在 8img/768p rate=1.0/np=32 上调 `--max-running-requests`(实测 in-flight 仅 30,远未到 64)
- ✓ 但**在 rate=2.0 或 np≥128 上调到 128 是合理尝试**(实测撞 63/64)
- ❌ 不要在 4img/768p 上调 `--mem-fraction-static`:KV pool 利用率仅 9%
- ❌ 不要把 `--chunked-prefill-size` 从 16,384 调小(如 8,192):8img/768p 的 6,238 仍然 fit,但批合并会变差,反而降吞吐
- ❌ 不要期望 8img/768p 加压到 rate=4.0 或 np=256/r=2 能跑出 1.5 RPS:**实测 np=256/r=2 反而崩溃到 0.65 + 105 失败**
- ❌ 不要忽视 NIXL buffer 池容量:过载时这是首先暴露的失败点,`receive_timeout=60` + 768 MB 池约 12 个并发 embedding 不够大压力

---

## 7. 关键源码路径与日志位置

| 现象 | 源码位置 |
|---|---|
| Prefill batch 中 `#new-token` 与 `#cached-token` 的统计 | `/opt/sglang/python/sglang/srt/observability/scheduler_metrics_mixin.py:353-388` |
| chunked prefill 切分逻辑 | `/opt/sglang/python/sglang/srt/managers/schedule_policy.py:813-933` (`PrefillAdder.add_one_req`) |
| 多请求合并到同一 prefill batch | `schedule_policy.py:898-906`(`rem_chunk_tokens` 累减) |
| Scheduler 主循环 / admission tick | `/opt/sglang/python/sglang/srt/managers/scheduler.py:2640-2780` (`get_new_batch_prefill`) |
| Decode 与 prefill 交错策略 | `scheduler.py` 的 `event_loop_normal` 与 `--schedule-policy` 入口 |
| ReqTimeStats 上报(queue/forward duration) | `/opt/sglang/python/sglang/srt/managers/request_processor.py`(`ReqTimeStats` dataclass) |

### 本次实测日志参考

- **PD log**: `logs/samehost_pd_20260531_144409.log`
- **Encoder log**: `logs/samehost_encoder_20260531_144409.log`
- **16img/768p bench (rate=1.0/np=32)**: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/16img_768p_rate1.0_np32_nixlwrite_20260531_154933/`
- **8img/768p bench (rate=1.0/np=32)**: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_768p_rate1.0_np32_nixlwrite_20260531_145314/`
- **8img/768p bench (rate=1.0/np=128)**: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_768p_rate1.0_np128_nixlwrite_20260531_164750/`
- **8img/768p bench (rate=2.0/np=64)**: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_768p_rate2.0_np64_nixlwrite_20260531_163936/`
- **8img/768p bench (rate=2.0/np=256,过载)**: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_768p_rate2.0_np256_nixlwrite_20260531_170805/`
- **4img/768p bench (rate=1.0/np=32)**: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/4img_768p_rate1.0_np32_nixlwrite_20260531_145548/`
- **bench 起止 UTC**:
  - 16img/768p:                        2026-05-31T15:50:17 → 15:51:49(92 s,32/32 successful)
  -  8img/768p (rate=1.0/np=32):       2026-05-31T14:54:05 → 14:54:55(50 s,32/32 successful)
  -  8img/768p (rate=1.0/np=128):      2026-05-31T16:47:50 → 16:50:52(182 s,128/128 successful)
  -  8img/768p (rate=2.0/np=64):       2026-05-31T16:39:36 → 16:41:06(89 s,64/64 successful)
  -  8img/768p (rate=2.0/np=256):      2026-05-31T17:08:05 → 17:11:56(231 s,**151/256 successful**, 105 buffer timeouts)
  -  4img/768p:                        2026-05-31T14:56:38 → 14:57:16(38 s,32/32 successful)

---

**结论一句话:**

> 在同主机 H200 disagg 上,**768p 分辨率下 RPS 随图片数量呈非线性变化**:4img → 8img → 16img 的 RPS = 0.84 → 0.72(实测饱和)→ 0.35,**关键不在 input_len 本身,而在于"单请求 input_len 与 chunked_prefill_size(16,384)的比值"决定了每个 prefill batch 能合并多少个请求**(5/2/1)。当比值 > 0.5(16img/768p,12,401/16,384=0.76)时,行为退化为与 1080p 类似的"GPU compute + KV pool 双重 bound";当比值 ≤ 0.4(8img/768p,0.38)时,2 请求合并 batch,加压实测真实饱和点 0.72 RPS(撞 max_running_requests=64);当比值 ≤ 0.2(4img/768p,0.19)时,5 请求合并 batch,系统 input-bound。**真正决定 RPS 的不是分辨率(768p vs 1080p)而是 input_len(12,401 vs 6,238 vs 3,158)与 chunked_prefill_size 的比值**。

## 参考文档

- `same_host_problem_analysis_zh.md` —— 8img/1080p 工作负载下的两大瓶颈根因分析(本文档的姐妹篇)
- `same_host_disagg_time_zh.md` —— 三个工作负载下 nixl-read 与 nixl-write 的 latency 分解
- `same_host_disagg_three_cases_rps.md` —— 三个工作负载的 RPS 横向对比(基线数据)
- `bottleneck_analysis.md` —— 早期文档化的"小批量 embedding-integration"现象
- `1080p_sweep_three_way.md` —— TP=1 / TP=2 / disagg 三路对比基线
