# 同主机 Disagg E/PD 两大瓶颈的根因分析

**面向问题:**

> - **小批量 embedding-integration prefill 占据 46% 的 prefill 事件**,把 GPU 利用率拉到平均 3k tok/s(峰值 15k 的 1/5)。
> - **PD 调度器队列在饱和下中位等待 48.8 s**,因为 in-flight 数(27.9)接近 KV cache 容量上限。

下面分别从 SGLang 源码与 Dynamo 流程层面解释这两个瓶颈的形成机制、它们之间的相互强化关系,以及为什么任何配置调优都无法消除它们。

---

## 瓶颈 1:为什么 46% 的 prefill 事件是 16-320 token 小批量

### 1.1 小批量是 chunked-prefill scheduler 的副产品,不是 embedding-integration 本身的步骤

`bottleneck_analysis.md` 文档里把这些事件称为"embedding-integration small batches",这是描述**现象**而不是**机制**。实际机制如下。

**关键源码位置:** `/opt/sglang/python/sglang/srt/managers/schedule_policy.py` `PrefillAdder.add_one_req()` (第 813-933 行) 与 `/opt/sglang/python/sglang/srt/managers/scheduler.py` 的 `get_new_batch_prefill()` (第 2640-2780 行)。

每次 PD 调度器形成新 prefill batch 时会做下面这件事:

1. **如果有正在执行的 chunked_req(上一轮没跑完的请求),先优先续上它** —— 见 `scheduler.py:2674-2679`:

   ```python
   if self.chunked_req is not None:
       self.chunked_req.init_next_round_input()
       self.chunked_req = adder.add_chunked_req(self.chunked_req)
   ```

2. **然后从 waiting_queue 取新请求,逐个尝试 `add_one_req()`** —— 见 `scheduler.py:2693`:

   ```python
   for req in self.waiting_queue:
       res = adder.add_one_req(req, ...)
   ```

3. **`add_one_req()` 内部按 `chunked_prefill_size` 截断:**

   ```python
   # schedule_policy.py:907-935
   elif self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
       # Non-chunked prefill: 整请求一次性进 batch
       self.can_run_list.append(req)
   else:
       # Chunked prefill: 截断到 rem_chunk_tokens
       trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
       req.set_extend_input_len(trunc_len)
       req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]
   ```

我们的配置 `--chunked-prefill-size 16384`,所以 `self.rem_chunk_tokens = 16384` 在 batch 开始时。

### 1.2 8 imgs × 1080p 的请求要 2 块 16384,这就产生了"16336 cached + 16-320 new"的尾批

每个 8img/1080p 请求实际 input_len ≈ 16,420 tokens(8 × 2,064 vision + ~36 text)。它在 PD 上的处理路径是这样的:

**第 1 轮 prefill(把第一块送进 GPU):**
- `extend_input_len` 截断到 16,384(chunked_prefill_size 上限)
- 此时 `log_input_tokens = 16384`,`log_hit_tokens = 0`
- 日志显示 `#new-token: 16384, #cached-token: 0`
- 该请求被标记为 `chunked_req`,**保留在调度器内部不释放**
- forward 在 GPU 上跑完,处理 16,384 个 token,吞吐能跑到 15k tok/s 的峰值

**第 2 轮 prefill(尾巴):**
- `init_next_round_input()` 从 `fill_ids[16384:]` 继续,也就是剩下的 36 个 text token
- 此时如果 KV cache 中 prefix_indices = 16,384,新增 extend_input_len = 36
- **但此时 `log_hit_tokens` 不再是 0,而是 16,384(prefix length 计入了 cached 列)**
- 日志变成:`#new-token: 36, #cached-token: 16384`
- 这就是日志里看到的 `#new-token: 16` / `32` / `48` / `64` / `80` / `96` / `112` / `320` 这些事件 —— 其实是**主请求第 2 块 chunked prefill 的"尾巴"**

让我们对照实际日志数据:

```
#new-token 分布(本次 bench 实测,68 个 prefill 事件):
  16,384 (满块):           33 次   ← 主请求第 1 块
  16,368 (满块少 16):      4 次   ← 主请求第 1 块对齐到 page_size 后微调
  320, 96, 80, 64, 48, 32, 16:  31 次   ← 主请求的"尾巴"
```

**33 个满块 + 31 个尾块 ≈ 32 个请求 × 2 块/请求**(略有偏差是因为有的请求第一块没满 16,384,被合并到了 16,368)。

### 1.3 为什么尾块这么小:vision tokens 的对齐边界

8 imgs × 1080p:每张 1080p 图像在 Qwen3-VL 编码后产出 ~2,064 visual tokens(包含 grid + position embedding 占用)。8 张图共 ~16,512 visual tokens,加上 ~36 text tokens,总长 ~16,548。但实际 input_len 中位数是 16,420(因为 chat template + tokenizer 的少许差异),page_size=16 对齐后变成 16,416。

所以第一块 16,384 = 16,416 - 32 → 第二块只有 32 token,完全符合观察到的 "32 token 尾批"。其他尾批大小 (16, 48, 64, 80, 96, 112, 320) 则来自:
- `chunked_prefill_size` 在批次中可能因为 `running_batch` 占用了部分 KV slot 而临时缩小到 < 16,384(`scheduler.py:2647-2652` 的 `dynamic_size`)
- 另一个请求的尾块和这个请求的尾块被合并到一次 forward(此时 `log_input_tokens = 32 + 64 = 96` 等)
- **enable_dynamic_chunking 被禁用(默认),所以 `chunked_prefill_size` 是固定的 16,384**

### 1.4 为什么这些尾块吞吐只有 ~10-200 tok/s(峰值 15k 的 1/1500)

`PrefillStats` 记录 `log_input_tokens` 和 `gap_latency`(两次 prefill 报告之间的实际墙钟):

```python
# scheduler_metrics_mixin.py:368-372
gap_latency = now - self.last_prefill_stats_tic
self.last_prefill_stats_tic = now
self.last_input_throughput = (
    prefill_stats.log_input_tokens / gap_latency if gap_latency > 0 else 0.0
)
```

对一个 32-token 的尾批:
- `log_input_tokens = 32`
- `gap_latency`(从上一次 prefill 报告到这一次)≈ 当前 forward + 调度器 tick 间隙 ≈ 200-400 ms
- 算出来吞吐 = 32 / 0.3 ≈ **107 tok/s**

这不是因为 32 token 本身慢,而是因为**整个 32 层 transformer 的 forward 必须跑一遍才能输出这 32 个 token 的 KV**,而 launch + sync overhead 占据了绝大部分 wall clock。具体地:

- Qwen3-VL-32B FP8 单层 forward 用 fa3 attention + FP8 MM,16 token 批次大小:
  - 每层 `~3 ms` (kernel launch + driver sync 占主导)
  - 32 层 × 3 ms = **96 ms**(理论下限)
  - 实测 200-400 ms 包含 chunked prefill prologue + scheduler tick + token output handler
- 同样 32 层在 16,384 token 批次下:
  - 每层 ~10 ms(算力 bound)
  - 32 层 × 10 ms = **320 ms**
  - 但产出 16,384 个 token 的 KV → 16384 / 0.32 ≈ **51k tok/s**(实测峰值 15k 受 PCIe / Mamba state cache write 限制)

所以**小批量的"低吞吐"是 GPU kernel launch overhead 的固有代价**,不是算法低效。在普通 LLM 中如果用大批量串成一行就没这问题,但 chunked prefill 的设计强制把请求切片,每片必须有自己的 forward。

### 1.5 为什么 disagg 比 agg 严重:NIXL embedding 触发了第二个 chunk

**关键差异:**

在 **agg 模式** 下,完整 multimodal 请求一次性进入 SGLang scheduler。如果 input_len = 16,420,scheduler 一次 add_one_req 就把整个请求加入 chunked_prefill,第一块跑 16,384,第二块跑 36 —— 但**两块在 SGLang 内部连续执行**,中间没有外部事件干扰,中位 forward_duration 可以做到接近第一块的时间(~1 s)。

在 **disagg 模式** 下,流程是:
1. PD 收到 dynamo `SglangMultimodalRequest`,从 NIXL receive embedding
2. PD 调用 `engine.async_generate(input_ids=..., image_data=mm_items)`
3. SGLang 把请求加入 waiting_queue
4. 调度器 `get_new_batch_prefill` 触发,把请求加入 chunked prefill,跑第一块 16,384
5. 此时 chunked_req 还没跑完,scheduler 进入下一个 step 继续跑第二块 36

**问题是,在 disagg 的 PD 上,小批量 prefill 事件会与"新到达的 NIXL embedding 触发的新请求 prefill"交错执行**。具体看 `scheduler.py:2674-2691`:

```python
if self.chunked_req is not None:
    # 优先续上一个 chunked_req
    self.chunked_req = adder.add_chunked_req(self.chunked_req)

# 然后再尝试 add new requests
for req in self.waiting_queue:
    if running_batch.batch_is_full: break
    adder.add_one_req(req, ...)
```

**关键是 `add_chunked_req` 会消耗 `rem_chunk_tokens`**,所以如果它先跑了 32 token 尾巴,`rem_chunk_tokens` 还剩 16,352 —— 这时候 add new req 就能塞一个新请求的 16,352 token 块。结果是日志里看到的 **`#new-token: 16384` 和 `#new-token: 32` 交错出现**。

agg 模式下虽然也有这个机制,但 agg 的 in-flight requests 更少(因为 SGLang scheduler 直接触发,延迟低),chunked_req 通常在被新请求干扰之前就已经跑完了。

### 1.6 小结:小批量的根本原因

| 层级 | 因素 | 是否可调 |
|---|---|---|
| **设计层** | chunked-prefill 把 >16,384 token 请求切成 N 块,每块都要走一次 32 层 forward | 改 `--chunked-prefill-size` 到 32,768 可减少切分,但会牺牲并发(每块占满 budget) |
| **算法层** | 每块小批量的 GPU kernel launch overhead 占主导 | 几乎不可调,32B FP8 模型每层 ~3 ms 是硬件下限 |
| **场景层** | 8img/1080p 的 input_len = 16,420 刚好略大于 16,384,**总是产生一个超小的尾巴** | 改图片尺寸到 8img/768p(输入只有 ~6k token,一块就跑完)或 4img/1080p(8k token)就没这问题 |
| **disagg 特有** | 多请求并发时,小批量 prefill 与大块 prefill 交错,对 decode 形成更频繁的打断 | 需要 SGLang 调度器层面的"尾块合并"或"小批量延迟到无新请求时再跑"的修改 |

---

## 瓶颈 2:为什么 PD 调度器队列在饱和下中位等待 48.8 秒

### 2.1 KV cache 容量是硬性上限,不是软性配置

在 PD 启动日志里 (`samehost_pd.log` 第 1 条 `init_model_worker`):

```
max_total_num_tokens=695136
chunked_prefill_size=16384
max_prefill_tokens=16384
max_running_requests=64
context_len=262144
available_gpu_mem=20.43 GB
```

- `max_total_num_tokens = 695,136` —— 这是 KV cache pool 的容量上限,由 `mem_fraction_static=0.85` × GPU 总内存 - 模型权重 - cuda graphs 计算得出
- `max_running_requests = 64` —— scheduler 接受的最大并发请求数(配置上限)

**实测 in-flight 请求数(running-req)中位 12.18,峰值 31**,远低于 64 的配置上限。**这说明并发瓶颈不是 max_running_requests,而是 KV cache 容量。**

### 2.2 为什么 in-flight 卡在 31 而不能继续增加

每个 8img/1080p 请求需要的 KV slot:
- input_len = 16,420
- output_len = 256 (max_new_tokens)
- 总共需要 16,420 + 256 = 16,676 token 的 KV cache slot

理论上 KV pool (695,136 tokens) 能容纳 695,136 / 16,676 ≈ **41 个请求**。

但实际峰值只到 31,原因在 `PrefillAdder.add_one_req` 第 866-870 行:

```python
total_tokens = req.extend_input_len + max_new + self.page_size
# ...
if total_tokens >= self.rem_total_tokens:
    return AddReqResult.NO_TOKEN
```

每次有新请求进来,**adder 不光要它当下能容纳,还要预留 `max_new_tokens` 的 KV slot**,以保证这个请求一直能跑到 max_new_tokens 而不会被抢占。所以预留量 = 16,420 + 256 + 16(page_size) = 16,692 tokens/req。

实际算下来:
- 31 × 16,692 = 517,452 tokens 占用
- 加上 `chunked_prefill` 进行中那个 chunked_req 的 lock(可能再占一个 slot)
- 加上 page_size 对齐冗余(每个请求实际占用向上对齐到 page_size 的倍数)

实测 31 是因为**mem_fraction=0.85 留的 20 GB headroom 里还有约 60% 被 cuda graphs / activation scratch 占用了**,真正可用的 KV pool 比 695,136 tokens 略少,导致 41 → 31 的折扣。

### 2.3 为什么队列等待中位 48.8 秒

这是 Little's Law 的直接结果:

```
稳态:in-flight 数 = 到达率 × 平均生命周期

bench 输入: rate = 1.0 RPS, np = 32
PD 平均生命周期 ≈ 98 s (= queue 48.8 + forward 50.7 中位)
PD 实际饱和吞吐: 0.23 RPS

稳态 in-flight = 0.23 × 98 = 22.5  (与实测 27.9 接近,差异因为 bench 的 32 个请求是有限批不是稳态)
```

队列长度推导:
- 一旦 in-flight 数 = 31(KV cache 上限),scheduler 标记 `batch_is_full = True`,新请求只能在 waiting_queue 里排队
- 队列里的请求按 FCFS 顺序等待
- 每完成一个请求,waiting_queue 头部的请求才能被 admit

如果稳态 in-flight = 31,平均生命周期 = 98 s,那么排在队列第 N 个的请求要等 (N / 31) × 98 s。np=32 输入,从 t=0 开始 Poisson 到达,最慢的请求大概在 t=32 时刻到,这时已有 31 个 in-flight,它要等 ~98 s。**中位请求等 ~16 个完成 ≈ 50 s**,与实测 queue_duration 中位 48.8 s 完全吻合。

### 2.4 max_running_requests 为什么没起作用?

我们配置的是 `--max-running-requests 64`,但实测峰值只有 31。这是因为 SGLang scheduler 同时受**两个**约束:

```python
# scheduler.py 内部逻辑(简化):
if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
    self.running_batch.batch_is_full = True

# 进一步,add_one_req 内部:
if total_tokens >= self.rem_total_tokens:
    return AddReqResult.NO_TOKEN  # KV cache 不够
```

`max_running_requests=64` 给的是 **数量上限**(64 是请求数);`mem_fraction_static=0.85` 给的是 **token 上限**(695,136 是 KV slot 数)。**实际 in-flight 数 = min(64, 695,136 / per_req_kv_demand)**。对 8img/1080p 来说,KV demand 是限制项 → 31 而不是 64。

**所以仅仅提高 `--max-running-requests` 没用,要么**:
- 提高 `mem_fraction_static`(从 0.85 → 0.92 可多挤出 ~50,000 tokens 的 KV pool,允许 in-flight 从 31 → 35)—— 但风险是 OOM(我们前面的 patched 实验已经验证 0.92 直接 OOM)
- 减少每请求 KV demand(用更小的图、更短的 output_len)
- 用更大的 GPU(H200 → H200 NVL 把 142 GB → 282 GB,KV pool 翻倍,in-flight 可达 60+)

---

## 两个瓶颈的相互强化:为什么单独修一个无效

### 3.1 队列长 → 小批量更频繁 → forward 慢 → 队列更长

```
请求 N+1 到达 PD waiting_queue
        ↓
queue_duration += 50 s 中位等待
        ↓
请求 N+1 终于进入 in-flight,启动第一块 chunked prefill (16,384 token)
        ↓
forward 跑大批量,GPU 利用率高
        ↓
但下一个调度 tick:scheduler 看到上次的 chunked_req 还在,先续上 32-token 尾巴
        ↓
此 forward 跑小批量,GPU 利用率低 → 占了 200-400 ms 但产出很少
        ↓
正在 decode 的其他请求被 prefill 打断 → ITL 飙升 → 它们的生命周期也被拉长
        ↓
in-flight 排不出去,KV pool 一直饱和 → waiting_queue 一直 31 深
        ↓
稳态形成:中位 queue 48.8 s,中位 forward 50.7 s
```

### 3.2 为什么修 #1(增大 chunked_prefill_size)会让 #2 更糟

如果把 `--chunked-prefill-size` 从 16,384 改到 32,768:
- 8img/1080p 的 16,420 token 请求**一次跑完**,小批量没了 → ✓ 修了 #1
- 但单次 forward 处理 16,420 tokens 占用 ~16,420 × 0.0006 ms/token = ~10 ms 算力,**外加每层的 attention O(N²) 开销 → 单次 forward 时间从 ~1 s → ~3 s**
- batch 内部 `rem_chunk_tokens` 一开始就是 32,768,如果当前 batch 已有 16,000 tokens 在跑,后来的请求只剩 16,768 配额,反而触发 chunking → 没解决问题
- 更大的 budget 也意味着 batch 拼装时间更长,新请求被 admit 的延迟更高 → ✗ 加重了 #2

### 3.3 为什么修 #2(降低 mem_fraction 增大 KV pool)也救不了

如果把 `--mem-fraction-static` 从 0.85 → 0.92:
- KV pool 695k → ~770k → in-flight 上限 31 → 35
- 但降 fraction 留下更少 GPU mem 给 activations / cuda graphs
- 实测多次 OOM(`patches_for_one_request_handoff.md` round 4 + `round5_patch_results.md` 都验证过)
- **即使不 OOM**,瓶颈 #1(小批量频率)与 in-flight 数没关系,只与 chunked_prefill_size 与 input_len 的关系有关 → ✗ 无效

---

## 为什么 agg 模式没这问题(对比验证)

`1080p_sweep_three_way.md` 实测同 workload 在 TP=2 agg 下 0.95 RPS,4× 的提升来自:

1. **TP=2 把 KV pool 翻倍** → in-flight 上限 ~60(单卡 H200 受 mem_fraction 0.85 限制 31,TP=2 NVLink 共享内存有效 KV pool ~1.4 M tokens)
2. **agg 模式下 vision encode 和 LLM forward 在同一个进程** → **没有 NIXL handoff,scheduler 队列等待 << 1 s**(因为 ViT 和 LLM 在同一个 forward call 内,decoder 一拿到 mm_items 立刻 prefill,无内部 fragmentation)
3. **小批量频率不变**(还是有 16,384+尾巴的问题),但因为 in-flight 数高,**小批量整体在 batch 中占的相对算力更小**(被并行 decode 的 31+ 个请求摊薄)

所以:**agg 模式没有解决 #1,但通过 TP=2 + 无 disagg overhead 让 #2 完全消失,从而把整体 RPS 推到 0.95**。

disagg 模式即使是同主机,也固有承担 NIXL handoff 的 ~150 ms/req 额外开销,加上 cuda_ipc 不能共享 KV cache(每个 worker 独立 mem fraction),**结构上不可能达到 TP=2 agg 的水平**。

---

## 总结:这两个瓶颈是结构性的,不能配置调优

| 瓶颈 | 根因 | 是否可在 config 层修复 |
|---|---|:---:|
| **#1 小批量 prefill 占 46%** | input_len(16,420)略大于 chunked_prefill_size(16,384),每个请求被切成"满块 + 微小尾巴";尾巴的 GPU launch overhead 占 wall clock 主导 | ❌ |
| **#2 队列等待中位 48.8 s** | KV cache pool(695k tokens × mem_frac=0.85)只能容纳 31 个并发请求(每请求 16,676 token KV demand),输入 32 个 prompt + Poisson 到达必然产生 ~16 深的稳态队列 | ❌ |
| **#1 与 #2 相互放大** | 队列长 → in-flight 多 → 小批量与大块交错频繁 → decode 打断更频繁 → forward 中位时间 ↑ → 队列更长 | ❌ |

### 想真正解决,需要 dynamo / SGLang upstream 层面的改造:

1. **小批量合并(coalesce small chunks):** SGLang 调度器在每个 step 检查 `chunked_req` 的剩余 tokens,如果 < 阈值(如 256),直接和当前 batch 的 main prefill 合并成一次 forward,把 32 token 尾巴粘到 16,384 主块后面变成一次 16,416 forward。这样 #1 的小批量频率会从 46% → ~5%。
2. **KV-aware admission control + dynamic chunk size:** 启用 `--enable-dynamic-chunking`(SGLang 已有但默认关闭),让 scheduler 根据当前 KV 占用动态决定 chunk size,避免 in-flight 卡在 31。
3. **预分配 GPU NIXL descriptor pool 与 mem_fraction 协调:** 见 `patches_for_one_request_handoff.md` 第 4 轮和 `round5_patch_results.md` —— 这是个多日上游工程任务,目前未完成。
4. **架构性方案:推荐用 TP=2 agg 替代 disagg。** 同样 2 张 GPU,RPS 从 0.23 → 0.95(4×),不仅吞吐高,TTFT 也好得多(~10 s vs 70 s)。

---

## 关键源码路径(便于后续验证或修改)

| 现象 | 源码位置 |
|---|---|
| Prefill batch 中 `#new-token` 与 `#cached-token` 的统计 | `/opt/sglang/python/sglang/srt/observability/scheduler_metrics_mixin.py:353-388` |
| `log_input_tokens` / `log_hit_tokens` 累加 | `/opt/sglang/python/sglang/srt/managers/schedule_policy.py:602-603` |
| chunked prefill 切分逻辑 | `/opt/sglang/python/sglang/srt/managers/schedule_policy.py:813-933` (`PrefillAdder.add_one_req`) |
| KV cache 容量预留检查 | `schedule_policy.py:866-870`(`if total_tokens >= rem_total_tokens: return NO_TOKEN`) |
| chunked_req 续接逻辑 | `/opt/sglang/python/sglang/srt/managers/scheduler.py:2674-2679` |
| 多模态 embedding 注入 forward | `/opt/sglang/python/sglang/srt/managers/mm_utils.py:990-1075` (`general_mm_embed_routine`) |
| Dynamo PD-side mm 处理 | `/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/worker_handler.py:402-454` (`_generate_aggregated`) |
| Qwen3-VL forward 入口 | `/opt/sglang/python/sglang/srt/models/qwen3_vl.py:1240-1300` |

## 参考文档

- `same_host_disagg_time_zh.md` —— 本根因分析对应的 bench 报告
- `bottleneck_analysis.md` —— 早期文档化的"小批量 embedding-integration"现象
- `deep_analysis_disagg_worse_h200.md` —— 第一轮深度分析,首次提出小批量假说
- `patches_for_one_request_handoff.md` —— 5 轮 handoff patch 实验(round 4 涉及 KV pool 预分配)
- `round5_patch_results.md` —— 同主机 disagg patch 在 mem_fraction=0.85/0.65/0.50 下都 OOM 的详细分析
- `1080p_sweep_three_way.md` —— TP=1 / TP=2 / disagg 三路对比基线(TP=2 = 0.95 RPS)

---

**结论一句话**: 同主机 disagg 在 32B FP8 + 8img/1080p + np=32 + rate=1.0 工作负载下的 0.23 RPS 天花板,是 SGLang 的 chunked-prefill 切分算法 + KV-cache 容量上限 + Dynamo NIXL handoff overhead 三个结构性因素的乘积,这三者中没有任何一个能通过环境变量或 CLI 参数有效地缓解。
