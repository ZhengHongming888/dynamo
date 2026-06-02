# Disagg 8img/768p np=128: max_running_requests 64 → 128 实测

**测试日期:** 2026-06-01 00:11-00:28 UTC
**环境:** sc09super21-h200, GPU 0(encoder) + GPU 1(PD), 同主机 disagg, NIXL_WRITE
**模型:** Qwen3-VL-32B-Instruct-FP8
**唯一变量:** PD `--max-running-requests` 从 64 → 128(其它配置完全相同)

## 1. 头版结果

| 指标 | max=64 r=1.0 | **max=128 r=1.0** | max=64 r=2.0 | **max=128 r=2.0** |
|---|---:|---:|---:|---:|
| **RPS** | 0.70 | **0.74** | 0.67 | **0.92 ★** |
| **成功请求** | 128/128 | **128/128** | **107/128** | **128/128** ★ |
| **失败数** | 0 | 0 | **21** | **0** |
| 失败模式 | — | — | NIXL buffer 池耗尽 | — |
| Total tput (tok/s) | 4,470 | 4,682 | 4,282 | **5,868** |
| 中位 TTFT (s) | 39.9 | 32.4 | 60.4 | 42.9 |
| 中位 E2E (s) | 84.9 | 101.8 | 109.9 | 97.4 |
| P99 E2E (s) | 170.2 | 164.4 | 152.4 | **132.7** |
| 中位 TPOT (ms) | 489 | 633 | 392 | 495 |
| 平均并发 | 64.6 | 72.5 | 49.0 | 90.5 |
| **PD running-req 峰值** | **63** | **110** | **63** | **110** |
| Queue-req 峰值 | 33 | 27 | 33 | 27 |

**关键观察:**

- ✅ **rate=2.0 RPS +37.3%(0.67 → 0.92)**,且**21 个失败请求全部成功** —— 这是最大的收益
- ✅ **rate=1.0 RPS +5.7%(0.70 → 0.74)** —— 边际改善,因为 max=64 时已经基本满负荷
- ✅ **NIXL buffer 池耗尽崩溃完全消失**(rate=2.0 失败 21 → 0)—— 因为 in-flight 翻倍后 buffer 释放速度更快
- 📊 **PD running-req 峰值从 63 → 110**(KV pool 理论上限 106,实际 110 略超是动态内存调整)

## 2. 为什么 rate=2.0 收益如此大?

之前 max=64 时,rate=2.0 的瓶颈链(详见 `same_host_3_cases_problem_analysis_zh_v01.md`):
1. PD running 卡 63(max_running_requests=64 上限)
2. Input rate 2.0 ≫ PD admit 0.67 → embedding 在 NIXL ring buffer 池累积
3. 768 MB 池只装 ~12 个并发 8img/768p embedding(每个 ~63 MB)
4. Buffer 池 60 秒内耗尽 → `Timeout while waiting for available buffer` → 21 个失败

放开到 max=128 后:
1. **PD running 上升到 110** —— 接近 KV pool 真饱和点(per-req KV 6,510 × 110 = 716k > 695k pool;实测刚好压在 mem_fraction=0.85 留的 cuda_graph 余量边界)
2. Buffer 池里的 embedding 释放速度从 0.67 → 0.92 RPS(+37%),**buffer 池占用率稳定在低位**
3. 没有任何 buffer timeout 失败

## 3. Prefill batch 组成对比(rate=2.0)

| Batch 组成 | max=64 | **max=128** | 解读 |
|---|---:|---:|---|
| Single-request batch(2-7k token) | **40%** | 33% | max=128 让 batch 合并率更高 |
| Double-request batch(7-13k token) | **59%** | 44% | max=128 时 KV 占用更高,合并 budget 紧张 |
| Tiny tail batch(<500 token) | 1% | **23%** | RadixCache 命中(input_ids 前缀共享),不是切分尾巴 |

注:max=128 时 23% 的 <500 token batch 是 **SGLang RadixCache 命中**(median cached=12,320,即两个 6,160 token 前缀已被缓存),而不是 chunked-prefill 强制切分的小尾巴。这些是**有益的 cache 命中,加速实际计算**,不是新问题。

## 4. 与 TP=1 Aggregate 对比

| 配置 | r=1.0 RPS | r=2.0 RPS | 失败 r=2.0 | 中位 E2E r=1.0 |
|---|---:|---:|:---:|---:|
| Disagg max=64 | 0.70 | 0.67 | 21 | 84.9 s |
| **Disagg max=128** | **0.74** | **0.92** | **0** | 101.8 s |
| **TP=1 Agg(参考)** | **0.90** | **1.49** | **0** | **5.3 s** |

- **max=128 把 disagg 从 0.67 拉到 0.92,但仍然只有 agg(1.49)的 62%**
- **agg 在 8img/768p r=2.0 仍然 1.62× 快于 disagg max=128**
- **TTFT 差距巨大**:agg 1.7 s vs disagg max=128 32-43 s,因为 agg 没有 NIXL handoff 排队

## 5. KV pool 理论分析

```
per-req KV demand = input_len + max_new + page = 6,238 + 256 + 16 = 6,510 tokens
KV pool = 695,136 tokens (mem_fraction=0.85)
理论 in-flight 上限 = 695,136 / 6,510 ≈ 106
```

实测 max=128 时 in-flight 峰值 **110**,基本等于理论上限。**这意味着 max=128 已经把 disagg 推到 KV pool 真饱和**:
- KV pool 占用 110 × 6,510 = 716k tokens > 695k 池(动态内存压缩)
- 进一步放开 max=256 不会再涨,因为 KV 已经满 → Out-Of-Memory 风险

**结论:在当前 mem_fraction=0.85 配置下,max_running_requests=128 是 8img/768p disagg 的接近最优值**。再放开需要降低 mem_fraction(留更少 cuda_graph headroom,可能 OOM)。

## 6. RPS 模型验证

公式 `RPS = total_tput / per_req_tokens`:

| 配置 | total_tput | per_req tokens | 预测 RPS | 实测 RPS |
|---|---:|---:|---:|---:|
| max=128 r=1.0 | 4,682 | 6,238 + 118 = 6,356 | **0.737** | 0.74 ✓ |
| max=128 r=2.0 | 5,868 | 6,238 + 118 = 6,356 | **0.923** | 0.92 ✓ |

误差 < 0.5%,与之前 14 个数据点的预测精度一致。

## 7. 同样改造对其它工作负载的预期效果

基于本次 8img/768p 实测和瓶颈分析:

| 工作负载 | 当前 max=64 RPS | max=128 预期 RPS | 解释 |
|---|---:|---:|---|
| **8img/768p** | 0.67-0.70 | **0.74-0.92(实测)** | KV pool 利用率从 59% → 100%,真饱和 |
| 16img/768p | 0.33-0.34 | ~0.35-0.45(估算) | 已被 KV pool 在 in-flight=54 真饱和;max=128 给不了多少额外 in-flight |
| 8img/1080p | 0.18-0.23 | ~0.22-0.28(估算) | 已被 KV pool + 切分尾巴双重制约;max 不会改善 |
| 4img/768p | 0.98-1.79 | 与 max=64 几乎一致 | 远未达 max=64,放开无效 |

**结论:`--max-running-requests 128` 主要受益的是 8img/768p**(从 max=64 是真正的有效约束,放开后 KV pool 才成为瓶颈)。其它 case 该改造收益小或无。

## 8. 部署建议更新

| 工作负载 | 之前推荐 | **更新推荐** |
|---|---|---|
| 8img/768p disagg | TP=1 agg(1.5 RPS) > disagg max=64(0.67 RPS) | **TP=1 agg(1.5 RPS) > disagg max=128(0.92 RPS)** —— agg 仍然胜出 1.62×,但 disagg max=128 是合理 fallback |
| 8img/768p,需要 disagg(跨主机) | max=64 在 rate=2.0 会失败 21/128 | **max=128 在 rate=2.0 100% 成功,RPS 0.92** —— 必须放开 |

## 9. 结果文件

- 8img/768p np=128 r=1.0 (max=128): `/hongming/res_samehost_disagg_32b_gpu01_max128/8img_768p_rate1.0_np128_20260601_002039/`
- 8img/768p np=128 r=2.0 (max=128): `/hongming/res_samehost_disagg_32b_gpu01_max128/8img_768p_rate2.0_np128_20260601_002431/`
- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/samehost_pd_20260601_001142.log`
- Encoder log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/samehost_encoder_20260601_001142.log`
- 启动脚本: `/tmp/start_samehost_disagg_max128.sh`(基于 `start_samehost_disagg_super21.sh` 改 max_running 64 → 128)

## 10. 一句话结论

> **在 8img/768p disagg 同主机配置下,`--max-running-requests 64 → 128` 把 rate=2.0 的 RPS 从 0.67 提升到 0.92(+37%),并消除了原本 21/128 的 NIXL buffer 池耗尽崩溃**。但仍然只有 TP=1 agg(1.49 RPS)的 62%,所以 agg 仍是同主机部署的首选;disagg 仅在跨主机或异构 GPU 场景下有意义,且必须使用 `max_running=128` 而不是默认 64。
