# 同主机 TP=1 Aggregate vs Disagg E/PD 对比 (Qwen3-VL-32B-FP8, np=128, rate∈{1.0, 2.0})

**测试日期:** 2026-05-31
**主机:** sc09super21-h200 (172.26.46.133)
**模型:** Qwen3-VL-32B-Instruct-FP8

**Aggregate 配置:**
- TP=1, GPU 1, mem-fraction-static=0.85
- max-running-requests=64(与 disagg PD 相同)
- chunked-prefill-size=16384, page-size=16, kv-cache-dtype=fp8_e4m3
- `--enable-mm-global-cache`(启用)
- ViT + LLM 同进程,无 NIXL handoff

**Disagg E/PD 配置(参考,见 `same_host_3_cases_problem_analysis_zh_v01.md`):**
- Encoder GPU 0 + PD GPU 1, NVLink, NIXL_WRITE
- PD: max-running-requests=64, mem-fraction=0.85
- Encoder: mem-fraction=0.85
- 其它参数与 agg 一致

---

## 1. 头版结论:**TP=1 Aggregate 在每个 case 都吊打同主机 Disagg**

| 工作负载 | rate | **Agg RPS** | **Disagg RPS** | **Agg / Disagg** | Agg 成功率 | Disagg 成功率 |
|---|---:|---:|---:|---:|:---:|:---:|
| 8img/1080p | 1.0 | **0.47** | 0.23 | **2.0×** | 128/128 ✓ | 65/128 ⚠ |
| 8img/1080p | 2.0 | **0.49** | 0.18 | **2.7×** | 128/128 ✓ | 51/128 ⚠ |
| 16img/768p | 1.0 | **0.71** | 0.34 | **2.1×** | 128/128 ✓ | 77/128 ⚠ |
| 16img/768p | 2.0 | **0.76** | 0.33 | **2.3×** | 128/128 ✓ | 72/128 ⚠ |
| 8img/768p  | 1.0 | **0.90** | 0.70 | 1.3× | 128/128 ✓ | 128/128 ✓ |
| 8img/768p  | 2.0 | **1.49** | 0.67 | **2.2×** | 128/128 ✓ | 107/128 ⚠ |
| 4img/768p  | 1.0 | **0.99** | 0.98 | 1.0× | 128/128 ✓ | 128/128 ✓ |
| 4img/768p  | 2.0 | **1.93** | 1.79 | 1.08× | 128/128 ✓ | 128/128 ✓ |

**关键观察:**

1. **Aggregate 100% 成功率**(全部 8 组 128/128),无任何 NIXL buffer pool 耗尽崩溃 — 因为 agg 没有 NIXL handoff
2. **Aggregate 在重负载场景(1080p、16img、8img)下 RPS 提升 2-2.7×**,在轻负载(4img/768p)下两者基本持平
3. **Aggregate rate=1.0 → 2.0 RPS 普遍上升**:6/8 个 case 提升,2 个持平。Disagg 反过来普遍下降。
4. **小图轻负载 4img/768p**:agg vs disagg 几乎相同(0.99/1.93 vs 0.98/1.79),因为该 case 在 disagg 下也不饱和

---

## 2. 完整指标对比

### 2.1 8img/1080p

| 指标 | Agg r=1.0 | Disagg r=1.0 | Agg r=2.0 | Disagg r=2.0 |
|---|---:|---:|---:|---:|
| **RPS** | **0.47** | 0.23 | **0.49** | 0.18 |
| 成功 | **128/128** | 65/128 | **128/128** | 51/128 |
| total tput (tok/s) | **7,787** | 3,781 | **8,033** | 2,975 |
| 平均并发 | 67.9 | 39.7 | 82.8 | 36.9 |
| 中位 TTFT | 82.3 s | 112.0 s | 107.7 s | 169.9 s |
| 中位 E2E | 148.7 s | 175.4 s | 187.7 s | 214.9 s |
| P99 E2E | 217.7 s | 229.1 s | 226.1 s | 256.7 s |
| 中位 TPOT | 663 ms | 726 ms | 475 ms | 355 ms |

**Agg 比 disagg 在该 case 加倍**:
- GPU compute throughput 从 3,781 → 7,787 (+106%) — agg 没有 NIXL backpressure 拖累 GPU
- in-flight 从 ~30 → 68(可以堆更高 in-flight,因为没有 NIXL buffer 上游限制)
- 所有 128 个请求都完成(disagg 失败一半)

### 2.2 16img/768p

| 指标 | Agg r=1.0 | Disagg r=1.0 | Agg r=2.0 | Disagg r=2.0 |
|---|---:|---:|---:|---:|
| **RPS** | **0.71** | 0.34 | **0.76** | 0.33 |
| 成功 | **128/128** | 77/128 | **128/128** | 72/128 |
| total tput (tok/s) | **8,934** | 4,242 | **9,553** | 4,109 |
| 平均并发 | 62.2 | 53.7 | 82.8 | 48.3 |
| 中位 TTFT | 36.2 s | 71.4 s | 78.2 s | 104.9 s |
| 中位 E2E | 80.5 s | 163.2 s | 109.1 s | 155.7 s |
| P99 E2E | 167.7 s | 223.6 s | 158.7 s | 201.3 s |

### 2.3 8img/768p

| 指标 | Agg r=1.0 | Disagg r=1.0 | Agg r=2.0 | Disagg r=2.0 |
|---|---:|---:|---:|---:|
| **RPS** | **0.90** | 0.70 | **1.49** | 0.67 |
| 成功 | **128/128** | 128/128 | **128/128** | 107/128 |
| total tput (tok/s) | **5,704** | 4,470 | **9,481** | 4,282 |
| 平均并发 | **6.0** | 64.6 | 55.7 | 49.0 |
| 中位 TTFT | **1.7 s** | 39.9 s | 9.2 s | 60.4 s |
| 中位 E2E | **5.3 s** | 84.9 s | 33.8 s | 109.9 s |
| P99 E2E | 17.7 s | 170.2 s | 78.0 s | 152.4 s |

**Agg r=1.0 在 8img/768p 上极快**(中位 E2E 5.3 秒 vs disagg 84.9 秒,**16× 加速**)。这是因为 agg 不需要排队等 NIXL embedding,scheduler 拿到请求立刻 prefill。Agg 平均并发仅 6 个(disagg 65)说明 PD 容量充裕。

### 2.4 4img/768p

| 指标 | Agg r=1.0 | Disagg r=1.0 | Agg r=2.0 | Disagg r=2.0 |
|---|---:|---:|---:|---:|
| **RPS** | **0.99** | 0.98 | **1.93** | 1.79 |
| 成功 | 128/128 | 128/128 | 128/128 | 128/128 |
| total tput (tok/s) | 3,233 | 3,198 | **6,306** | 5,856 |
| 平均并发 | 2.9 | 7.5 | **5.2** | 23.7 |
| 中位 TTFT | **0.7 s** | 2.5 s | **0.7 s** | 3.3 s |
| 中位 E2E | **2.9 s** | 6.5 s | **2.7 s** | 11.7 s |
| P99 E2E | **6.8 s** | 17.3 s | **5.5 s** | 33.2 s |

**4img/768p 的 RPS 几乎相同(0.99/1.93 vs 0.98/1.79)**,因为两边都受 input rate 限制(都没饱和 PD)。**但 agg 的 latency 显著更好**(TTFT 0.7s vs 2.5s,E2E 2.7-2.9s vs 6.5-11.7s),因为 agg 省去了 NIXL handoff 150 ms + 排队等 embedding。

---

## 3. 为什么 Agg 远胜 Disagg(在重负载场景)

### 3.1 Agg 没有 NIXL embedding handoff 这一关键瓶颈

Disagg 的核心瓶颈链:
```
Encoder ViT 编码 → 写入 NIXL ring buffer → PD 读 → admit 到 SGLang scheduler
                  ↑
                  容量 768 MB,8img/1080p 时只能存 5-6 个并发 embedding
                  PD 慢则 buffer 池耗尽 → 60s 后超时失败
```

Agg 完全省去这一步:**ViT 和 LLM 在同一个 sglang 引擎进程内**,vision encoder 输出直接喂给 LLM forward,无需 cross-process handoff。

### 3.2 Agg 的 GPU compute 利用率明显更高

| 工作负载 | rate | Agg total_tput | Disagg total_tput | Agg/Disagg |
|---|---:|---:|---:|---:|
| 8img/1080p | 1.0 | 7,787 | 3,781 | **2.06×** |
| 8img/1080p | 2.0 | 8,033 | 2,975 | **2.70×** |
| 16img/768p | 1.0 | 8,934 | 4,242 | **2.11×** |
| 16img/768p | 2.0 | 9,553 | 4,109 | **2.32×** |
| 8img/768p  | 2.0 | 9,481 | 4,282 | **2.21×** |

**Disagg 的 GPU 算力被 NIXL handoff 反向 backpressure 拖累一半**。Agg 没这个问题。

### 3.3 Agg 没有 max_running_requests=64 的有效约束

Disagg 在 8img/768p rate=2.0 时 PD running peak=63(撞 max_running=64 上限)。但 agg 在同一 case 下 PD 平均并发 55.7,**bench 客户端报告的并发(82.8 在 r=2.0)实际就是 sglang 引擎内的运行并发,因为没有 dynamo NIXL pipeline 在中间堆积**。

Agg 也用 max_running_requests=64,但因为它不被 NIXL pipeline 限速,实际能维持的稳态 in-flight 更高、batch 合并率更高。

### 3.4 Agg 没有 chunked prefill 与 NIXL admit 的相互干扰

Disagg 中 8img/768p rate=2.0 出现的"prefill batch 退化为单请求(40% single-req)"在 agg 里没有发生(因为 agg 的 scheduler 不需要等 NIXL embedding 才能 admit)。

---

## 4. RPS 模型验证(agg 数据)

公式 `RPS = total_tput / (input_len + output_len)`:

| 工作负载 | rate | total_tput | per_req tokens | 预测 RPS | 实测 RPS | 误差 |
|---|---:|---:|---:|---:|---:|---:|
| 8img/1080p | 1.0 | 7,787 | 16,420 + 162 = 16,582 | 0.470 | 0.47 | 0% |
| 8img/1080p | 2.0 | 8,033 | 16,420 + 187 = 16,607 | 0.484 | 0.49 | +1.2% |
| 16img/768p | 1.0 | 8,934 | 12,401 + 162 = 12,563 | 0.711 | 0.71 | 0% |
| 16img/768p | 2.0 | 9,553 | 12,401 + 167 = 12,568 | 0.760 | 0.76 | 0% |
| 8img/768p  | 1.0 | 5,704 | 6,238 + 110 = 6,348 | 0.899 | 0.90 | 0% |
| 8img/768p  | 2.0 | 9,481 | 6,238 + 116 = 6,354 | 1.492 | 1.49 | 0% |
| 4img/768p  | 1.0 | 3,233 | 3,158 + 110 = 3,268 | 0.989 | 0.99 | 0% |
| 4img/768p  | 2.0 | 6,306 | 3,158 + 110 = 3,268 | 1.929 | 1.93 | 0% |

**所有 8 组实测 RPS 与"`total_tput / per_req tokens`"模型完全吻合(误差 ≤ 1.2%)**。

注意 agg 的 total_tput 在加压下 **几乎线性增加**,而 disagg 的 total_tput 在加压下 **平稳或下降**:

| 工作负载 | Agg tput r=1.0 → 2.0 | Disagg tput r=1.0 → 2.0 |
|---|---:|---:|
| 8img/1080p | 7,787 → 8,033 (+3%) | 3,781 → 2,975 (-21%) |
| 16img/768p | 8,934 → 9,553 (+7%) | 4,242 → 4,109 (-3%) |
| 8img/768p  | 5,704 → 9,481 (**+66%**) | 4,470 → 4,282 (-4%) |
| 4img/768p  | 3,233 → 6,306 (**+95%**) | 3,198 → 5,856 (+83%) |

agg 的 GPU 在加压下能继续提升利用率,disagg 不能。

---

## 5. 部署建议(基于本次实测)

| 工作负载 | 推荐部署 | 原因 |
|---|---|---|
| **8img/1080p** | **TP=1 Agg** | 0.47 RPS vs disagg 0.23,2× 提升;无失败 |
| **16img/768p** | **TP=1 Agg** | 0.71 RPS vs disagg 0.34,2.1× 提升;无失败 |
| **8img/768p**  | **TP=1 Agg** | rate=1.0 中位 E2E 5.3s(disagg 84.9s),**16×** 加速 |
| **4img/768p**  | **TP=1 Agg 或 Disagg 都可**(两者基本相同),**Agg latency 更好** |

**全场景统一推荐:TP=1 Aggregate**(在单 GPU 同主机部署中)。

**Disagg 同主机的优势在哪里?** 本次实测下**没有**。Disagg 同主机的设计目标(独立 scale encoder/decoder)在多机或多 GPU 场景才有意义;在单 GPU + 单主机下,disagg 引入的 NIXL handoff overhead 完全没有被任何收益抵消。

**Disagg 真正的应用场景**:
- Cross-host: encoder 在 B70/4E/8E 集群,PD 在 H200,通过 RoCE 跨网传输 — 这时 NIXL handoff 是必要的
- Encoder 计算瓶颈不在同一 GPU 上(比如 ViT-L 跑在小 GPU 上,LLM 在大 GPU)
- Encoder 需要跑 mm-global-cache 共享多个 PD

---

## 6. 结果文件路径

所有 8 组 agg 实测结果在 `/hongming/res_samehost_agg_tp1_32b_gpu1/`:

| Workload | rate | Path | 起 UTC | 时长 | 成功 |
|---|---:|---|---|---:|:---:|
| 8img/1080p | 1.0 | `8img_1080p_rate1.0_np128_20260531_205956/` | 20:59:56 | 393 s | 128/128 |
| 8img/1080p | 2.0 | `8img_1080p_rate2.0_np128_20260531_210629/` | 21:06:29 | 384 s | 128/128 |
| 16img/768p | 1.0 | `16img_768p_rate1.0_np128_20260531_211306/` | 21:13:06 | 259 s | 128/128 |
| 16img/768p | 2.0 | `16img_768p_rate2.0_np128_20260531_211725/` | 21:17:25 | 246 s | 128/128 |
| 8img/768p  | 1.0 | `8img_768p_rate1.0_np128_20260531_212144/`  | 21:21:44 | 191 s | 128/128 |
| 8img/768p  | 2.0 | `8img_768p_rate2.0_np128_20260531_212455/`  | 21:24:55 | 137 s | 128/128 |
| 4img/768p  | 1.0 | `4img_768p_rate1.0_np128_20260531_212713/`  | 21:27:13 | 165 s | 128/128 |
| 4img/768p  | 2.0 | `4img_768p_rate2.0_np128_20260531_212958/`  | 21:29:58 | 104 s | 128/128 |

Agg worker 日志: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/agg_epd_worker_20260531_205248.log`

每个目录含:
- `benchmark_output.json` — 原始指标
- `results.txt` — 完整 bench stdout
- `bench_start_utc.txt` — 起始时间

## 参考文档

- `same_host_3_cases_problem_analysis_zh_v01.md` — 同主机 disagg 4 cases × 2 rates 的瓶颈分析(本文档的对比对象)
- `same_host_768p_problem_analysis_zh.md` — 768p 工作负载的 disagg 瓶颈
- `same_host_problem_analysis_zh.md` — 1080p 工作负载的 disagg 瓶颈根因
- `1080p_sweep_three_way.md` — TP=1 / TP=2 / disagg 的早期对比
