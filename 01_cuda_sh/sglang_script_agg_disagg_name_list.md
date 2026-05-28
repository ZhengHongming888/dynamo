# SGLang start scripts on H200 — name list

Catalog of every `start_*.sh` under `/hongming/dynamo/01_cuda_sh/` that launches a
dynamo-SGLang server on this H200 host. Use this to pick the right script for
the agg/disagg config + model size + role you want.

**Conventions used everywhere:**
- IP: `172.26.46.75` (this giga01 H200 host) unless noted
- Frontend port: `7001`, NATS: `14222`, etcd: `12379`
- All scripts launch NATS + etcd + frontend + worker(s) — except encoder-only
  scripts, which assume NATS/etcd/frontend already running on a remote PD host

---

## Aggregated EPD (TP=1 / TP=2 on the same H200, all-in-one worker)

The `aggregate_epd_server` scripts launch encoder + prefill + decode in a
single SGLang worker (no NIXL transfer).

### Qwen3-VL-32B-Instruct-FP8 (dense)

| Script | TP | GPUs | Notes |
|---|---:|---|---|
| `agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh` | 1 | GPU 4 | TP=1 baseline, mem_fraction=0.85, max_running=40 |
| `agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh` | 2 | GPUs 4,5 | TP=2 NVLink (NV18), mem_fraction=0.85, max_running=25 |
| `agg_h200_32b/start_h200_aggregate_epd_server_32b_two_tp1.sh` | 1×2 | GPUs 4,5 | Two independent TP=1 workers (DP), no TP-sharding |

### Qwen3-VL-8B-Instruct (BF16 dense)

| Script | TP | GPUs | Notes |
|---|---:|---|---|
| `agg_h200_8b/start_h200_aggregate_epd_server_8b_tp1.sh` | 1 | GPU 4 | 8B baseline |
| `agg_h200_8b/start_h200_aggregate_epd_server_8b_tp2.sh` | 2 | GPUs 4,5 | 8B TP=2 NVLink |

### Qwen3.5-35B-A3B (BF16 MoE, multimodal)

| Script | TP | GPUs | Notes |
|---|---:|---|---|
| `agg_h200_35b/start_h200_aggregate_epd_server_35b_tp1.sh` | 1 | GPU 4 | mem_fraction=0.75, max_running=40, `--router-mode round-robin`, `--linear-attn-backend triton` |
| `agg_h200_35b/start_h200_aggregate_epd_server_35b_tp2.sh` | 2 | GPUs 4,5 | mem_fraction=0.75, max_running=25, `--router-mode round-robin`. **TP=2 ties or loses to TP=1 for this MoE — see results in `/hongming/res20_agg_h200_35b/`.** |

> **Note for 35B:** mem_fraction must stay at 0.75 (0.85 OOM-kills the
> scheduler during cuda graph capture). Frontend must use
> `--router-mode round-robin` (kv-router panics with
> `block_size must be greater than 1` because hybrid linear-attention forces
> sglang `page_size=1`).

---

## Disaggregated (encoder + PD as separate workers, NIXL-transferred embeddings)

### Same-host (encoder + PD on the same H200 host)

#### Qwen3-VL-32B-Instruct-FP8

| Script | Encoder GPU | PD GPU(s) | PD-TP | Purpose |
|---|---:|---|---:|---|
| `disagg_h200_32b/start_disagg_h200_32b_combined.sh` | 4 | 5 | 1 | **Canonical 2-GPU same-host disagg.** NIXL-READ mode, max_running=64, chunked_prefill=16384. mem_fraction=0.85. |
| `disagg_h200_32b/start_disagg_h200_32b_pd_tp2.sh` | 4 | 5,7 | 2 | 3-GPU disagg (encoder + PD-TP=2). Used for the round-3 PD-TP=2 measurement. |
| `disagg_h200_32b/start_disagg_h200_32b_pd_tp2_rdma.sh` | 4 | 5,7 | 2 | Same as `_pd_tp2.sh` but `UCX_TLS=rc,ud,rc_verbs,ud_verbs,gdr_copy` (no cuda_ipc) — forces RDMA path on same-host. Round-3 RDMA-only test. |
| `disagg_h200_32b/start_disagg_h200_32b_pd_tp2_tuned.sh` | 4 | 5,7 | 2 | Same as `_pd_tp2.sh` plus encoder-side tuning patches (round 1+2 attempts). |
| `disagg_h200_32b/start_disagg_h200_32b_pd_tp2_gpubuf.sh` | 4 | 5,7 | 2 | Round-4 GPU NIXL descriptor patch test. Lower mem_fraction=0.70, max_running=16. **OOMs in practice** — kept for the documented round-4 attempt. |

#### Qwen3-VL-8B-Instruct (BF16)

| Script | Encoder GPU | PD GPU | PD-TP | Notes |
|---|---:|---:|---:|---|
| `disagg_h200_8b/start_disagg_h200_8b_combined.sh` | 4 | 5 | 1 | 8B same-host disagg, mirrors the 32B `_combined` script |

#### Older / experimental 32B disagg variants

These exist as snapshots from earlier sessions; differ from the current
`disagg_h200_32b/` scripts in IP (`172.26.46.162` instead of `.75`) and lack
the later patches. Use `disagg_h200_32b/` for new work.

| Script | Role |
|---|---|
| `disagg_h200_32b_bf/start_h200_encode_32b_fp8.sh` | encoder-only, 32B-FP8, paired with a remote PD |
| `disagg_h200_32b_bf/start_sglang_pd_cuda_32b_fp8.sh` | PD-only, 32B-FP8 |
| `disagg_h200_32b_bf2/start_h200_encode_32b_fp8.sh` | encoder-only (variant 2) |
| `disagg_h200_32b_bf2/start_sglang_pd_cuda_32b_fp8.sh` | PD-only (variant 2) |
| `disagg_h200_32b/start_h200_encode_32b_fp8.sh` | encoder-only (legacy, IP=.162) |
| `disagg_h200_32b/start_sglang_pd_cuda_32b_fp8.sh` | PD-only (legacy, IP=.162) |

---

### Cross-host (PD on this H200, encoder on a remote host)

#### Qwen3-VL-32B-Instruct-FP8 PD on giga01, encoder on B70

| Script | Encoder | PD GPU | PD-TP | Notes |
|---|---|---:|---:|---|
| `disagg_h200_32b/start_sglang_pd_cuda_32b_fp8_giga01.sh` | remote (B70 over RoCE) | 4 | 1 | **Canonical cross-host PD launcher.** Used for the 4E patched sweep. NATS/etcd bind to 0.0.0.0 (B70-reachable). UCX_TLS uses RDMA only (no cuda_ipc). Pairs with B70 `start_sglang_pd_xpu_32b_b70_4E.sh` over RoCE 192.165.123.0/24. |

#### Qwen3.5-35B-A3B PD on giga01, encoder on B70

| Script | Encoder | PD GPU | PD-TP | Notes |
|---|---|---:|---:|---|
| `disagg_h200_35b/start_sglang_pd_cuda_35b_giga01.sh` | remote (B70 over RoCE) | 4 | 1 | **35B cross-host PD launcher.** Mirrors the 32B equivalent but with 35B-specific tweaks: BF16 (no FP8 KV cache), `mem_fraction_static=0.75` (0.85 OOM-kills scheduler), `--router-mode round-robin` (kv-router panics on `page_size=1`), `--linear-attn-backend triton` + `--attention-backend fa3` for the hybrid attention layers. Requires a matching 35B encoder script on B70 (32B B70 encoder won't work — different vision-tower dimensions). |

> **Companion encoder scripts live on the B70 host** (under
> `/hongming/dynamo/02_xpu_sh/`), not on this H200 — they are not part of
> this catalog.

---

### Encoder-only on this H200 (PD lives on a remote host)

Only relevant if you are using this H200 as the encoder side of a
cross-host setup with the PD elsewhere. Requires NATS/etcd/frontend to be
running on the remote PD host.

| Script | Model | GPU |
|---|---|---:|
| `disagg_h200_32b/start_h200_encode_32b_fp8.sh` | Qwen3-VL-32B-FP8 | 0 (default `CUDA_DEVICE=0`) |
| `disagg_h200_32b_bf/start_h200_encode_32b_fp8.sh` | Qwen3-VL-32B-FP8 | 0 |
| `disagg_h200_32b_bf2/start_h200_encode_32b_fp8.sh` | Qwen3-VL-32B-FP8 | 0 |

---

### 3B class (smaller VL model, exploratory)

Qwen2.5-VL-3B-Instruct, sized for testing the dynamo plumbing without
loading a 32B-class model. Uses the legacy `IP_LOCAL=172.26.46.162`
convention.

| Script | Role |
|---|---|
| `3b/start_h200_encode_3b.sh` | encoder-only (this H200, paired with remote PD) |
| `3b/start_sglang_pd_cuda_3b.sh` | PD-only (3B model, NIXL-disagg) |

---

## Quick reference by task

**"I want same-host TP=1 agg, 32B-FP8":**
→ `agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh`

**"I want same-host TP=2 agg, 32B-FP8":**
→ `agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh`

**"I want same-host disagg, 32B-FP8":**
→ `disagg_h200_32b/start_disagg_h200_32b_combined.sh`

**"I want cross-host disagg PD here, encoder on B70":**
→ `disagg_h200_32b/start_sglang_pd_cuda_32b_fp8_giga01.sh` (32B-FP8)
→ `disagg_h200_35b/start_sglang_pd_cuda_35b_giga01.sh` (35B-A3B; needs matching B70 35B encoder)

**"I want 35B-A3B MoE":**
→ `agg_h200_35b/start_h200_aggregate_epd_server_35b_tp1.sh` (TP=1 — recommended; TP=2 doesn't help for this MoE)

**"I want 8B dense":**
→ `agg_h200_8b/start_h200_aggregate_epd_server_8b_tp1.sh` (or `_tp2.sh`)

---

## Companion bench scripts (in `/hongming/dynamo/`)

For reference, the bench scripts most commonly used with these servers are
under `/hongming/dynamo/test_sglang_*.sh`. Notable ones:

- `test_sglang_mult_rates_32b_1080p_np64_over_rates.sh` — 32B 1080p sweep
- `test_sglang_mult_rates_35b_1080p_np64_over_rates.sh` — 35B 1080p sweep
- `test_sglang_8b_1080p_np64_over_rates.sh` — 8B 1080p sweep
- `test_sglang_8img_768p.sh` / `_sweep.sh` — 768p workloads
- `test_sglang_8img_4k.sh` — 4K workload (requires `DYN_HTTP_BODY_LIMIT_MB=256`)
- `test_sglang_decode_heavy.sh` — decode-heavy (1024 output tokens)
- `test_sglang_cache_hit.sh` — fixed-image cache-hit workload

All of these honor `RESULT_BASE` env override; otherwise they write to
`/hongming/res*/...`.
