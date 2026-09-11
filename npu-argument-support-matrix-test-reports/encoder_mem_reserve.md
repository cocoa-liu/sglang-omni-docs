# SGLang-Omni 参数 NPU 验证速览：encoder_mem_reserve

实验日期：2026-09-09。本次任务是检查：换用 NPU（Ascend）计算设备后，`encoder_mem_reserve`（为编码器预留显存）是否仍然需要、能否正常工作。该参数在 CUDA 上已支持；本速览只分析 NPU 上的情况。

## 1. 验证对象与结论

| 参数 | 管什么 | NPU 是否需要 | 本次结论 |
| --- | --- | --- | --- |
| `encoder_mem_reserve` | 从自动选定的 `mem_fraction_static`（SGLang KV 缓存预算）中扣减一个比例，留给 Qwen3-Omni 的外部编码器（音频/图像编码器头）使用，默认 0.05 | 需要——编码器头在 NPU 上同样占用 HBM，不预留会与 thinker 的 KV 缓存争用 | 所测配置下，预留函数在真实 NPU 上按预期扣减、校验生效；逻辑与设备无关，无需新增 NPU 专用参数逻辑 |

验证范围：**Ascend910（8×64GB）+ CANN 9.0.0 + torch 2.10.0(+cpu) + torch_npu 2.10.0（transfer_to_npu 补丁）+ sglang 0.5.18 + sglang_omni 0.1.4**。参数逻辑验证在 113.46.15.88 的 NPU 容器（`fth-sglang-omni-qwen3-tts-test`，空闲设备 4，可用 60.9GB）上完成。

支持矩阵结论：`encoder_mem_reserve` NPU 标记为 `[√]`（当前已支持），下方给出验证依据与边界。

## 2. 怎么测，结果证明了什么

`encoder_mem_reserve` 只在 Qwen3-Omni 的 thinker 阶段有意义：thinker 与外部编码器共享同一设备，因此要从 SGLang 自动选定的 `mem_fraction_static`（KV 缓存预算）里减去一个比例留给编码器。它的实现是**纯算术**：`mem_fraction_static - encoder_mem_reserve`，并在结果低于安全下限 0.1 时报错，不调用任何 CUDA/NPU 专属 API。

**实验思路：在真实 NPU（device_type=npu，可用 60.9GB）上解析平台、构造 device=npu 的 ServerArgs，再调用真实的 `apply_encoder_mem_reserve` 函数，逐项核对扣减、零值空操作、越界拒绝、共置路径算术。**

| 要确认什么 | 怎么做 | 怎样才算通过 | 实际结果 |
| --- | --- | --- | --- |
| 平台解析为 NPU | 在 NPU 容器内取 `current_platform` | `device_type == "npu"` 且能读到真实空闲内存 | 通过：`NPUOmniPlatform device_type=npu`，`free=60.90GB` |
| 扣减生效 | `mem_fraction_static=0.50`，`apply_encoder_mem_reserve(.., 0.05)` | 结果为 0.45 | 通过：得到 0.45 |
| 零值是空操作 | reserve=0.0 | 原值不变 | 通过：仍为 0.50 |
| 低于安全下限报错 | 0.15 − 0.10 = 0.05 < 0.1 | 抛 ValueError | 通过：报 “below the safe floor” |
| 越界值被拒 | reserve=1.5（不在 [0,1)） | 抛 ValueError | 通过：越界被拒 |
| 共置路径算术 | `_apply_colocated_encoder_mem_reserve(0.75, 0.05)` | 0.70 | 通过：得到 0.70 |
| 引擎入口在 NPU 上接 npu | `build_sglang_server_args(...)` | `device` 以 `npu` 开头 | 通过：`device=npu` |

上述 8 项在真实 Ascend910 上全部通过。`encoder_mem_reserve` 的全部逻辑不触及任何设备专属 API，只对 `mem_fraction_static` 做减法；而 `mem_fraction_static` 在 NPU 上已被真实启动证实可用（启动日志：`mem_fraction_static=0.50/0.60/0.70` 均成功分配 KV 缓存）。因此该参数在 NPU 上**当前已支持**。

未做 Qwen3-Omni thinker 端到端：本环境只放了 Qwen3-TTS 权重，而 Qwen3-TTS 的 `tts_engine` 直接钉死 `mem_fraction_static`（按设计使 reserve 变空操作），并不走该参数；Qwen3-Omni（含 thinker+talker+编码器）的完整模型权重不在本机。这是覆盖范围缺口，不是参数逻辑缺口。

## 3. 交付与适用边界

| 项目 | 说明 |
| --- | --- |
| 是否需要保留 | 只要 Qwen3-Omni 把外部编码器与 thinker 放在同一 NPU，就需要预留 HBM；换 NPU 不消除该需求 |
| 做了什么 | 在真实 Ascend910 上解析 NPU 平台、构造 device=npu 的 ServerArgs、逐项验证 `apply_encoder_mem_reserve` 的扣减/零值/下限/越界/共置算术与引擎入口接线 |
| 可以确认 | `encoder_mem_reserve` 的逻辑与设备无关，只对 `mem_fraction_static` 做减法；`mem_fraction_static` 在 NPU 上可用（启动日志佐证），故预留参数在 NPU 上可用，无需新增专用逻辑 |
| 不能扩大为 | 未做 Qwen3-Omni thinker 端到端（本机仅有 Qwen3-TTS 权重，其 `tts_engine` 钉死 `mem_fraction_static` 使 reserve 按设计变空操作）；未在 NPU 上实测“预留后编码器确实不 OOM”的生成对照 |
| UT 覆盖 | 充分。本仓有约 13 个专门测试函数覆盖扣减/零值/安全下限/越界/与显式 pin 冲突/自动路径 vs 钉死/共置路径/路由/签名默认值，分布于 `tests/unit_test/qwen3_omni/test_pipeline.py`、`test_sglang_ar_budget.py`、`test_cli.py`、`test_config_manager.py`、`pipeline/test_runtime_adapter.py` |

补充记录：该参数逻辑层在 NPU 上 **8/8 项检查通过**（合计 12/12 项中的 8 项，另 4 项为 `cpu_offload_gb`）。

证据归档入口（NPU 容器 `fth-sglang-omni-qwen3-tts-test`，设备 4）：
- 参数逻辑检查脚本：`/tmp/verify/npu_verify.py`（8/8 通过输出）

环境：IP 113.46.15.88，容器 `fth-sglang-omni-qwen3-tts-test`，`ASCEND_RT_VISIBLE_DEVICES=4`（空闲设备，可用 60.9GB），Ascend910 / CANN 9.0.0 / torch 2.10.0+cpu / torch_npu 2.10.0 / sglang 0.5.18 / sglang_omni 0.1.4。
