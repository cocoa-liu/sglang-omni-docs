# 设置理解生成阶段一次编译计算的批大小上限：thinker_torch_compile_max_bs

本次任务是检查：换用 NPU（昇腾计算设备）后，理解生成阶段的“编译批大小上限”设置能否在 NPU 上起作用。

## 1. 验证对象与结论

Qwen3-Omni 的 thinker 负责理解输入、推理和生成文字等结果；语音生成还会使用其他阶段。

这里的“编译”是先把部分模型计算整理成便于执行的形式，再用这种形式运行；本参数需要与编译开关（`thinker_torch_compile`）一起看。

模型一次计算可以同时处理多条请求，这些请求合起来叫一批，条数就是批大小。本参数限制与编译计算有关的批大小范围：batch_size ≤ `torch_compile_max_bs` 走编译图，> `torch_compile_max_bs` 走 eager。它只作用于 thinker 阶段。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `thinker_torch_compile_max_bs` | 不需要，依赖 `enable_torch_compile`；后者在 NPU 上不可用，最终受 torch_npu/ATB 阻塞 | **不需要开发支持**（sglang-omni 侧）。路由/取值校验可用；实际编译收益无法验证。 |

**当前定位**：各模型的失败点不同，不是同一个根因；Qwen3-Omni 和 Qwen3-ASR 在绕过各自问题后都会卡在 torch_npu/ATB。`enable_torch_compile` 在 NPU 上当前跑不通，因此 `max_bs` 无法完成一次成功的编译运行。

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | Ascend910（64GB HBM ×8） |
| 软件 | CANN 9.1 + SGLang 0.5.18 + sglang_omni 0.1.4（c828）+ torch_npu 2.10.0.post4 |
| 模型 | Qwen3-Omni-30B-A3B-Instruct（TP=2 thinker） |
| 容器 | `codex-newmodels-base`（113.46.15.88） |

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 参数路由与取值校验 | 用 ConfigManager 设 `thinker.engine.torch_compile_max_bs=64` 和 `=0`，读回 overrides | 64 读回正确，0 被拒绝（ge=1） | 通过：64 读回正确，0 被拒绝 | — |
| 端到端文本生成（disable_cuda_graph=true） | 设 `torch_compile_max_bs=8` + `enable_torch_compile=true` + `disable_cuda_graph=true`，发文本和图片请求 | HTTP 200 + 编译发生 | 通过 HTTP 200，但 torch.compile 从未被调用 | 静默空跑，`max_bs` 无从生效 |
| 批大小选择（bs=32） | `torch_compile_max_bs=2` + `disable_cuda_graph=false`，观察 capture 的 shape | bs=32 > 2，应走 eager | 观察到 bs=32 不在 `compile_bs`，未调用 torch.compile | 证明 `max_bs` 的选择逻辑生效；该 shape 仍崩在 ATB |
| 批大小选择（bs=1/2） | 同上，限制 capture bs=1/2，使 bs≤2 走编译 | bs=1/2 走 `npugraph_ex` 编译 | 观察到 bs=2 进入 dynamo / npugraph_ex，但连续失败，最终 ATB | 编译路径本身未通过，`max_bs` 的实际收益无法验证 |

### 2.3 当前处理建议

- 在 Qwen3-Omni 模型层默认关闭 torch.compile（已实现），覆盖显式 true 和未设置两种情况；
- `max_bs` 的取值/路由逻辑本身没问题，等 `enable_torch_compile` 的上游问题修复后再补验批大小边界；
- 真正的修复按层提给 SGLang / sgl-kernel-npu / triton-ascend / torch_npu。

> 拦截范围的理由见 `thinker_torch_compile_2.md` 的 2.5：各模型的失败点不同，不是同一个根因；仓库习惯也是各模型自己处理编译开关，所以本次只在 Qwen3-Omni 模型层拦截；不是 ATB 的模型见 `thinker_torch_compile_2.md` 的 2.6。

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 参数路由 | ✓ `test_pipeline.py::test_s2pro_torch_compile_max_bs_rejects_non_positive` | ✓ |
| 取值校验（0 被拒，ge=1） | ✓ | ✓ |
| 批大小选择（bs > max_bs 走 eager） | ✗ 依赖编译路径 | ✓ 观察到 bs=32 未编译 |
| 批大小实际收益（编译 vs eager） | ✗ | ✗ 编译路径被 dynamo/ATB 阻塞 |

UT 只验路由和取值校验。批大小的实际收益需要 torch.compile 先能跑通；目前没有“真实模型在 NPU 上完成 torch.compile”的正例。
