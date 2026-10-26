# 一次设置多个阶段的编译开关：torch_compile

本次任务是检查：换用 NPU（昇腾计算设备）后，给各个适用的生成阶段开启“编译计算”后，设置能否在 NPU 上正常工作。

## 1. 验证对象与结论

Qwen3-Omni 把任务分成多个阶段：thinker 负责理解、推理和生成文字等结果，talker 负责生成声音所需的表示。

模型要反复做很多相似计算。“编译”可以理解为先把其中一些计算整理成便于执行的形式，再用这种形式运行。`torch_compile` 控制是否尝试使用这条路径，并把同类开关统一发给多个适用的生成阶段（thinker + talker_ar）。它不意味着服务里的所有组件都被编译。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `torch_compile` | 不需要，广播到 thinker/talker 后同样 dynamo 崩溃，最终受 torch_npu/ATB 阻塞 | **不需要开发支持**（sglang-omni 侧）。函数级透传可用；当前定位到多层上游兼容问题，最终阻塞在 ATB。 |

**当前定位**：各模型的失败点不同，不是同一个根因；Qwen3-Omni 最终卡在 torch_npu/ATB。`enable_torch_compile` 在 NPU 上当前跑不通，因此本次不启用。

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | Ascend910（64GB HBM ×8） |
| 软件 | CANN 9.1 + SGLang 0.5.18 + sglang_omni 0.1.4（c828）+ torch_npu 2.10.0.post4 |
| 模型 | Qwen3-Omni-30B-A3B-Instruct（TP=2 thinker + talker_ar） |
| 容器 | `codex-newmodels-base`（113.46.15.88） |

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 参数路由与取值校验 | 用 ConfigManager 设广播式 `enable_torch_compile=true`，读回 thinker 和 talker_ar 的 overrides | 两个 stage 都收到 True | 通过：两个 stage 都读到 True | — |
| 端到端文本生成（disable_cuda_graph=true） | 设 `torch_compile=true`（广播）+ `disable_cuda_graph=true`，启动后发文本和图片请求 | HTTP 200 + 编译发生 | 通过 HTTP 200，但 torch.compile 从未被调用 | 静默空跑：graph capture 被跳过 |
| 编译是否真正发生 | 开 `TORCH_LOGS=+dynamo`，查缓存和 dynamo 输出 | 有编译产物或 dynamo 输出 | 未通过：缓存为空，dynamo 零行输出 | 同 `thinker_torch_compile` |
| 端到端启动（disable_cuda_graph=false，bs=2） | 设 `torch_compile=true`（广播）+ `disable_cuda_graph=false`，限制 capture bs=1/2 | 完成编译/图捕获 | 未通过：dynamo 连续失败，最终 ATB 阻塞 | 逐层与 thinker 相同 |
| 第一层：`is_cuda()` | dynamo 追踪 `fused_op._platform_key()` | dynamo 能追踪 | 未通过：`torch.cuda.is_available()` 被 `transfer_to_npu` 替换，返回 bool | 广播到各 stage 后行为一致 |
| 第二层：`is_cutedsl()` | 绕过第一层后继续追踪 | dynamo 能追踪 | 未通过：`unquant.py:239` → `builtins.method` | 同 Fun-CosyVoice3；不是 fake tensor |
| 后续 Triton/MoE/通信层 | 继续逐层定位 | dynamo 能追踪 | 未通过：mrope、split_qkv、l1_norm、MoE permute、symm_mem | 归属 SGLang / sgl-kernel-npu / triton-ascend |
| 最终 ATB | 绕过以上层后重新跑 | npugraph_ex 完成捕获 | 未通过：`PagedAttentionOperation setup failed!` | 属 torch_npu/ATB；`run_eagerly=False` 同样失败 |

### 2.3 逐层结论

与 `thinker_torch_compile_2.md` 的 10 层表相同：`is_cuda`、`is_cutedsl`、mrope Triton、split_qkv/l1_norm、MoE permute、symm_mem、ATB。广播只是把开关同时发给 thinker 和 talker，不改变各层的根因。

### 2.4 当前处理建议

- 在 Qwen3-Omni 模型层默认关闭 torch.compile（`disable_torch_compile_on_npu` 同时覆盖 thinker 和 talker）；
- 不在通用 resolver 里按“所有 NPU 模型都不支持”做全局拦截；
- 真正的修复按层提给 SGLang / sgl-kernel-npu / triton-ascend / torch_npu。

### 2.5 为什么只在 Qwen3-Omni 模型层处理

与 `thinker_torch_compile_2.md` 的 2.5 相同：各模型的失败点不同，不是同一个根因（Qwen3-ASR 是 `get_device_properties` / Triton，Fun-CosyVoice3 是 `is_cutedsl()`，Whisper 是模型加载阶段）；ATB 只是已测 AR 模型的共同最终阻塞。仓库现有习惯也是各模型自己处理编译开关（Higgs 检查 `is_npu()`、Ming/Qwen3-TTS 直接 raise、Qwen3-ASR/Whisper 各自设默认值）。因此本次只在 Qwen3-Omni 的 thinker/talker builder 里拦截，不改通用 resolver；不是 ATB 的模型见 `thinker_torch_compile_2.md` 的 2.6。

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 参数路由（广播到各 stage） | ✓ `test_cli.py::test_torch_compile_dotted_flags_reach_resolved_sglang_args` | ✓ |
| 取值校验 | ✓ | ✓ |
| NPU 上模型级关闭 | ✓ `test_sglang_ar_budget.py::test_torch_compile_is_disabled_on_npu` | ✓（配置层） |
| torch.compile 是否被调用 | ✗ UT 不启动 SGLang | ✓ 空跑已验 |
| 广播是否送到各 stage | ✓ 路由验证 | ✓ |
| dynamo / ATB 实际行为 | ✗ | ✓ 已定位到 ATB |

UT 只验参数路由和模型级开关。广播写到各 stage 后的 actual compile 行为需要端到端测试；目前没有“真实模型在 NPU 上完成 torch.compile”的正例。
