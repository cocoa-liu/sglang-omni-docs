# 开启理解生成阶段的编译计算：thinker_torch_compile

本次任务是检查：换用 NPU（昇腾计算设备）后，给理解生成阶段开启“编译计算”后，设置能否在 NPU 上正常工作。

## 1. 验证对象与结论

Qwen3-Omni 的 thinker 负责理解输入、推理和生成文字等结果；语音生成还会使用其他阶段。

模型要反复做很多相似计算。"编译"可以理解为先把其中一些计算整理成便于执行的形式，再用这种形式运行。`talker_torch_compile` 控制是否尝试使用这条路径，目的是便于优化计算，但开启后不保证一定更快。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `thinker_torch_compile` | 不需要，Qwen3-Omni 在 NPU 上 dynamo 崩溃，最终受 torch_npu/ATB 阻塞 | **不需要开发支持**（sglang-omni 侧）。函数级透传可用；当前定位到多层上游兼容问题，最终阻塞在 ATB。 |

**当前定位**：各模型的失败点不同，不是同一个根因。Qwen3-Omni 的失败点是 `is_cuda()`、`is_cutedsl()`（`builtins.method`）、Triton driver、MoE permute、symm_mem，最终阻塞在 ATB `PagedAttentionOperation`。

跨模型对照：Qwen3-Omni 的失败点是 `is_cuda()` / `is_cutedsl()` / Triton driver / MoE permute / symm_mem，最终卡在 ATB；Qwen3-ASR 的失败点是 `get_device_properties` / Triton，绕过这些后同样卡在 ATB；Fun-CosyVoice3 的失败点是 `is_cutedsl()`；Whisper 在模型加载阶段就因 `get_device_capability` 失败。

这些模型的失败点不同，不是同一个根因。ATB 只是已测 AR 模型的共同最终阻塞，不是所有模型的共同根因。

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | Ascend910（64GB HBM ×8） |
| 软件 | CANN 9.1 + SGLang 0.5.18 + sglang_omni 0.1.4（c828）+ torch_npu 2.10.0.post4 |
| 模型 | Qwen3-Omni-30B-A3B-Instruct（TP=2 thinker）；对照 Qwen3-ASR-1.7B、Fun-CosyVoice3 |
| 容器 | `codex-newmodels-base`（113.46.15.88） |

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 参数路由与取值校验 | 用 ConfigManager 设 `thinker.engine.enable_torch_compile=true`，读回 overrides | 值读回正确为 True | 通过：读回 True | — |
| 端到端文本生成（disable_cuda_graph=true） | 设 `enable_torch_compile=true` + `disable_cuda_graph=true`，启动后发文本和图片请求 | 编译发生且 HTTP 200 | 通过 HTTP 200，但 torch.compile 从未被调用 | 静默空跑：graph capture 被跳过，`patch_model_npu` 未进入 |
| 编译是否真正发生 | 开 `TORCH_LOGS=+dynamo`，查 torchinductor 缓存和 dynamo 输出 | 有编译产物或 dynamo 输出 | 未通过：缓存 delta=0，dynamo 零行输出 | 证明上一行是空跑 |
| 端到端启动（disable_cuda_graph=false，bs=2） | 设 `enable_torch_compile=true` + `disable_cuda_graph=false` + `torch_compile_max_bs=2` + `max_running_requests=2`，让 capture 只出现 bs=1/2 | torch.compile 被调用并完成图捕获 | 未通过：torch.compile 被调用，dynamo trace 连续失败 | 逐层见下 |
| 第一层：`is_cuda()` | dynamo 追踪到 `sglang/kernels/fused_op.py:180` 的 `_platform_key()` | dynamo 能追踪 | 未通过：`is_cuda()` → `torch.cuda.is_available()`（被 `transfer_to_npu` 替换）返回 bool，报 `torch.* op returned non-Tensor` | 已确认；修法：把 `is_npu()` 放到 `is_cuda()` 前（SGLang） |
| 第二层：`is_cutedsl()` | 绕过第一层后继续追踪 | dynamo 能追踪 | 未通过：`unquant.py:239` `get_bf16_gemm_backend().is_cutedsl()` → `Unexpected type in sourceless builder`，`value_type=builtins.method` | 与 Fun-CosyVoice3 同因；不是 fake tensor |
| 后续 Triton/driver 层 | 继续逐层定位 | dynamo 能追踪 | 未通过：`mrope` 的 Triton kernel → `NPUDriver.get_current_device`；`split_qkv_rmsnorm_rope` → `get_device_properties`；`l1_norm` 同类问题 | 归属 SGLang / sgl-kernel-npu / triton-ascend；测试中临时用 custom op/常量绕过 |
| MoE 与通信层 | 继续逐层定位 | dynamo 能追踪 | 未通过：MoE pre/post permute 用 namedtuple 字段做 dict key；`triton_symm_mem_ag` 的 CUDA-only 路径 | 归属 SGLang |
| 最终 ATB | 绕过以上层后重新跑 bs=2 | npugraph_ex 完成编译/捕获 | 未通过：`PagedAttentionOperation setup failed!`（`torch_npu/dynamo/torchair/_acl_concrete_graph/acl_graph.py:830 fx_run_eagerly`） | 属 torch_npu/ATB；`run_eagerly=False` 时在 `acl_graph.py:1219 capture` 同样失败，说明与 eager/capture 无关 |

### 2.3 逐层结论

| 层 | 崩溃点 | 归属 |
| --- | --- | --- |
| 1 | `fused_op.py:180 is_cuda()` | SGLang |
| 2 | `unquant.py:239 is_cutedsl()`（`builtins.method`） | SGLang |
| 3 | `mrope` Triton → `NPUDriver.get_current_device` | SGLang / triton-ascend |
| 4 | `split_qkv_rmsnorm_rope` → `get_device_properties` | sgl-kernel-npu / triton-ascend |
| 5 | `l1_norm` → `get_device_properties` | sgl-kernel-npu / triton-ascend |
| 6 | `l1_norm_kernel` → `NPUDriver.get_current_device` | sgl-kernel-npu / triton-ascend |
| 7 | MoE pre/post permute dict key | SGLang |
| 8 | `triton_symm_mem_ag` 的 `torch.cuda.is_available` | SGLang |
| 9 | `triton_symm_mem_ag.create_state` 的 CUDA-only 路径 | SGLang |
| 10 | ATB `PagedAttentionOperation` | torch_npu / ATB |

### 2.4 当前处理建议

- 在 Qwen3-Omni 模型层默认关闭 torch.compile（已实现：`disable_torch_compile_on_npu`），覆盖“显式 true”和“未设置默认 true”两种情况；
- 不再在通用 resolver 里按“所有 NPU 模型都不支持”做全局拦截；
- 真正的修复按上表分别提给 SGLang / sgl-kernel-npu / triton-ascend / torch_npu。

### 2.5 为什么只在 Qwen3-Omni 内部拦截

不是因为 NPU 上有一个公共方法必然失败，而是因为：

- **各模型的失败点不同，不是同一个根因**：Qwen3-Omni 是 `is_cuda()` / `is_cutedsl()` / Triton driver / MoE permute / symm_mem，最终卡在 ATB；Qwen3-ASR 是 `get_device_properties` / Triton，最终也卡在 ATB；Fun-CosyVoice3 是 `is_cutedsl()`；Whisper 是模型加载阶段的 `get_device_capability`。ATB 只是已测 AR 模型的共同最终阻塞，不是所有模型的共同根因，也不足以支撑“所有 NPU 模型统一关闭”。
- **仓库现有习惯是各模型自己处理编译开关**：
  - `higgs_tts/stages.py:467`：模型内检查 `current_platform.is_npu()`，NPU 上跳过该 codec 的 `torch.compile`；
  - `ming_tts/engine_builder.py:197-198`、`qwen3_tts/engine_builder.py:332-333`：模型内检查到 `enable_torch_compile` 为真时直接 raise；
  - `ming_tts/__init__.py:11`、`qwen3_tts/__init__.py:13`：在 `ModelCapabilities` 里声明 `supports_torch_compile=False`；
  - `qwen3_asr/config.py:96`、`qwen3_asr/engine_builder.py:377`、`whisper_asr/engine_builder.py:391`：各自设置或调整默认值。
- 因此本次把策略放在 `qwen3_omni/stages.py` 的 thinker/talker builder，只影响 Qwen3-Omni；通用 resolver 不做平台级拦截。其他模型如果要关，应按同样方式在各自模型层处理，或等上游修复。

### 2.6 不是 ATB 的模型：Whisper 的四层定位

不是所有模型都是因为 ATB 失败。Whisper 就是一个例子：它在 NPU 上定位到的四层失败，全部发生在 ATB 之前。

| 层 | 失败点 | 归属 |
| --- | --- | --- |
| 1 | `get_device_capability()` 返回 None（`transfer_to_npu`），发生在模型加载阶段 | SGLang `maybe_downgrade_dtype_for_legacy_gpu` |
| 2 | decoder cross-attention 传入 `k=None` / `v=None`，NPU graph 路径调用 `set_kv_buffer` 报 `NoneType.dtype` | SGLang `ascend_backend.forward_decode_graph`（compile 路径下不触发） |
| 3 | `torch.empty_like` 被 `transfer_to_npu` 的 `torch.compiler.disable` 包裹，dynamo fullgraph 失败 | torch_npu `transfer_to_npu` |
| 4 | `ascend_torch_native_backend.py:289` 的数据相关切片 `req_to_token[..., atten_start_kv:atten_end_kv]`；`fullgraph=True` 无法特化，`fullgraph=False` 退化为 host 同步，graph capture 失败 | SGLang / dynamo |

测试中第 1-3 层用临时补丁绕过；第 4 层是当前阻塞。Whisper 走的是 `ascend_torch_native`（SDPA），不是 ATB paged attention，所以当前失败原因不是 ATB；修完第 4 层后大概率也不会走到 ATB，除非切换到 `ascend` paged attention backend。

因此，ATB 不是所有模型共同的失败原因；不能用 ATB 概括所有模型。

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 参数路由（值传到 ServerArgs） | ✓ `test_cli.py::test_torch_compile_dotted_flags_reach_resolved_sglang_args` | ✓ |
| 取值校验 | ✓ | ✓ |
| NPU 上模型级关闭（显式 true + 未设置） | ✓ `test_sglang_ar_budget.py::test_torch_compile_is_disabled_on_npu` | ✓（配置层） |
| torch.compile 是否被调用 | ✗ UT 不启动 SGLang | ✓ 空跑已验 |
| dynamo 在 NPU 上能否追踪 | ✗ UT 不跑 dynamo | ✓ 已定位到 ATB 前的 9 层 |
| ATB 是否支持该图 | ✗ | ✓ `PagedAttentionOperation setup failed` |

UT 只验参数路由和模型级开关。torch.compile 的实际行为、各层 dynamo 失败和 ATB 阻塞都需要端到端测试；目前没有“真实模型在 NPU 上完成 torch.compile”的正例。
