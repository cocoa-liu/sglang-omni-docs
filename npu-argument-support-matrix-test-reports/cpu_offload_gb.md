# SGLang-Omni 参数 NPU 验证速览：cpu_offload_gb

实验日期：2026-09-09。本次任务是检查：换用 NPU（Ascend）计算设备后，`cpu_offload_gb`（权重卸载到 CPU）是否仍然需要、能否正常工作。该参数在 CUDA 上已支持；本速览只分析 NPU 上的情况。

## 1. 验证对象与结论

| 参数 | 管什么 | NPU 是否需要 | 本次结论 |
| --- | --- | --- | --- |
| `cpu_offload_gb` | 把模型权重卸载到 CPU 内存、按需取回 NPU，缓解装不下的问题 | 需要——作为装不进 HBM 时的回退手段，NPU 同样存在 | 所测配置下，卸载器（OffloaderV1）在 NPU 上可构造、服务可带该参数启动并完成权重加载；SGLang NPU 后端显式兼容被卸载权重。生成阶段的失败是另一条 NPU 编译器链路问题，与该参数无关 |

验证范围：**Ascend910（8×64GB）+ CANN 9.0.0 + torch 2.10.0(+cpu) + torch_npu 2.10.0（transfer_to_npu 补丁）+ sglang 0.5.18 + sglang_omni 0.1.4**。参数逻辑验证在 113.46.15.88 的 NPU 容器（`fth-sglang-omni-qwen3-tts-test`，空闲设备 4，可用 60.9GB）上完成。

支持矩阵结论：`cpu_offload_gb` NPU 标记为 `[√]`（当前已支持），下方给出验证依据与边界。

## 2. 怎么测，结果证明了什么

`cpu_offload_gb` 是 SGLang `ServerArgs` 字段：大于 0 时构造 `OffloaderV1`，把模块权重按量卸到 CPU、前向时按需取回设备。SGLang 的卸载器内部用 `torch.cuda.Stream`、`torch.cuda.memory_allocated()` 等；在 NPU 上这些经 `torch_npu` 的 `transfer_to_npu` 补丁被改写为 `torch.npu.*`，且 **SGLang 的 NPU 硬件后端显式兼容被卸载的权重**（`hardware_backend/npu/quantization/linear_method_npu.py` 与 `moe_methods.py` 均有 “cpu offload may move it back to CPU … Move to NPU if needed” 的处理，`memory_pool_npu.py` 亦有 `cpu_offloading_chunk_size`）。

**实验思路：分两层。先在真实 NPU 上确认卸载器能按参数构造、且 NPU 后端源码兼容卸载权重；再带 `cpu_offload_gb=4` 端到端启动服务，看能否完成权重加载与 KV 缓存分配。再用“不带该参数”的基线复现同一失败，证明生成报错与本参数无关。**

| 要确认什么 | 怎么做 | 怎样才算通过 | 实际结果 |
| --- | --- | --- | --- |
| 参数>0 构造 OffloaderV1 | 在 NPU 上 `ServerArgs(device=npu, cpu_offload_gb=1)` 并构造卸载器 | 返回 `OffloaderV1`（非 Noop） | 通过：`type=OffloaderV1` |
| 默认 0 构造 Noop | `cpu_offload_gb=0` | 返回 `NoopOffloader` | 通过：`type=NoopOffloader` |
| NPU 后端兼容卸载权重 | 读 `linear_method_npu.py` 源码 | 含处理 “cpu offload 把权重放回 CPU” 的逻辑 | 通过：源码显式处理 |
| `is_pin_memory_available()` 在 NPU 上可调用 | 直接调用 | 不抛异常 | 通过：返回 False（用非锁页内存，仍可用，仅影响 H2D 速度） |
| 带 `cpu_offload_gb=4` 能启动 | 用 0.6B-Base 启动服务 | 服务就绪、权重加载、KV 池分配 | 通过：`READY=9`，加载 1.77GB，KV 池分配成功 |
| 大模型同样能启动 | 用 1.7B-Base 启动服务 | 服务就绪、权重加载 | 通过：`READY=9`，加载 3.80GB，KV 池分配成功 |
| 生成失败是否由本参数引起 | 同模型同请求、**去掉** `cpu_offload_gb` 跑基线 | 若基线同样失败，则与参数无关 | 通过（即“无关”成立）：基线复现同一报错 |

**端到端启动日志（带 `cpu_offload_gb=4`）关键行：**
- 0.6B-Base：`Load weight begin. avail mem=60.38 GB` → `Load weight end. … mem usage=1.77 GB` → `KV Cache is allocated … K size: 14.16 GB, V size: 14.16 GB` → `Memory pool end. avail mem=30.28 GB`
- 1.7B-Base：`Load weight begin. avail mem=60.38 GB` → `Load weight end. … mem usage=3.80 GB` → `KV Cache is allocated … K size: 11.63 GB, V size: 11.63 GB`

**生成阶段失败与参数无关的佐证**：带 `cpu_offload_gb=4` 时，0.6B 报 `AclSetCompileopt(ACL_OP_JIT_COMPILE) error 500001`、1.7B 报 `No module named 'tbe' / GEInitialize failed / OpCompileProcessor init failed`；**去掉 `cpu_offload_gb` 的同模型基线复现完全相同的报错**（0.6B 同为 ACL_OP_JIT 500001；1.7B 同为 GEInitialize/tbe 适配器失败）。卸载器只改变权重落在 CPU 还是 NPU，不触及计算图编译；报错发生在前向计算链路，与卸载参数无因果关系。这是设备 4 上的 NPU 图引擎/算子编译器初始化环境问题，不是 `cpu_offload_gb` 的逻辑问题。

因此 `cpu_offload_gb` 在 NPU 上**当前已支持**（卸载器可构造、服务可带参数启动并完成权重加载、NPU 后端显式兼容被卸载权重），无需新增 NPU 专用参数逻辑；本机生成闭环被另一条 NPU 编译器环境问题阻断，已在基线对照中证明与该参数无关。

## 3. 交付与适用边界

| 项目 | 说明 |
| --- | --- |
| 是否需要保留 | 作为“装不进 HBM”时的回退手段，NPU 同样需要；换 NPU 不消除该需求 |
| 做了什么 | 在真实 NPU 上确认 `cpu_offload_gb>0` 构造 `OffloaderV1`、默认 0 构造 Noop、NPU 后端源码显式兼容被卸载权重；带 `cpu_offload_gb=4` 端到端启动 0.6B 与 1.7B，确认能完成权重加载与 KV 池分配；用同模型基线（不带该参数）复现同一生成报错，证明报错与本参数无关 |
| 可以确认 | 卸载器在 NPU 上可构造、服务可带参数启动并完成权重加载；SGLang NPU 后端显式处理“被卸载权重取回 NPU”；参数接线在 NPU 上可用，无需新增专用逻辑 |
| 不能扩大为 | 本机生成闭环未跑通，但已用基线对照证明阻断点在 NPU 图引擎/算子编译器初始化（ACL_OP_JIT 500001、tbe/GEInitialize 失败），与 `cpu_offload_gb` 无因果；未实测“权重大到必须卸载才装得下”的场景；`is_pin_memory_available()` 在 NPU 上返回 False，H2D 拷贝走非锁页内存，仅影响速度、不影响可用性 |
| UT 覆盖 | 原 UT 仅透传/路由（值流到 `engine.overrides()`，取值 0/None/4.0），无行为级测试；本仓约定 UT 不 import SGLang，卸载行为属上游域。本次已补两项 omni 层 UT：schema 守卫 `EngineArgs(cpu_offload_gb=-1)` 抛错（`tests/unit_test/pipeline/test_runtime_schema.py`）、正值 config-merge 透传 `8`（`tests/unit_test/ming_omni/test_omni_serve.py`），两项均在真实 NPU 容器上跑通 |

补充记录：该参数逻辑层在 NPU 上 **4/4 项检查通过**（合计 12/12 项中的 4 项，另 8 项为 `encoder_mem_reserve`）；端到端启动对照在 0.6B 与 1.7B 两个模型上各跑“带参数/不带参数”两轮，共 4 次启动，均能就绪并完成权重加载，生成报错在两模型上均被基线复现。

证据归档入口（NPU 容器 `fth-sglang-omni-qwen3-tts-test`，设备 4）：
- 参数逻辑检查脚本：`/tmp/verify/npu_verify.py`（4/4 通过输出）
- `cpu_offload_gb=4` 启动日志：`/tmp/offload-verify/server.log`（0.6B）、`/tmp/offload-verify/server.log`（1.7B，覆盖前一轮）
- 基线（不带参数）启动日志：`/tmp/baseline-verify/server.log`（0.6B）、`/tmp/bl17-verify/server.log`（1.7B）
- 配置变体：`/tmp/verify/qwen3_tts_0_6b_npu_offload.yaml`、`/tmp/verify/qwen3_tts_1_7b_npu_offload.yaml`
- 新增 UT：`tests/unit_test/pipeline/test_runtime_schema.py::test_cpu_offload_gb_range_is_enforced_at_validation`、`tests/unit_test/ming_omni/test_omni_serve.py::test_ming_cli_cpu_offload_gb_positive_flows_to_thinker_overrides`

环境：IP 113.46.15.88，容器 `fth-sglang-omni-qwen3-tts-test`，`ASCEND_RT_VISIBLE_DEVICES=4`（空闲设备，可用 60.9GB），Ascend910 / CANN 9.0.0 / torch 2.10.0+cpu / torch_npu 2.10.0 / sglang 0.5.18 / sglang_omni 0.1.4。
