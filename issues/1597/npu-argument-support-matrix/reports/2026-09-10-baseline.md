# NPU 参数支持矩阵：初始验证记录

- 记录日期：2026-09-10
- 来源：[Issue #1597](https://github.com/sgl-project/sglang-omni/issues/1597)
- 本文性质：上游矩阵快照与已有证据整理，非本仓库新执行的测试报告。
- 独立实测状态：待验证；本次仅创建文档仓库，未启动 NPU 测试。

## 上游矩阵快照

以下保留 Issue 原始列、默认值及标记。`[√]`、`[x]` 和空白均为来源标记，不额外推断 UT 或端到端测试结果。参数默认值可能随代码版本改变，应以测试 commit 为准。

| Args                           | Default value | Options                                           | UTs | supported |
| ------------------------------ | ------------- | ------------------------------------------------- | --- | --------- |
| model_path                                  | None          | str                                               |     | [√]        |
| config                                           | None          | str                                               |     | [√]        |
| text_only                                       | FALSE         | bool                                              |     | [√]        |
| colocate                                        | FALSE         | bool                                              |     | [x]       |
| isolate_stage                                 | None          | list[str]                                         |     |           |
| stage_process                               | None          | list[str]                                         |     |           |
| host                                              | 0.0.0.0         | str                                               |     | [√]        |
| port                                              | 8000           | int                                               |     | [√]        |
| model_name                                | None          | str                                               |     | [√] |
| allowed_local_media_path           | None          | str                                               |     | [√] |
| allowed_media_domain               | None          | list[str]                                         |     |           |
| tts_batch_max_items                    | 32               | int                                               |     |           |
| mem_fraction_static                     | None          | float                                             |     |           |
| thinker_mem_fraction_static        | None          | float                                             |     |           |
| talker_mem_fraction_static          | None          | float                                             |     |           |
| encoder_mem_reserve                | None          | float                                             |     |           |
| cpu_offload_gb                           | None          | int                                               |     |           |
| quantization                                | None          | str                                               |     | [x]       |
| log_level                                      | "info"   | ["debug", "info", "warning", "error", "critical"] |     | [√]        |
| thinker_tp_size                            | None          | int                                               |     | [√]        |
| thinker_gpus                               | None          | str                                               |     | [√]        |
| image_encoder_tp_size               | None          | int                                               |     |           |
| image_encoder_gpus                  | None          | str                                               |     |           |
| talker_gpu                                   | None          | int                                               |     | [√]        |
| code2wav_gpu                            | None          | int                                               |     | [√]        |
| thinker_cuda_graph                    | "default"     | default\|on\|off                                  |     | [√]        |
| talker_cuda_graph                      | "default"     | default\|on\|off                                  |     | [x]       |
| talker_partial_start                      | "default"     | default\|on\|off                                  |     |           |
| thinker_torch_compile               | "default"     | default\|on\|off                                  |     |           |
| talker_torch_compile                  | "default"     | default\|on\|off                                  |     |           |
| thinker_torch_compile_max_bs   | None          | int                                               |     |           |
| talker_torch_compile_max_bs     | None          | int                                               |     |           |
| torch_compile                             | "default"     | default\|on\|off                                  |     |           |
| torch_compile_max_bs                | None          | int                                               |     |           |
| enable_realtime                           | FALSE         | bool                                              |     |           |
| decode_mode                              | None          | str                                               |     |           |
| async_lookahead_min_batch_size | None          | int                                               |     |           |
| thinker_max_running_requests     | None          | int                                               |     |           |
| prefill_coalesce_requests              | None          | int                                               |     |           |
| prefill_coalesce_wait_ms               | None          | float                                             |     |           |
| max_running_requests                  | None          | int                                               |     |           |
| max_queued_requests                  | None          | int                                               |     |           |
| max_total_tokens                         | None          | int                                               |     |           |
| cuda_graph_max_bs                    | None          | int                                               |     | [√]        |

## Issue 中已有的验证反馈

[2026-09-07 的 allowed_local_media_path 反馈](https://github.com/sgl-project/sglang-omni/issues/1597#issuecomment-5564446769)报告：参数路径校验逻辑未发现问题，但在 NPU 上为 Qwen3-TTS 输入 WAV 时遇到 kernel 编译失败，没有音频输出。

该评论将原因归于 NPU 平台集成，但未给出完整环境、启动命令和错误日志，本文未独立确认根因。此证据不能认定该场景端到端通过，也不能仅凭模型执行失败认定参数本身不受支持。

## 后续验收要求

每份实测报告应提供模型与代码版本、软件硬件环境、参数取值、可复现命令、预期行为、实际结果及证据。分别记录参数解析、配置传递、实际功能及端到端结果。失败时标明是参数本身、模型算子、资源约束还是尚未定位。
