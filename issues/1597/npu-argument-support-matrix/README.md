# NPU Argument Support Matrix 测试验证

关联：[sgl-project/sglang-omni#1597](https://github.com/sgl-project/sglang-omni/issues/1597) — [Roadmap][NPU] Ascend NPU support Roadmap

本目录用于汇总该 Issue 参数支持矩阵的测试验证报告。

## 当前资料

- [2026-09-10 矩阵快照与验证状态](reports/2026-09-10-baseline.md)
- [测试报告模板](report-template.md)

当前尚未在此仓库录入独立实机验证结果。矩阵中的支持标记来自上游 Issue，不等于所有模型、所有参数组合均已通过验证。

## 提交报告

复制模板到 `reports/YYYY-MM-DD-模型-参数.md`，填写环境、命令、预期与实际结果，以及可公开的日志或附件。参数支持结论限定到实际验证过的模型与配置。
