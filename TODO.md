# Lamina ToDo

本文件按优先级分组：P0（紧急/正确性） -> P1（重要） -> P2（常规改进） -> P3（低优先） -> P4（长期/可选）。

每项使用任务复选框标注当前状态：- [ ] 未开始  - [~] 进行中  - [x] 已完成

> 说明：请尽量把需要代码改动的项拆成小任务并在 PR/Issue 中追踪。

---

## P0 — 紧急 / 正确性
- [x] impl-extract-variable — 合并同类项（例如 3*x + 2*x -> 5*x）
- [x] impl-multiply-identity — 处理乘以 0/1/-1 的快速路径并避免冗余循环
- [x] impl-power-equality — 安全的底相等判断，避免错误合并指数
- [x] fix-undefined-forms — 保护未定式（例如 0^0、inf*0）
- [~] tests-symbolic-simplify — 为 simplify_add/multiply/power/sqrt 增加单元测试（进行中）
 - [ ] #126 Fix composite-root multiplication with constants — 复合根式中带常数因子的乘法结果不正确（可能导致崩溃） (Issue #126)

## P1 — 重要（表达能力提升）
- [x] impl-exponent-merge — 安全的幂合并：同底或可约分的有理指数合并（保守实现已添加）
- [x] flatten-multiply-improve — 乘法扁平化并提前合并数值系数，减少中间树构造
- [x] bigint-sqrt-optimization — BigInt 下的轻量级平方因子提取（小素数优先策略）
 - [ ] #117 Support real cube root of negative integers (e.g. -8) — 处理负数的三次根（Issue #117)

## P2 — 常规改进 / 工具
- [~] debug-logging-switch — 用编译时宏与运行时环境变量控制符号化简调试输出（已完成，运行时 env 支持）
- [ ] hash-and-eq-improvements — 改进 `HashData` 的规范化与相等/哈希策略以支持线性合并
- [ ] ci-and-build-checks — 在 CI 中添加轻量构建/测试检查，保证变更不会回退
 - [ ] #35 Add installation instructions to docs — 在文档中补充 Lamina 安装/构建指南 (Issue #35)

## P3 — 次要/性能优化
- [ ] 更好的 BigInt 因式/平方提取算法（如 Pollard、改进试除策略）
- [ ] 将两两 O(n^2) 合并替换为基于哈希/多重集合的一次合并

## P4 — 长期/愿望清单
- [ ] 引入更完整的数学化简规则（例如三角恒等式、对数幂合并等）
- [ ] 更成熟的自动演绎/化简策略（可配置的化简级别、成本模型）

---

贡献者说明：请在提交 PR 前把对应 TODO 的子任务列在 PR 描述中，关联 Issue（若有）并在此处打勾或移动到正确优先级段。

```
