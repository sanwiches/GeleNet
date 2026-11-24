# 区域监督三组实验六份评估结果

记录日期2026年11月24日。来源为用户提供的终端输出，evaluation_runs/v2_region_threegroups_YqmWlAII状态COMPLETED，六份metrics.json均已生成，每份600张。下表使用该次终端汇总四位小数，尚未导入此目录完整precision summary.json、曲线或pair_hashes.json。本记录新增，不覆盖另一次四份评估的REGION_SUPERVISION_RESULTS_20261124.md。

## 实验范围

S-EOR涂鸦1260训练/140内部验证，seed42，44个实际epoch，V2原模型，无KTM区域偏置。已收到服务器核心模型、骨干权重哈希和训练/验证样本顺序核对一致的输出。固定区域监督权重原始/粗/中/细为0.50/0.10/0.15/0.25；渐进方案前10轮0.90/0.02/0.03/0.05，11至20轮线性过渡到最终权重，之后保持。区域前背景冲突、未知及原始涂鸦像素不参与扩展损失，不使用随机区域标签。

训练目录相对于/home/data2/XintongHu/GeleNet：
- baseline：experiments/v2_regionloss_seor_train_WXGCeXqh/baseline
- constant：experiments/v2_regionloss_seor_train_WXGCeXqh/region
- staged：experiments/v2_regionstaged_only_QlAqIVUi/region

三组均报告完整epoch1至44与completed状态。best_val按内部验证涂鸦BCE选取，分别14/18/23轮，没有根据本次测试分数重新挑选。预测352输入，logits恢复原尺寸后sigmoid、逐图minmax、保存PNG；GT>127，pysodmetrics1.6.2。测试时不需要NAMLab输入。

## 六份评分

| 组别 | 口径 | epoch | S ↑ | wF ↑ | maxF ↑ | maxE ↑ | MAE ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | final44 | 44 | 0.5236 | 0.2080 | 0.2771 | 0.5178 | 0.1264 |
| baseline | best_val | 14 | 0.5178 | 0.1994 | 0.2873 | 0.5294 | 0.1365 |
| constant | final44 | 44 | 0.7931 | 0.5501 | 0.7685 | 0.9352 | 0.0221 |
| constant | best_val | 18 | 0.7532 | 0.4660 | 0.7557 | 0.9294 | 0.0276 |
| staged | final44 | 44 | 0.7885 | 0.5393 | 0.7726 | 0.9334 | 0.0231 |
| staged | best_val | 23 | 0.6944 | 0.3504 | 0.7161 | 0.9190 | 0.0459 |

staged/best_val打印的原始精度：S0.6943685775822278，wF0.3504211258029571，maxF0.7161491598325131，maxE0.9190102994023323，MAE0.04585816200279911。diagnostic_only的MAE不作为正式评分。

## 当前解释

1. 本次单次运行中，两种区域扩展监督在final44以及各自best_val两种口径下，相对对应基线五项均改善。final44固定方案S约增加0.2695、MAE相对降低约82.5%，由四位小数近似计算。这是完整训练配方的改善，不将其唯一归因于某一个组件，也不是多种子统计显著性结论。
2. 渐进方案未整体优于固定方案。final44的S、wF、maxE、MAE固定方案更好，渐进仅maxF高0.0041；best_val口径固定方案五项均更好。因此不把渐进日程记为已证实的进一步改进。
3. 对两种区域监督，验证涂鸦BCE选择的checkpoint五项测试指标都不如各自final44。这提示当前稀疏BCE与完整掩码质量的选模排序可能不一致。保留两套预先确定的口径，不据本次测试结果回头修改验证规则，不拼接不同轮次最优指标。
4. 分数还不能证明边界精细度提升，需同图对照、前景漏检/背景误检分解及边界诊断。staged/best_val的near_empty_predictions157指阈值0.5下前景占比小于1%的图像数，不等于157张完全漏检。

## 限制与下一步

此前已查看同一EORSSD600张测试图的旧结果和可视化，内部验证不能消除该暴露。新1260训练不能与旧1400全量训练作严格架构对照。未进行多随机种子重复、独立额外测试、完整训练稳定性或边界指标验证。

下一步先冻结当前配置，使用已保存预测制作同图原图/GT/baseline/constant/staged对照，检查极端改善/退化与固定随机样本；同时审查训练图的区域扩展目标。完整精度JSON、预测GT哈希、曲线及训练记录应继续归档。暂不继续按这些测试分数调节日程、层数或权重。
