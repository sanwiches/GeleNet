# V2 NAMLab 多粒度区域扩展监督实验

## 范围与版本

独立分支V2-region-supervision，直接基于V2提交bb5ededc6a47fa8464045a4409aaa410726c2ce4。模型与KTM保持V2原结构，不加入region_strength、region_ids注意力偏置、RC或自适应尺度权重。正在运行的V3三模式实验不受影响。这里是新损失实验，不是旧固定层V3训练代码，也不宣称已有训练收益。

## 已约定的第一版规则

每张训练图根据实际segs可用层数L，选max(L//4,1)、max(L//2,1)、max(3*L//4,1)，依次视为较粗、中等、较细候选。层号增大通常区域数增加，层序来自当前NAMLab导出顺序，不保证跨图有相同绝对粒度。L60选15/30/45，L10选2/5/7。实际层号和区域统计逐图保存。重复层保留三个损失槽，其权重对相同监督自然相加，不假装是三个不同尺度。

在原始分辨率按区域分别统计前景和背景涂鸦像素。仅有前景涂鸦的区域向未标注像素扩展前景；仅有背景涂鸦的区域扩展背景。前景背景同时存在则忽略该区域扩展；没有涂鸦也忽略，绝不能将未知默认成背景。原始已标注像素从所有区域扩展掩码中排除，始终仅参与原始涂鸦损失；不修改原始标注文件。随后对目标图和掩码采用最近邻同步缩放至352。细粒度也不保证区域纯净，保守规则不能消除跨真实边界的错误扩展。

此版不是按点数置信度采样的随机标签，也不是冲突区域前景比例的软标签，避免将讨论中的不同方案混为已经实现。目标和掩码分别保存，不能将四张0/1图直接混合而丢弃未知掩码。

## 损失与效率

总损失=0.50*原始涂鸦BCE+0.10*较粗区域BCE+0.15*中等区域BCE+0.25*较细区域BCE。每项按该batch自己的有效像素数归一化；空项返回可反传的0，不动态重分配权重。四项共享一次模型前向，并在总损失上做一次反向，不是四次骨干前向。显式保留各项而不融合标签，便于核对含义和诊断；每轮记录原始四项BCE与区域扩展覆盖率。

baseline模式使用标准V2原始涂鸦BCE权重1；region模式使用上述四项。两者总损失数值不能直接比较优劣，统一按相同内部验证涂鸦BCE比较。当扩展监督为空时region模式只有0.5倍原始项，这是固定权重设计的实际行为，不应称为与baseline梯度完全相同。

## 数据划分与训练

tools/prepare_region_supervision.py从现有prepared_seor读取train.json和val.json，保持相同1260训练/140验证图像身份；不重新划分、不读取完整GT。仅为训练图预生成区域扩展目标，验证仍只用原始留出涂鸦Partial BCE，按已标注像素数加权选best_val.pth。它不能充分衡量完整目标边界质量。

baseline和region都使用相同seed42、预训练PVT骨干加新初始化其他参数、Adam lr1e-4、batch8、352、44个实际epoch、epoch30乘0.1、梯度逐元素裁剪0.5、num_workers0，无新增几何增强。Python/NumPy/torch/CUDA显式设种子并设cuDNN deterministic，不保证跨版本逐位复现。DataLoader使用独立相同种子的生成器。新增目标加载无随机操作。

先各运行epochs1/max-steps3做完整模型真实数据检查，再从头运行44轮。短程权重不可用于正式初始化。输出目录必须不存在，状态区分smoke_completed/completed/failed，不提供自动恢复。保存配置、代码/骨干/划分哈希、epoch日志、第4/9/.../44轮和内部验证最佳权重。

已查看过EORSSD600张测试图的旧实验评分、checkpoint选择和可视化；引入内部验证不能消除该暴露。新1260训练结果不能与旧1400训练作严格架构对照。最终测试在配方冻结后，统一比较第44轮和内部验证选模结果，不按测试集再挑checkpoint或层数。当前V3自适应融合的固定10/20/40与本实验按L分位取层不同，不能把两者差异只归因于注入位置。

## 使用

在代码根目录运行，model/pvt_v2_b2.pth为现有预训练权重链接。准备示例：

```
python tools/prepare_region_supervision.py --split-source EXISTING_PREPARED_SEOR --regions TRAIN_MAT_DIRECTORY --output NEW_PREPARED_DIRECTORY
python train_region_supervision.py --prepared PREPARED_DIRECTORY --output NEW_SMOKE_DIRECTORY --mode region --epochs 1 --max-steps 3
python train_region_supervision.py --prepared PREPARED_DIRECTORY --output NEW_RUN_DIRECTORY --mode baseline --epochs 44
python train_region_supervision.py --prepared PREPARED_DIRECTORY --output NEW_RUN_DIRECTORY --mode region --epochs 44
```

数据和权重不包含于仓库。准备需要NumPy/Pillow/SciPy；训练另需已有torch/torchvision/timm环境。预测可复用V2的evaluate_weak_GeleNet.py，损失不需要测试期NAMLab输入。正式评分继续使用既定GT>127及pysodmetrics1.6.2协议，避免换用未固定的新口径。

## 当前验证范围

本地6项CPU测试覆盖分位层选择、冲突与未知忽略、原始标注保护、四项权重与梯度、空项不重分配，以及合成MAT到真实数据加载器的端到端准备。全部12个Python文件通过语法解析。原V2模型文件直接取自GitHub V2。测试没有加载完整PVT模型，尚未在服务器真实数据上准备目标或进行完整GPU更新，因此没有新增训练或评估结果。
