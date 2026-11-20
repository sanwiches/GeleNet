# V2/V3 checkpoint evaluation — 2026-11-20

## Scope and provenance

User-supplied server summary: v2_v3_checkpoint_summary_a8vzfsb9/summary.txt and summary.json. Four runs, nine saved checkpoints each (4,9,14,19,24,29,34,39,44). Both scribble training variants use the same EORSSD dense test set of 600 images; these are two annotation protocols, not independent test datasets. Existing runs end at epoch 44, not epoch 50.

Prediction: 352 input, resize logits to native size (bilinear, align_corners=False), sigmoid, per-image min/max, grayscale PNG. Scoring: pysodmetrics 1.6.2, GT >127 foreground. V3 selects level20, with explicit final-available-level fallback for test image0754 (level10,12 regions). This records the chosen project protocol; exact equivalence to every published implementation has not been established.

## Interpretation

- EORSSD-S: V3 is worse on all five final-checkpoint metrics and all five metrics at each run's best-by-S checkpoint. Across checkpoints24–44, V3 has lower standard deviations for all five metrics, but mean performance is mixed. Lower dispersion does not establish better accuracy.
- S-EOR: V3 improves all five final metrics and all five last-five means. Last-five standard deviations are higher for all five metrics; this interval includes V3's improvement from epoch24 to29. It therefore does not support a blanket claim of greater stability.
- At each S-EOR run's best-by-S checkpoint, V3 improves S,wF,MAE but has lower maxF,maxE than V2. Keep all metrics from the same checkpoint. Best-by-S selection on test data is exploratory, not validation-based checkpoint selection.
- No multi-seed repeats or significance analysis were performed. Old V2/EORSSD-S seed remains unverified. Temporal checkpoint dispersion is not seed robustness. No general improvement on both annotation protocols has been demonstrated.

Recommended next step: preserve these as pilot results; compare aligned qualitative failure cases and verify matching initialization, seed, data and training settings before a controlled repeat. Do not select NAMLab settings by these test scores.

## Supplied numerical summary

```text
===== 全部 checkpoint =====
group          epoch       S      wF    maxF    maxE     MAE
v2_eorrsd          4  0.5073  0.1598  0.3678  0.6505  0.1630
v2_eorrsd          9  0.5307  0.2002  0.3304  0.5814  0.1136
v2_eorrsd         14  0.5420  0.2118  0.3413  0.5926  0.1082
v2_eorrsd         19  0.5574  0.2437  0.3505  0.5966  0.0834
v2_eorrsd         24  0.5533  0.2149  0.2652  0.5525  0.0768
v2_eorrsd         29  0.5845  0.2787  0.3798  0.6259  0.0663
v2_eorrsd         34  0.5784  0.2691  0.3737  0.6186  0.0709
v2_eorrsd         39  0.5800  0.2718  0.3713  0.6149  0.0702
v2_eorrsd         44  0.5783  0.2698  0.3655  0.6084  0.0714
v3_eorrsd          4  0.4947  0.1507  0.3394  0.6027  0.1823
v3_eorrsd          9  0.5422  0.2046  0.3585  0.6106  0.1114
v3_eorrsd         14  0.5313  0.2075  0.3149  0.5609  0.1149
v3_eorrsd         19  0.5430  0.2238  0.3124  0.5529  0.1011
v3_eorrsd         24  0.5720  0.2335  0.3424  0.5969  0.0799
v3_eorrsd         29  0.5747  0.2566  0.3742  0.6230  0.0749
v3_eorrsd         34  0.5782  0.2665  0.3764  0.6209  0.0711
v3_eorrsd         39  0.5760  0.2642  0.3726  0.6162  0.0728
v3_eorrsd         44  0.5713  0.2592  0.3608  0.6047  0.0759
v2_seor            4  0.4750  0.1521  0.2634  0.5400  0.2065
v2_seor            9  0.4982  0.1821  0.2759  0.5242  0.1596
v2_seor           14  0.4974  0.1808  0.2561  0.4961  0.1575
v2_seor           19  0.5378  0.1974  0.3761  0.6489  0.1279
v2_seor           24  0.5304  0.2104  0.3133  0.5600  0.1173
v2_seor           29  0.5346  0.2110  0.2897  0.5397  0.1121
v2_seor           34  0.5292  0.2068  0.2817  0.5287  0.1195
v2_seor           39  0.5276  0.2070  0.2812  0.5264  0.1206
v2_seor           44  0.5296  0.2096  0.2820  0.5255  0.1175
v3_seor            4  0.4686  0.1459  0.2575  0.5294  0.2201
v3_seor            9  0.4840  0.1544  0.2570  0.5396  0.1937
v3_seor           14  0.5030  0.1865  0.2658  0.5054  0.1482
v3_seor           19  0.5245  0.2048  0.2800  0.5259  0.1198
v3_seor           24  0.5336  0.2113  0.2806  0.5220  0.1164
v3_seor           29  0.5638  0.2279  0.3187  0.5925  0.0904
v3_seor           34  0.5577  0.2265  0.3142  0.5708  0.0936
v3_seor           39  0.5547  0.2263  0.3087  0.5630  0.0947
v3_seor           44  0.5563  0.2320  0.3107  0.5608  0.0924

===== 最后一轮：epoch44 =====
group          epoch       S      wF    maxF    maxE     MAE
v2_eorrsd         44  0.5783  0.2698  0.3655  0.6084  0.0714
v3_eorrsd         44  0.5713  0.2592  0.3608  0.6047  0.0759
v2_seor           44  0.5296  0.2096  0.2820  0.5255  0.1175
v3_seor           44  0.5563  0.2320  0.3107  0.5608  0.0924

===== 按测试集 S 选出的探索性最佳轮次 =====
group          epoch       S      wF    maxF    maxE     MAE
v2_eorrsd         29  0.5845  0.2787  0.3798  0.6259  0.0663
v3_eorrsd         34  0.5782  0.2665  0.3764  0.6209  0.0711
v2_seor           19  0.5378  0.1974  0.3761  0.6489  0.1279
v3_seor           29  0.5638  0.2279  0.3187  0.5925  0.0904

===== 后5个 checkpoint：24、29、34、39、44 =====
标准差采用总体标准差；仅反映训练后期波动。

v2_eorrsd
  S     均值=0.5749 标准差=0.0110 范围=[0.5533, 0.5845]
  wF    均值=0.2609 标准差=0.0232 范围=[0.2149, 0.2787]
  maxF  均值=0.3511 标准差=0.0432 范围=[0.2652, 0.3798]
  maxE  均值=0.6041 标准差=0.0264 范围=[0.5525, 0.6259]
  MAE   均值=0.0711 标准差=0.0034 范围=[0.0663, 0.0768]

v3_eorrsd
  S     均值=0.5744 标准差=0.0026 范围=[0.5713, 0.5782]
  wF    均值=0.2560 标准差=0.0118 范围=[0.2335, 0.2665]
  maxF  均值=0.3653 标准差=0.0127 范围=[0.3424, 0.3764]
  maxE  均值=0.6123 标准差=0.0100 范围=[0.5969, 0.6230]
  MAE   均值=0.0749 标准差=0.0030 范围=[0.0711, 0.0799]

v2_seor
  S     均值=0.5303 标准差=0.0024 范围=[0.5276, 0.5346]
  wF    均值=0.2090 标准差=0.0017 范围=[0.2068, 0.2110]
  maxF  均值=0.2896 标准差=0.0123 范围=[0.2812, 0.3133]
  maxE  均值=0.5360 标准差=0.0130 范围=[0.5255, 0.5600]
  MAE   均值=0.1174 标准差=0.0029 范围=[0.1121, 0.1206]

v3_seor
  S     均值=0.5532 标准差=0.0103 范围=[0.5336, 0.5638]
  wF    均值=0.2248 标准差=0.0070 范围=[0.2113, 0.2320]
  maxF  均值=0.3066 标准差=0.0134 范围=[0.2806, 0.3187]
  maxE  均值=0.5618 标准差=0.0228 范围=[0.5220, 0.5925]
  MAE   均值=0.0975 标准差=0.0095 范围=[0.0904, 0.1164]

===== 第44轮 V3−V2：前四项正值更好，MAE负值更好 =====
eorrsd: S=-0.0071, wF=-0.0106, maxF=-0.0047, maxE=-0.0037, MAE=+0.0045
seor: S=+0.0268, wF=+0.0223, maxF=+0.0287, maxE=+0.0353, MAE=-0.0252
```
