# TimeGrad-baseline6：HEEW 条件联合源荷场景生成

在 `pytorch-ts` 中新增独立的 `baseline6/`，原始 `pts/`、GluonTS 接口和上游示例均不改动。
模型名称建议写作 **TimeGrad (condition-only adaptation)**，不是原论文历史预测任务的直接复现。
上游版本：`7860c9693d55b5c086867477cc33c89485ed0167`。

## 1. 适配范围与方法身份

目标：给定测试日天气、日历及 PV 年份条件，生成 `[N,S,4,24]` 场景，
通道顺序固定为 Electricity、Heat、Cooling、PV。推理不输入历史/测试日真实能源值。

保留 TimeGrad 核心机制：

- 原始 LSTM（可选 GRU）、维度 embedding、隐状态到扩散条件的线性投影。
- **直接加载未修改**的 `pts/model/time_grad/epsilon_theta.py`：门控残差、空洞卷积、噪声步 embedding、条件上采样。
- **直接复用未修改**的 `pts/modules/gaussian_diffusion.py`：噪声 MSE、前向加噪、DDPM 后验均值/方差。
- 每个小时对 **四维联合向量** 扩散；不是四个单独训练的模型。小时之间由自回归 RNN 联系。

相对原任务的明确改动：

| 原 TimeGrad 预测设置 | 本 baseline6 |
|---|---|
| 历史目标和 context warm-up | 每天重新开始，RNN 零状态；首小时 lag 输入为标准化空间的零（训练均值），不是物理量零 |
| 多个历史 lags | 只保留日内 lag-1；不跨天 |
| 历史 context MeanScaler | 2014–2020 训练集逐通道 Z-score |
| 原日历特征 | 与其他 baseline 一致的天气、日历、PV 年份协变量 |
| GluonTS 数据/训练流程 | 独立 HEEW 数据和固定六组预算训练入口 |
| 全局采样 RNG | 显式独立 torch.Generator；DDPM 数学更新不变 |

训练 teacher forcing 用真实的**前一小时**目标，当前/未来目标不进入该小时的 RNN 条件；
测试只回填此前**生成值**。这是自回归生成模型的训练分解，不是测试历史能源输入。
每条场景路径有自己的生成值及隐状态，不共享其他场景的目标轨迹。

RNN 按小时读取当前协变量并保留过去状态，不额外加双向网络或整日天气编码器；
因此虽然接口接收完整 24h 条件，小时 h 的生成不显式使用 h 之后的天气。
这一因果条件利用方式是本适配的结构限制，论文中应与整日条件模型区分。
未将 TimeGrad 替换为 Transformer、整日 DDPM 或独立边缘模型。

天气文件属于实测/再分析还是预报由输入数据来源决定。代码不会将实测天气转换为预报；
使用实测天气时，应表述为给定天气条件的场景生成，而不是已验证的日前概率预测。

## 2. 安装与数据

Python >=3.10。在仓库根目录运行：

```bash
python -m pip install -r requirements_baseline6.txt
```

此独立入口无需安装 GluonTS、MXNet、einops，也无需 `pip install -e .`。
依赖范围不是跨版本逐位复现保证；已完成本地 CPU 合成测试，真实训练须在你的 GPU 环境执行。

放入与其他 baseline **完全相同**的：

```text
Data/CN03_energy_cleaned.csv
Data/weather_cleaned.csv
```

也可以通过 `--set data.energy_path=/absolute/energy.csv` 和
`--set data.weather_path=/absolute/weather.csv` 指定（sweep/train 均支持）。
数据不包含在此代码提交中。

固定时间切分：2014–2020 训练、2021 验证、2022 测试；只使用对齐后的完整 24 小时日。
目标与天气的统计量仅来自训练完整日：`std = numpy.std(ddof=0) + 1e-6`。
默认 pv10：6 个基本天气 + 4 个 PV 辅助变量，加 8 个周期日历条件；
另有 `(year-2014)/6`。总输入信息与既有基线保持一致。
重复时间戳/非有限输入会报错；不完整日跳过，不插补为真实观测。
训练 checkpoint 保存 CSV SHA256、通道与训练统计，测试会校验。

## 3. 六组调参预算

`baseline6/configs/search_space.yaml` 预先列出：

| 配置 | LSTM hidden cells | 学习率 |
|---|---:|---:|
| c01_h40_lr1e3 | 40 | 0.001 |
| c02_h40_lr1e4 | 40 | 0.0001 |
| c03_h80_lr1e3 | 80 | 0.001 |
| c04_h80_lr1e4 | 80 | 0.0001 |
| c05_h128_lr1e3 | 128 | 0.001 |
| c06_h128_lr1e4 | 128 | 0.0001 |

其余固定：2 层 LSTM，dropout 0.1，conditioning length 100，
8 个 residual layers、8 个 residual channels，dilation cycle 2。
每组最多 300 epochs，batch 32，每 5 epochs 验证，12 次未改善停止；
Adam，gradient clip 1.0，默认不做 EMA（ema_decay=0）。
验证每条件生成 20 条场景，以**四通道等权 val_macro_nCRPS**选 checkpoint 和配置，
不以测试集挑参数。六组相同最大 epoch 不等于相同 FLOPs/GPU 时间。

```bash
python -m baseline6.sweep --seed 42
```

输出：

```text
experiments/baseline6/tuning_budget_6/protocol.yaml
experiments/baseline6/tuning_budget_6/tuning_results.csv
experiments/baseline6/tuning_budget_6/best_config.yaml
experiments/baseline6/tuning_budget_6/best_config.json
```

`--skip-completed` 仅跳过配置和训练数据一致的已完成试验，不是从中断 epoch 恢复。
协议、种子或数据改变需新的输出目录和实验输出根目录；非空训练目录拒绝覆盖。
不要编辑已选出的 best_config 来改变扩散协议后仍声称它是该六组搜索的最佳配置。
未知 `--set` 键会报错，避免拼写错误被忽略。

### 扩散步数与时间成本

默认 `diffusion.num_steps=100`、linear beta 从 0.0001 到 0.1，沿用上游默认。
完整 DDPM 每小时采样 100 步，没有独立 DDIM/跳步开关。
一条 24h 轨迹是 **24×100 次去噪调用**；四通道在每次调用中联合生成。
不同 baseline 不应只按同一个步数推断算力相等，应同时报告采样耗时。

保留原 embedding 表上限，`1 <= num_steps <= 500`；超出显式报错。
更改 T 要重新训练和验证，不能只在测试修改 checkpoint 配置。
不对标准化目标做 [-1,1] 裁剪；仅完整场景反归一化后将 PV 负值截为 0，
与其他 baseline 的评估后处理一致。自回归内部继续使用未截断生成值。

## 4. 最优配置的五种子训练

```bash
for seed in 42 123 777 2024 3407; do
  python -m baseline6.train \
    --config experiments/baseline6/tuning_budget_6/best_config.yaml \
    --run-name timegrad_final \
    --set run.seed=$seed || break
done
```

checkpoint：`experiments/baseline6/timegrad_final_seed42/best.pt` 等。
每个种子根据自身验证集选 epoch；最终五种子重复训练不计入六组超参配置数。
保存 resolved_config、history、环境版本、参数量、训练耗时、数据 manifest。
本任务新建模型需要训练；其他 baseline 已训练模型不受影响。

## 5. 测试与 baseline5 对齐输出

```bash
python -m baseline6.evaluate \
  --checkpoint experiments/baseline6/timegrad_final_seed42/best.pt \
  --outdir evaluation_results/baseline6_seed42
```

默认每条件生成100条；测试 seed 从 checkpoint 继承，采样 RNG = seed+90000，
验证 RNG = seed+50000。输出独立记录 training_seed、seed、sample_seed；
不能通过目录名推断种子。训练、评估、绘图 RNG 分离。
相同种子复现仍需相同设备、软件版本、batch、chunk size。

五种子批量评估：

```bash
for seed in 42 123 777 2024 3407; do
  python -m baseline6.evaluate \
    --checkpoint "experiments/baseline6/timegrad_final_seed${seed}/best.pt" \
    --outdir "evaluation_results/baseline6_seed${seed}" || break
done
```

`--scenarios/--num-scenarios`、`--outdir/--output-dir` 为别名。
`--no-plots` 跳过绘图及 Pearson 比较；`--max-days` 仅供排错并标记 partial_test，
不应进入完整测试主表。输出目录必须为空。

```text
channel_metrics.csv             # 每通道一行
global_metrics.csv              # metric,value
global_metrics_wide.csv         # 额外单行宽表
global_metrics.json
metrics_summary.json
daily_scores.csv
metric_definitions.json
baseline6_scenarios.npz
global_pearson.png               # 真实数据全局相关
pearson/real_global_pearson.png
pearson/generated_global_pearson.png
pearson/pearson_difference.png
random_timeseries_50/            # 最多50个不同测试日
```

数值口径以 baseline5 `17c217b` 为准：

- 点预测为场景均值；每通道 MAE、RMSE、R²。R² 为 1−SSE/SST，不是 Pearson²。
- MAE_Z/RMSE_Z 用训练集 Z-score，不是 min-max。
- CRPS 为等权经验分布，成对项分母 S²；nCRPS = sum(CRPS)/(sum(abs(truth))+1e-8)。
  macro_nCRPS 四通道等权；mean_nCRPS 为合并通道比值，不能混淆。
- CR、IW、IS 是95%中央区间，2.5%/97.5%线性分位数；IS_Z = IS/训练标准差。
  IS95 保留为 IS 别名。另保留 Coverage90/IntervalWidth90。
- Precision_Z/Recall_Z 保持既有最近中心半径近似，不是全 kNN 球并集判定；
  默认 k=5、生成池上限10000、随机子样本 seed+1000+channel。
- ES 在训练标准化96维日轨迹上计算，经验成对项分母 S²。
- VS：p=0.5；跨通道无序配对权重1、同通道0；包含同小时和跨小时；
  **3456对先求和再按天平均，不除以3456**。ES/VS 与 ES_Z/VS_Z 分别同值。
- daily_scores 包含逐日 ES_Z、VS_Z、各通道 CRPS、IS、IS_Z。
- 常量目标的 R² 未定义，JSON null / CSV空值；非有限场景报错，不把 NaN/Inf 改成0。

NPZ 包含物理量 scenarios/targets、归一化 conditions、pv_year、dates、channel_names、
训练 target_mean/std，以及随机种子和元数据。CSV/JSON 都使用同一批场景。

## 6. 已生成场景补算

```bash
python -m baseline6.score_npz \
  --npz evaluation_results/baseline6_seed42/baseline6_scenarios.npz \
  --outdir evaluation_results/baseline6_seed42_rescored
```

不训练、不重新采样、不修改源 NPZ。统计量来自源 NPZ 或匹配 checkpoint manifest，
绝不从测试目标估计。额外 `--checkpoint` 可核对训练统计。
只有旧/自定义归档缺少种子时才需要原指标 JSON 或显式 `--seed`；
补算 seed 仅控制指标子采样和绘图，不改变原生成场景。

## 7. 测试与来源

```bash
python -m pytest -c pytest_baseline6.ini -q
```

仅运行 baseline6 的 tests，不运行仓库原 GluonTS 的 test/。
测试覆盖上游扩散损失和 DDPM 数值一致性、teacher-forcing 因果性、日内生成值回填、
条件梯度、联合通道、自回归采样可复现/随机性隔离、时间切分与归一化、
完整六组微型搜索及再训练、checkpoint→评估→补算、绘图和论文 ES/VS。
合成测试不代表真实数据训练收敛或性能优越。

代码追踪：
- `pts/model/time_grad/epsilon_theta.py`：原样运行。
- `pts/modules/gaussian_diffusion.py`：原样运行；baseline6 子类仅提供显式生成器采样入口。
- `pts/model/time_grad/time_grad_network.py`：RNN/维度embedding/投影/自回归结构参考，原文件未改。
- `baseline6/model.py`：24h cold-start 接口、lag-1 teacher forcing 和自主回填。
- `baseline6/upstream.py`：只加载纯 PyTorch 文件，避免旧 GluonTS API 依赖；不是替换噪声网络。
- 数据与评估工具沿用已有基线，保持 baseline5 指标口径；原仓库 MIT 许可保留。

论文引用：Rasul et al., “Autoregressive Denoising Diffusion Models for Multivariate
Probabilistic Time Series Forecasting”, ICML 2021, pp. 8857–8868.
[原论文页面](https://proceedings.mlr.press/v139/rasul21a.html)。

