# MaleCNS 果蝇模拟与数字识别 · v1.0.0

Windows 本地运行的果蝇连接组交互实验，包含运动模拟、黑板手写数字识别、头顶回答气泡和读出网络训练。仓库附带连接组压缩数据与预训练模型。

## 快速开始

先安装 Python 3.11、Node.js 22.12+ 和 Git，在 PowerShell 中执行：

```powershell
git clone https://github.com/Liyucheng1997/malecns-fly-digit-lab.git
cd malecns-fly-digit-lab
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\start.cmd
```

打开 http://127.0.0.1:5173/ 。首次安装需要联网下载依赖；训练、教学与测试集示例首次使用时会下载 MNIST。已附带预训练权重，无需先训练。建议使用 32 GB 内存的电脑；当前验证使用 CPU。

后续双击 `start.cmd` 即可启动。版本变更见 [CHANGELOG.md](CHANGELOG.md)。

## 在果蝇模拟器里识别与学习

打开 http://127.0.0.1:5173/ ，启动后在 **Actions → 识别数字** 进入认知动作：

识别场景中，果蝇面前有一块带木框和粉笔托的黑板，直接在黑板上书写，点击 **让果蝇看看**。推理完成后，果蝇头顶的对话框显示“这是 X！”和分类分数；对话框按头部屏幕投影定位。重新书写会清除上一轮答案。黑板直接提供模型输入，这个展示并不额外模拟眼睛或相机的视觉成像。

- 在同一页面的画板手写数字，点击 **让果蝇识别**。果蝇旁显示判断，左侧解剖脑图展示实际计算状态的四帧慢放（输入态 + 三步连接传播），可点击 **重放脑内传播**。
- 选择 **正确数字**，点击 **教它这个数字**，对当前笔迹进行监督训练并立即重新识别。更新的权重写入 `artifacts/personal-readout.pt`，下次请求自动加载。保存最近 256 条教学特征，配合原训练划分中的 128 张 MNIST 回放以减少遗忘。个人模型未重新测评，不能沿用 96.16% 为个人模型的准确率。
- 点击 **用 MNIST 训练**，在界面内启动完整的 10,000 张 / 25 轮训练，查看进度，完成后自动使用新模型。旧模型和报告保存在 `artifacts/history/`，过程日志为 `artifacts/train-ui.log`。此操作重建初始模型，会优先使用新训练的模型；历史个人模型保留。
- 点击 **返回运动** 或 Walk / Turn / Fly 恢复运动模式。识别期间运动推进暂停；脑图显示速率状态，不把它伪装成 LIF 脉冲。个人教学拒绝使用 UI 测试集示例，避免测试泄漏。

脑图抽样显示每 16 个节点中的一个，共 10,419 个节点；每帧携带神经元索引和 body ID，前端核对数据身份后才按真实 soma 坐标着色。无 soma 的节点仍参与计算，但不能在解剖图上显示。

认知动作调用的是同一真实连接图上的独立速率模型，而非运动 LIF 模型的同一运行状态。连接图固定，训练更新读出网络。并未实现全部生物突触可塑性或果蝇自主选择任务；动作由用户触发。

- 数字实验室：http://127.0.0.1:8000/ （手写数字、测试集示例、分类分数、神经活动、训练曲线）
- 3D 果蝇模拟：http://127.0.0.1:5173/ （首次点击 **Download & start**，然后点 **Walk / Turn / Fly** 或涂抹脑区）

启动脚本仅在端口空闲时启动进程；若端口被其他程序占用，请先检查占用程序。服务只绑定本机。模拟器后台标签会暂停计算；切回页面继续。执行 `stop.ps1` 停止本项目启动脚本创建的隐藏服务；它会核实进程路径和命令。

## 已完成的训练

固定随机种子 328，MNIST 官方训练集随机抽取 10,000 张训练和互不重叠的 2,000 张验证，官方测试集全部 10,000 张仅用于最终评价。25 个 epoch，根据验证准确率选择模型。CPU 运行约 101 秒。

| 模型 | 测试准确率 |
|---|---:|
| MaleCNS 固定速率网络 + 可训练 MLP 读出层 | 96.16% |
| 同一模型关闭连接传播，不重训 | 95.49% |
| 原始像素 MLP 基线 | 95.59% |

完整结果、混淆矩阵、训练曲线与源码哈希：`artifacts/metrics.json`。权重：`artifacts/readout.pt`。本地训练日志：`artifacts/train.log`（不随仓库分发）。训练 / 验证索引：`artifacts/split_indices.npz`。

这是单种子工程实验。像素基线与连接组读出使用相同隐藏层宽度，输入维度和参数量不同；结果不能证明果蝇连接组优于普通神经网络。关闭传播属于不重训干预，也不是严格的结构随机化对照。手写画板上的个人字迹与 MNIST 存在分布差异，准确率不能直接套用。

## 真实数据如何进入网络

使用上游打包的 MaleCNS v1.0 分类神经元诱导子图：166,700 个节点、25,582,938 条有向边，边权合计代表 124,177,617 个突触。不是整个官方数据表的全部分割片段。所有节点与边均保留，无额外阈值或抽取小子图。

`digitlab/model.py` 读取上游压缩 CSR 文件并逐块检查 SHA256、节点 / 边 / 突触总数。矩阵行是突触后神经元，列是突触前神经元。边符号取自上游元数据：ACh 为正，GABA / glutamate 为负，其余快电流为零。零符号连接仍保留在稀疏图中，但不传递快电流；不能称为所有突触均有非零功能。权重按每个目标神经元的总输入突触数归一化。

数字识别是单独实现的固定速率储备池：

1. 28×28 图像通过固定随机稀疏编码映射到全部节点，每节点采样 4 个像素；这是工程编码，不是重建的果蝇视网膜。
2. 初态 `h0 = tanh(drive)`；三次更新 `h = tanh(0.7 * drive + 1.5 * W @ h)`。每张图重置状态；无跨样本记忆。
3. 固定抽取 4,096 个神经元作为读出特征，标准化参数仅在训练集拟合。
4. 训练 `4096 → 256 → 10` 的 MLP，AdamW、交叉熵；真实图和编码器固定。

因此训练的是连接组网络之后的人工读出层，不是让果蝇本体突触学习数字。速率状态可以为负，不应解释为实际放电频率。

## 与 3D 模拟的关系

`simulation/` 是 Xenova 的 Neural Canvas，使用 MaleCNS 数据和改编的 LIF 脉冲神经模型。它的运动为神经活动驱动的设计动画，不是经过验证的动物行为、肌肉或空气动力学模型；用于展示的 NeuroMechFly 身体来自雌性标本。

数字实验已接入模拟器的动作菜单、模式切换与解剖脑图。它和 LIF 演示共享同一套真实连接组数据，但动力学不同，识别结果不直接驱动肌肉或改变走路策略。数字面板显示的是推理产生的实际速率状态，不是预录脉冲。

本地兼容修复：Vite 8 静态服务器会为 `.gz` 自动添加解压行为；`simulation/vite.config.js` 添加原始 gzip 字节中间件，以符合上游手动解压和校验逻辑。`start.ps1` 使用 Python 静态服务器提供构建版本，也不会添加 `Content-Encoding: gzip`。

## 重新安装、训练与检查

需要 Python 3.11、Node.js 22.12+、Git。当前安装为 CPU PyTorch，不依赖 CUDA。浏览器模拟器可自行选择 WebGPU，环境不支持时回退 JavaScript；本次实际验证使用 JavaScript 后端。

```powershell
.\setup.ps1
.\train.ps1
.\start.ps1
.\.venv\Scripts\python -m digitlab.verify
```

扩大训练为 55,000 张（另留 2,000 张验证；剩余不使用）：

```powershell
.\train.ps1 -TrainSize 55000 -Epochs 30
```

重训会替换 `artifacts/readout.pt` 和报告；先备份可保留旧实验。推理服务会在后续请求中自动载入新权重。缓存特征的键包含参数、模型源码及数据 manifest 哈希。

可选 GPU：安装适配 RTX 50 系列的 CUDA PyTorch 与配套 torchvision，然后用 `-Device cuda`；当前已交付训练不需要它。

## 来源与许可

- 官方数据：https://male-cns.janelia.org/download/ ，FlyEM / HHMI Janelia、University of Cambridge、MRC LMB、Google Research，CC BY 4.0。
- 上游模拟器：https://huggingface.co/spaces/Xenova/fruit-fly-simulation ，本地版本 `776d115ee5aa934578a87fd6d260d138084f59c1`。
- LIF 参考：https://www.nature.com/articles/s41586-024-07763-9 。原研究验证不自动覆盖 MaleCNS 移植或本数字实验。
- MNIST 由 torchvision 下载，原始数据与校验信息保存在 `data/MNIST/raw`；https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html 。
- 上游程序与身体 / 字体等资产许可详见 `simulation/LICENSE`、`simulation/licenses/`、`simulation/public/body/assets/NOTICE`。
