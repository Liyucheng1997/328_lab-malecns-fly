# 上游来源与修改

`simulation/` 以源码及资源方式纳入本仓库，上游为：
https://huggingface.co/spaces/Xenova/fruit-fly-simulation

基准提交：`776d115ee5aa934578a87fd6d260d138084f59c1`。

本项目增加数字认知动作、黑板和对话框、速率状态回放与本地 gzip 服务兼容处理。原有许可证保留在 `simulation/LICENSE`、`simulation/licenses/` 和各资源 NOTICE 中。

MaleCNS 数据采用 CC BY 4.0，来源 https://male-cns.janelia.org/download/ ，归属 FlyEM / HHMI Janelia、University of Cambridge、MRC LMB、Google Research。数据来自上游预处理包，哈希见 `simulation/public/data/manifest.json`。

MNIST 在使用时通过 torchvision 下载，不随本仓库分发。
