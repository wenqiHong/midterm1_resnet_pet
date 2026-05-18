# 宠物识别：预训练 ResNet18 微调与注意力机制对比

本项目基于 Oxford‑IIIT Pet 数据集，使用预训练的 ResNet18 进行微调，实现 37 类宠物品种识别。同时实现了随机初始化消融实验和 SE 注意力模块，对比三种模型的性能。

## 数据集准备
代码会自动下载 Oxford‑IIIT Pet Dataset 至 ./data 目录，有时可能会超时，需要手动下载。数据集包含 37 种猫狗品种，共 7390 张图像。

训练集：从官方 trainval 中按 8:2 划分训练/验证（保持类别分布）

测试集：官方 test 集

## 训练模型
运行 finetune.py 将依次执行三个实验：

Baseline：预训练 ResNet18 + 分层学习率（主干 1e-5，分类头 1e-3）

Ablation：随机初始化 ResNet18 + 统一学习率（1e-3）

SE‑Attention：预训练 ResNet18 + SE 注意力模块 + 分层学习率

## 主要超参数（可在 finetune.py 中修改）
BATCH_SIZE	32	批量大小

NUM_EPOCHS	15	每个实验的训练轮数

LR_BACKBONE	1e-5	主干网络学习率（微调）

LR_HEAD	1e-3	分类头学习率（从头训练）

optimizer	Adam	优化器

scheduler	StepLR(step=7, gamma=0.1)	学习率调度器

## 输出结果

每个 epoch 输出验证准确率，并保存验证集上最优的模型

训练结束后输出测试集准确率

使用 wandb 记录训练曲线（需先 wandb login）

Val_Acc（验证准确率）和 loss曲线
