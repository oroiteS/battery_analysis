# -*- coding: utf-8 -*-
"""
训练工具模块

此模块包含从 power_soh/functions.py 提取的核心训练工具函数和类。
用于电池健康状态(SoH)估算的神经网络模型训练。

包含的组件:
- standardize_tensor: 张量标准化函数
- TensorDataset: 自定义PyTorch数据集类
- DataDrivenNN: 数据驱动神经网络模型
- My_loss: 自定义损失函数
- train: 训练函数
"""

import torch
from torch import nn
from torch.utils.data import Dataset


def standardize_tensor(data, mode, mean=0, std=1):
    """
    通过去除均值和缩放到单位方差来标准化张量数据。

    此函数对输入张量执行z-score归一化。

    参数:
        data: 要标准化的输入张量
        mode: 'fit'从数据计算均值和标准差，或'transform'使用提供的值
        mean: 用于标准化的均值（如果mode != 'fit'）
        std: 用于标准化的标准差（如果mode != 'fit'）

    返回:
        data_norm: 标准化张量
        mean: 用于标准化的均值
        std: 用于标准化的标准差
    """
    data_2D = data.contiguous().view((-1, data.shape[-1]))  # 重塑为2D
    if mode == "fit":
        mean = torch.mean(data_2D, dim=0)
        std = torch.std(data_2D, dim=0)
    data_norm_2D = (data_2D - mean) / (std + 1e-8)
    data_norm = data_norm_2D.contiguous().view((-1, data.shape[-2], data.shape[-1]))
    return data_norm, mean, std


def inverse_standardize_tensor(data_norm, mean, std):
    """
    逆标准化，将归一化数据转换回原始尺度。

    参数:
        data_norm: 标准化张量
        mean: 原始标准化使用的均值
        std: 原始标准化使用的标准差

    返回:
        data: 原始尺度的张量
    """
    data_norm_2D = data_norm.contiguous().view((-1, data_norm.shape[-1]))  # 重塑为2D
    data_2D = data_norm_2D * std + mean
    data = data_2D.contiguous().view((-1, data_norm.shape[-2], data_norm.shape[-1]))
    return data


class Sin(nn.Module):
    """
    实现正弦激活函数的自定义PyTorch模块。

    此激活函数可以在神经网络中用作tanh或ReLU等标准激活函数的替代。
    """

    def forward(self, input):
        return torch.sin(input)


class Neural_Net(nn.Module):
    """
    具有可定制架构的基础前馈神经网络。

    此类创建具有指定层数和激活函数的全连接神经网络。
    包括用于正则化的dropout层。

    参数:
        seq_len: 输入序列的长度
        inputs_dim: 输入特征的维度
        outputs_dim: 输出特征的维度
        layers: 指定每个隐藏层中神经元数量的列表
        activation: 要使用的激活函数（'Tanh'或'Sin'）
        dropout_rate: Dropout比例
    """

    def __init__(
        self,
        seq_len,
        inputs_dim,
        outputs_dim,
        layers,
        activation="Tanh",
        dropout_rate=0.2,
    ):
        super(Neural_Net, self).__init__()

        self.seq_len, self.inputs_dim, self.outputs_dim = (
            seq_len,
            inputs_dim,
            outputs_dim,
        )

        self.layers = []

        self.layers.append(nn.Linear(in_features=inputs_dim, out_features=layers[0]))
        nn.init.xavier_normal_(self.layers[-1].weight)

        if activation == "Tanh":
            self.layers.append(nn.Tanh())
        elif activation == "Sin":
            self.layers.append(Sin())
        self.layers.append(nn.Dropout(p=dropout_rate))

        for i in range(len(layers) - 1):
            self.layers.append(
                nn.Linear(in_features=layers[i], out_features=layers[i + 1])
            )
            nn.init.xavier_normal_(self.layers[-1].weight)

            if activation == "Tanh":
                self.layers.append(nn.Tanh())
            elif activation == "Sin":
                self.layers.append(Sin())
            self.layers.append(nn.Dropout(p=dropout_rate))

        if len(layers) > 0:
            self.layers.append(
                nn.Linear(in_features=layers[-1], out_features=outputs_dim)
            )
        else:
            self.layers.append(
                nn.Linear(in_features=inputs_dim, out_features=outputs_dim)
            )
        nn.init.xavier_normal_(self.layers[-1].weight)

        self.NN = nn.Sequential(*self.layers)

    def forward(self, x):
        self.x = x
        self.x.requires_grad_(True)
        self.x_2D = self.x.contiguous().view((-1, self.inputs_dim))
        NN_out_2D = self.NN(self.x_2D)
        self.u_pred = NN_out_2D.contiguous().view((-1, self.seq_len, self.outputs_dim))

        return self.u_pred


class TensorDataset(Dataset):
    """
    用于处理张量对的自定义PyTorch数据集。

    此类将输入和目标张量包装在数据集中，可以与PyTorch DataLoader
    一起使用进行批量训练。

    参数:
        data_tensor: 输入张量
        target_tensor: 目标张量
    """

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class DataDrivenNN(nn.Module):
    """
    用于电池状态估算的数据驱动神经网络。

    这是一个没有物理约束的纯神经网络模型。
    它通过代理神经网络处理输入并返回电池状态的预测。

    参数:
        seq_len: 输入序列的长度
        inputs_dim: 输入特征的维度
        outputs_dim: 输出特征的维度
        layers: 指定每个隐藏层中神经元数量的列表
        scaler_inputs: 输入归一化参数（均值，标准差）
        scaler_targets: 目标归一化参数（均值，标准差）
        dropout_rate: Dropout比例
    """

    def __init__(
        self,
        seq_len,
        inputs_dim,
        outputs_dim,
        layers,
        scaler_inputs,
        scaler_targets,
        dropout_rate=0.2,
    ):
        super(DataDrivenNN, self).__init__()
        self.seq_len, self.inputs_dim, self.outputs_dim = (
            seq_len,
            inputs_dim,
            outputs_dim,
        )
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets

        self.surrogateNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            layers=layers,
            dropout_rate=dropout_rate,
        )

    def forward(self, inputs):
        s = inputs[:, :, 0 : self.inputs_dim - 1]
        t = inputs[:, :, self.inputs_dim - 1 :]
        if torch.is_grad_enabled():
            t.requires_grad_(True)
        else:
            t = t.detach()

        s_norm, _, _ = standardize_tensor(
            s,
            mode="transform",
            mean=self.scaler_inputs[0][0 : self.inputs_dim - 1],
            std=self.scaler_inputs[1][0 : self.inputs_dim - 1],
        )

        t_norm, _, _ = standardize_tensor(
            t,
            mode="transform",
            mean=self.scaler_inputs[0][self.inputs_dim - 1 :],
            std=self.scaler_inputs[1][self.inputs_dim - 1 :],
        )

        U_norm = self.surrogateNN(x=torch.cat((s_norm, t_norm), dim=2))
        U = inverse_standardize_tensor(
            U_norm, mean=self.scaler_targets[0], std=self.scaler_targets[1]
        )

        if torch.is_grad_enabled():
            grad_outputs = torch.ones_like(U)
            U_t = torch.autograd.grad(
                U,
                t,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        else:
            U_t = torch.zeros_like(U)

        F = torch.zeros_like(U)
        F_t = torch.zeros_like(U)

        self.U_t = U_t

        return U, F, F_t


class DeepHPMNN(nn.Module):
    """
    深度混合物理-经验模型神经网络。

    此模型将用于状态预测的代理神经网络与建模基础物理的动态神经网络相结合。

    参数:
        seq_len: 输入序列的长度
        inputs_dim: 输入特征的维度
        outputs_dim: 输出特征的维度
        layers: 指定每个隐藏层中神经元数量的列表
        scaler_inputs: 输入归一化参数（均值，标准差）
        scaler_targets: 目标归一化参数（均值，标准差）
        inputs_dynamical: 动态模型的输入规格
        inputs_dim_dynamical: 动态模型输入的维度
        dropout_rate: Dropout比例
    """

    def __init__(
        self,
        seq_len,
        inputs_dim,
        outputs_dim,
        layers,
        scaler_inputs,
        scaler_targets,
        inputs_dynamical,
        inputs_dim_dynamical,
        dropout_rate=0.2,
    ):
        super(DeepHPMNN, self).__init__()
        self.seq_len, self.inputs_dim, self.outputs_dim = (
            seq_len,
            inputs_dim,
            outputs_dim,
        )
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets

        if len(inputs_dynamical.split(",")) <= 1:
            self.inputs_dynamical = inputs_dynamical
        else:
            self.inputs_dynamical = "torch.cat((" + inputs_dynamical + "), dim=2)"
        self.inputs_dim_dynamical = eval(inputs_dim_dynamical)

        self.surrogateNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            layers=layers,
            dropout_rate=dropout_rate,
        )

        self.dynamicalNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim_dynamical,
            outputs_dim=1,
            layers=layers,
            dropout_rate=dropout_rate,
        )

    def forward(self, inputs):
        s = inputs[:, :, 0 : self.inputs_dim - 1]
        t = inputs[:, :, self.inputs_dim - 1 :]

        if torch.is_grad_enabled():
            s.requires_grad_(True)
        s_norm, _, _ = standardize_tensor(
            s,
            mode="transform",
            mean=self.scaler_inputs[0][0 : self.inputs_dim - 1],
            std=self.scaler_inputs[1][0 : self.inputs_dim - 1],
        )
        if torch.is_grad_enabled():
            s_norm.requires_grad_(True)

        if torch.is_grad_enabled():
            t.requires_grad_(True)
        t_norm, _, _ = standardize_tensor(
            t,
            mode="transform",
            mean=self.scaler_inputs[0][self.inputs_dim - 1 :],
            std=self.scaler_inputs[1][self.inputs_dim - 1 :],
        )
        if torch.is_grad_enabled():
            t_norm.requires_grad_(True)

        U_norm = self.surrogateNN(x=torch.cat((s_norm, t_norm), dim=2))
        U = inverse_standardize_tensor(
            U_norm, mean=self.scaler_targets[0], std=self.scaler_targets[1]
        )

        if torch.is_grad_enabled():
            grad_outputs = torch.ones_like(U)
            U_t = torch.autograd.grad(
                U,
                t,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            G = eval("self.dynamicalNN(x=" + self.inputs_dynamical + ")")

            F = U_t - G
            F_t = torch.autograd.grad(
                F,
                t,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            self.U_t = U_t
        else:
            F = torch.zeros_like(U)
            F_t = torch.zeros_like(U)

        return U, F, F_t


class My_loss(nn.Module):
    """
    用于不同训练模式的自定义损失函数。

    此损失函数支持不同的模式：
    - 'Baseline': 标准数据拟合损失
    - 'Sum': 包含物理约束
    - 'AdpBal': 不同损失项的自适应平衡

    参数:
        mode: 训练模式（'Baseline'、'Sum'或'AdpBal'）
        weights: 损失权重 (loss_U, loss_F, loss_F_t)
    """

    def __init__(self, mode, weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.mode = mode
        self.weights = weights

    def forward(
        self,
        outputs_U,
        targets_U,
        outputs_F,
        outputs_F_t,
        log_sigma_u,
        log_sigma_f,
        log_sigma_f_t,
    ):
        loss_U = torch.sum((outputs_U - targets_U) ** 2)
        loss_F = torch.sum(outputs_F**2)
        loss_F_t = torch.sum((outputs_F_t) ** 2)

        if self.mode == "Baseline":
            loss = self.weights[0] * loss_U
        elif self.mode == "Sum":
            loss = (
                self.weights[0] * loss_U
                + self.weights[1] * loss_F
                + self.weights[2] * loss_F_t
            )
        elif self.mode == "AdpBal":
            loss = (
                torch.exp(-log_sigma_u) * loss_U
                + torch.exp(-log_sigma_f) * loss_F
                + torch.exp(-log_sigma_f_t) * loss_F_t
                + log_sigma_u
                + log_sigma_f
                + log_sigma_f_t
            )
        else:
            loss = self.weights[0] * loss_U

        self.loss_U = loss_U
        self.loss_F = loss_F
        self.loss_F_t = loss_F_t
        return loss


def train(
    num_epoch,
    batch_size,
    train_loader,
    num_slices_train,
    inputs_val,
    targets_val,
    model,
    optimizer,
    scheduler,
    criterion,
    log_sigma_u,
    log_sigma_f,
    log_sigma_f_t,
):
    """
    神经网络模型的训练函数。

    此函数执行训练循环，包括前向传播、损失计算、反向传播和验证评估。

    参数:
        num_epoch: 训练轮数
        batch_size: 训练批次大小
        train_loader: 训练数据的DataLoader
        num_slices_train: 训练数据切片数量
        inputs_val: 验证输入
        targets_val: 验证目标
        model: 要训练的神经网络模型
        optimizer: 优化算法
        scheduler: 学习率调度器
        criterion: 损失函数
        log_sigma_u: U的不确定性参数的对数
        log_sigma_f: F的不确定性参数的对数
        log_sigma_f_t: F_t的不确定性参数的对数

    返回:
        model: 训练后的模型
        results_epoch: 每个epoch的训练结果
    """
    num_period = int(num_slices_train / batch_size)
    results_epoch = dict()
    results_epoch["loss_train"] = torch.zeros(num_epoch)
    results_epoch["loss_val"] = torch.zeros(num_epoch)
    results_epoch["p_r"] = torch.zeros(num_epoch)
    results_epoch["p_K"] = torch.zeros(num_epoch)
    results_epoch["p_C"] = torch.zeros(num_epoch)
    results_epoch["var_U"] = torch.zeros(num_epoch)
    results_epoch["var_F"] = torch.zeros(num_epoch)
    results_epoch["var_F_t"] = torch.zeros(num_epoch)

    for epoch in range(num_epoch):
        model.train()
        results_period = dict()
        results_period["loss_train"] = torch.zeros(num_period)
        results_period["p_r"] = torch.zeros(num_period)
        results_period["p_K"] = torch.zeros(num_period)
        results_period["p_C"] = torch.zeros(num_period)
        results_period["var_U"] = torch.zeros(num_period)
        results_period["var_F"] = torch.zeros(num_period)
        results_period["var_F_t"] = torch.zeros(num_period)
        with torch.backends.cudnn.flags(enabled=False):
            for period, (inputs_train_batch, targets_train_batch) in enumerate(
                train_loader
            ):
                U_pred_train, F_pred_train, F_t_pred_train = model(
                    inputs=inputs_train_batch
                )
                loss = criterion(
                    outputs_U=U_pred_train,
                    targets_U=targets_train_batch,
                    outputs_F=F_pred_train,
                    outputs_F_t=F_t_pred_train,
                    log_sigma_u=log_sigma_u,
                    log_sigma_f=log_sigma_f,
                    log_sigma_f_t=log_sigma_f_t,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                results_period["loss_train"][period] = criterion.loss_U.detach()
                try:
                    results_period["p_r"][period] = model.p_r.detach()
                    results_period["p_K"][period] = model.p_K.detach()
                    results_period["p_C"][period] = model.p_C.detach()
                except AttributeError:
                    pass
                results_period["var_U"][period] = torch.exp(-log_sigma_u).detach()
                results_period["var_F"][period] = torch.exp(-log_sigma_f).detach()
                results_period["var_F_t"][period] = torch.exp(-log_sigma_f_t).detach()

                # 详细的训练日志已通过回调函数记录，无需在此打印

        results_epoch["loss_train"][epoch] = torch.mean(results_period["loss_train"])
        results_epoch["p_r"][epoch] = torch.mean(results_period["p_r"])
        results_epoch["p_K"][epoch] = torch.mean(results_period["p_K"])
        results_epoch["p_C"][epoch] = torch.mean(results_period["p_C"])
        results_epoch["var_U"][epoch] = torch.mean(results_period["var_U"])
        results_epoch["var_F"][epoch] = torch.mean(results_period["var_F"])
        results_epoch["var_F_t"][epoch] = torch.mean(results_period["var_F_t"])

        model.eval()
        U_pred_val, F_pred_val, F_t_pred_val = model(inputs=inputs_val)
        criterion(
            outputs_U=U_pred_val,
            targets_U=targets_val,
            outputs_F=F_pred_val,
            outputs_F_t=F_t_pred_val,
            log_sigma_u=log_sigma_u,
            log_sigma_f=log_sigma_f,
            log_sigma_f_t=log_sigma_f_t,
        )
        scheduler.step()
        results_epoch["loss_val"][epoch] = criterion.loss_U.detach()

    return model, results_epoch
