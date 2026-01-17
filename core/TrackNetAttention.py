"""
TrackNetV3 的一个带注意力机制（CBAM）的实现版本。

整体结构可以理解为 U-Net 风格的编码器-解码器：
- 编码端：多层卷积 + 下采样（MaxPool），逐步扩大感受野提取语义特征
- 解码端：上采样 + skip connection（concat 编码端特征），恢复空间分辨率
- 输出端：1x1 卷积映射到 3 通道，再用 sigmoid 得到 [0,1] 的概率/热力图

注意力机制（CBAM）插入到各尺度特征上：
- ChannelAttention：学习“哪些通道更重要”
- SpatialAttention：学习“哪些空间位置更重要”
- 这里的实现使用可学习缩放系数 scale，初始为 0，使注意力从“恒等映射”开始训练更稳定
"""

import torch  # PyTorch：张量运算与模型计算图的核心库
from torch import nn  # PyTorch：神经网络层/模块定义（nn.Module、Conv2d 等）


class Conv(nn.Module):  # 组合层：Conv2d + ReLU/Identity + BatchNorm2d
    def __init__(self, ic, oc, k=(3, 3), p="same", act=True):  # ic/oc=通道数，k=卷积核，p=padding，act=是否启用 ReLU
        super().__init__()  # 初始化 nn.Module：注册子模块与参数，支持 state_dict 等机制
        # 卷积：负责空间特征提取；padding='same' 保持特征图尺寸不变（依赖 PyTorch 版本支持）
        self.conv = nn.Conv2d(ic, oc, kernel_size=k, padding=p)  # 卷积层：提取局部空间特征并改变通道维
        # BN：缓解内部协变量偏移，加速收敛并提升稳定性
        self.bn = nn.BatchNorm2d(oc)  # 批归一化：规范化特征分布，提升训练稳定性
        # 可选激活：默认 ReLU；act=False 时用 Identity 表示“不做激活”
        self.act = nn.ReLU() if act else nn.Identity()  # 激活层：默认 ReLU；不启用时保持恒等映射

    def forward(self, x):  # 前向传播：定义输入张量的计算过程
        # 顺序：Conv -> Act -> BN（这里把 BN 放在 act 外层；不同项目可能会采用 Conv->BN->Act）
        return self.bn(self.act(self.conv(x)))  # 依次执行 conv->act->bn，并返回输出特征图


class ChannelAttention(nn.Module):  # CBAM 的通道注意力：为每个通道学习一个重要性系数
    def __init__(self, in_channels, reduction_ratio=16):  # reduction_ratio 控制瓶颈层压缩比例，降低参数量
        super().__init__()  # 初始化 nn.Module 基类
        # 全局平均池化/最大池化：把 HxW 汇聚成 1x1，用于生成通道描述子
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化：把 HxW 聚合成 1x1 的通道均值
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化：把 HxW 聚合成 1x1 的通道最大值
        
        # 共享 MLP（用 1x1 Conv 实现）：先降维再升维，输出通道权重 logits
        self.fc = nn.Sequential(  # 共享“MLP”：用 1x1 卷积实现降维->升维的通道权重映射
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),  # 降维：C -> C/r
            nn.ReLU(),  # 中间激活：提供非线性表达能力
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)  # 升维：C/r -> C
        )  # 输出形状为 [N, C, 1, 1]（每个通道一个 logit）
        # Sigmoid 将 logits 映射到 (0,1)，作为注意力权重
        self.sigmoid = nn.Sigmoid()  # 把通道 logits 映射到 (0,1) 区间作为注意力权重
        
        # 可学习缩放系数：初始为 0 => 注意力分支一开始不改变主干特征，训练更平滑
        self.scale = nn.Parameter(torch.zeros(1))  # 可学习缩放：初始为 0 时注意力等价于恒等（更稳）
        
        # 将 fc 中的卷积权重置零：配合 scale=0，进一步让初期注意力更接近恒等
        self._init_weights()  # 初始化注意力分支权重（此处置零，降低初始扰动）

    def _init_weights(self):  # 初始化通道注意力内部的可学习参数
        for m in self.fc.modules():  # 遍历 Sequential 内部所有子模块
            if isinstance(m, nn.Conv2d):  # 只对 Conv2d 做初始化
                nn.init.zeros_(m.weight)  # 权重置零：使注意力分支初始输出更接近常数

    def forward(self, x):  # 前向：输入特征 x，输出“通道缩放因子”（形状 [N,C,1,1]）
        # 对输入特征分别做 avg/max pooling，再经过共享 MLP
        avg_out = self.fc(self.avg_pool(x))  # 平均池化分支：得到通道统计量并映射到 logits
        max_out = self.fc(self.max_pool(x))  # 最大池化分支：强调强响应区域的通道信息
        # 两种汇聚信息相加：更鲁棒地捕捉通道重要性
        out = avg_out + max_out  # 融合两分支信息（简单相加，常见 CBAM 做法）
        # 通道注意力权重，形状为 [N, C, 1, 1]
        att = self.sigmoid(out)  # Sigmoid 归一化得到通道注意力权重
        # 返回一个“通道缩放因子”：1 + scale*(att-1)
        # - scale=0 时等于 1（不改变）
        # - scale>0 时逐步靠近 att
        return 1.0 + self.scale * (att - 1.0)  # 输出通道缩放因子：恒等基线 + 可学习偏移（训练更稳定）


class SpatialAttention(nn.Module):  # CBAM 的空间注意力：为每个像素位置学习重要性系数
    def __init__(self, kernel_size=7):  # kernel_size 越大，空间注意力感受野越大
        super().__init__()  # 初始化 nn.Module 基类
        # 空间注意力用一个卷积在 (avg_pool, max_pool) 拼接后的 2 通道图上做融合
        padding = kernel_size // 2  # 为保持输出 H,W 不变，padding 取 floor(k/2)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 2 通道输入（avg+max）-> 1 通道空间 logits
        self.sigmoid = nn.Sigmoid()  # Sigmoid 把空间 logits 映射到 (0,1)
        
        # 与通道注意力一致：可学习缩放，使初期接近恒等映射
        self.scale = nn.Parameter(torch.zeros(1))  # 可学习缩放：初始为 0 让空间注意力从恒等开始
        
        self._init_weights()  # 初始化空间注意力卷积权重（置零以降低初期扰动）

    def _init_weights(self):  # 初始化空间注意力参数
        # 初始化为 0，配合 scale=0，使空间注意力初始不改变输入
        nn.init.zeros_(self.conv.weight)  # 卷积权重置零：logits 初始为 0，sigmoid 为 0.5

    def forward(self, x):  # 前向：输入特征 x，输出空间缩放因子（形状 [N,1,H,W]）
        # 按通道维做平均/最大，得到两张 [N,1,H,W] 的图
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道均值图：每个像素位置的平均激活强度
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大图：每个像素位置的最强响应
        # 拼成 [N,2,H,W] 作为空间注意力的输入
        x_cat = torch.cat([avg_out, max_out], dim=1)  # 拼接成 2 通道图，供空间注意力卷积融合
        # 经过卷积融合得到 [N,1,H,W] 的空间 logits
        out = self.conv(x_cat)  # 空间卷积：利用局部上下文计算空间 logits
        # Sigmoid 归一化到 (0,1)
        att = self.sigmoid(out)  # Sigmoid 归一化得到空间注意力权重
        # 返回空间缩放因子：1 + scale*(att-1)
        return 1.0 + self.scale * (att - 1.0)  # 输出空间缩放因子：恒等基线 + 可学习偏移


class CBAM(nn.Module):  # CBAM：通道注意力 + 空间注意力的组合模块
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):  # 初始化 CBAM 子模块参数
        super().__init__()  # 初始化 nn.Module 基类
        # CBAM = Channel Attention + Spatial Attention（顺序通常是先通道后空间）
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)  # 通道注意力子模块
        self.spatial_att = SpatialAttention(kernel_size)  # 空间注意力子模块

    def forward(self, x):  # 前向：对输入特征依次施加通道与空间注意力
        # 逐元素相乘应用通道注意力（广播到 H,W）
        x = x * self.channel_att(x)  # 通道加权：对每个通道乘以权重（广播到 H,W）
        # 再逐元素相乘应用空间注意力（广播到 C）
        x = x * self.spatial_att(x)  # 空间加权：对每个像素位置乘以权重（广播到 C）
        return x  # 返回注意力增强后的特征，供后续卷积继续处理


class TrackNetAttention(nn.Module):  # TrackNet 主干：编码器-解码器结构，带多尺度 CBAM
    def __init__(self):  # 构建网络层并注册为子模块
        super().__init__()  # 初始化 nn.Module 基类

        # 编码器 Stage 1：输入 9 通道（通常是连续 3 帧 RGB 拼接：3*3=9）
        self.conv2d_1 = Conv(9, 64)  # Stage1：输入 9 通道（常见是 3 帧 RGB 拼接），输出 64 通道
        self.conv2d_2 = Conv(64, 64)  # Stage1：进一步提取 64 通道特征
        self.cbam_1 = CBAM(64)  # Stage1：注意力增强（通道+空间）
        self.max_pooling_1 = nn.MaxPool2d((2, 2), stride=(2, 2))  # Stage1：下采样，空间尺寸减半

        # 编码器 Stage 2
        self.conv2d_3 = Conv(64, 128)  # Stage2：通道扩展到 128，进入更高层语义
        self.conv2d_4 = Conv(128, 128)  # Stage2：提取 128 通道特征
        self.cbam_2 = CBAM(128)  # Stage2：注意力增强
        self.max_pooling_2 = nn.MaxPool2d((2, 2), stride=(2, 2))  # Stage2：下采样

        # 编码器 Stage 3
        self.conv2d_5 = Conv(128, 256)  # Stage3：通道扩展到 256
        self.conv2d_6 = Conv(256, 256)  # Stage3：卷积细化
        self.conv2d_7 = Conv(256, 256)  # Stage3：输出 skip 特征 x3
        self.cbam_3 = CBAM(256)  # Stage3：注意力增强
        self.max_pooling_3 = nn.MaxPool2d((2, 2), stride=(2, 2))  # Stage3：下采样到 bottleneck

        # 编码器 bottleneck
        self.conv2d_8 = Conv(256, 512)  # Bottleneck：通道扩展到 512，聚合更全局的信息
        self.conv2d_9 = Conv(512, 512)  # Bottleneck：卷积细化
        self.conv2d_10 = Conv(512, 512)  # Bottleneck：卷积细化
        self.cbam_4 = CBAM(512)  # Bottleneck：注意力增强

        # 解码器 Stage 1：上采样回到与 x3 相同的空间尺寸，再 concat
        self.up_sampling_1 = nn.UpsamplingNearest2d(scale_factor=2)  # 解码1：上采样恢复空间分辨率

        self.conv2d_11 = Conv(768, 256)  # 解码1：concat 后通道 512+256=768，压缩回 256
        self.conv2d_12 = Conv(256, 256)  # 解码1：卷积细化
        self.conv2d_13 = Conv(256, 256)  # 解码1：卷积细化
        self.cbam_5 = CBAM(256)  # 解码1：注意力增强

        # 解码器 Stage 2：上采样回到与 x2 相同的空间尺寸，再 concat
        self.up_sampling_2 = nn.UpsamplingNearest2d(scale_factor=2)  # 解码2：上采样到 Stage2 分辨率

        self.conv2d_14 = Conv(384, 128)  # 解码2：concat 后通道 256+128=384，压缩回 128
        self.conv2d_15 = Conv(128, 128)  # 解码2：卷积细化
        self.cbam_6 = CBAM(128)  # 解码2：注意力增强

        # 解码器 Stage 3：上采样回到与 x1 相同的空间尺寸，再 concat
        self.up_sampling_3 = nn.UpsamplingNearest2d(scale_factor=2)  # 解码3：上采样到 Stage1 分辨率

        self.conv2d_16 = Conv(192, 64)  # 解码3：concat 后通道 128+64=192，压缩回 64
        self.conv2d_17 = Conv(64, 64)  # 解码3：卷积细化
        self.cbam_7 = CBAM(64)  # 解码3：注意力增强
        # 输出头：把 64 通道映射到 3 通道热力图（例如三类/三通道输出）
        self.conv2d_18 = nn.Conv2d(64, 3, kernel_size=(1, 1), padding='same')  # 输出头：1x1 卷积映射到 3 通道

    def forward(self, x):  # 前向：执行编码->解码，输出 3 通道概率/热力图
        # Stage 1：两层卷积 + 注意力，得到 x1 作为 skip 连接
        x = self.conv2d_1(x)  # Stage1：卷积提取低层特征
        x1 = self.conv2d_2(x)  # Stage1：得到 skip 特征 x1
        x1 = self.cbam_1(x1)  # Stage1：注意力增强 skip 特征
        x = self.max_pooling_1(x1)  # Stage1：下采样进入 Stage2

        # Stage 2：得到 x2 作为 skip 连接
        x = self.conv2d_3(x)  # Stage2：卷积提取中层特征
        x2 = self.conv2d_4(x)  # Stage2：得到 skip 特征 x2
        x2 = self.cbam_2(x2)  # Stage2：注意力增强 skip 特征
        x = self.max_pooling_2(x2)  # Stage2：下采样进入 Stage3

        # Stage 3：得到 x3 作为 skip 连接
        x = self.conv2d_5(x)  # Stage3：卷积提取更深层特征
        x = self.conv2d_6(x)  # Stage3：卷积细化
        x3 = self.conv2d_7(x)  # Stage3：得到 skip 特征 x3
        x3 = self.cbam_3(x3)  # Stage3：注意力增强 skip 特征
        x = self.max_pooling_3(x3)  # Stage3：下采样进入 bottleneck

        # bottleneck：最深层语义特征
        x = self.conv2d_8(x)  # Bottleneck：卷积提取全局语义
        x = self.conv2d_9(x)  # Bottleneck：卷积细化
        x = self.conv2d_10(x)  # Bottleneck：卷积细化
        x = self.cbam_4(x)  # Bottleneck：注意力增强

        # 解码 1：上采样后与 x3 concat（通道数 512 + 256 = 768）
        x = self.up_sampling_1(x)  # 解码1：上采样到与 x3 同分辨率
        x = torch.concat([x, x3], dim=1)  # 解码1：concat skip(x3)，补回高频细节

        x = self.conv2d_11(x)  # 解码1：卷积融合 concat 特征并压缩通道
        x = self.conv2d_12(x)  # 解码1：卷积细化
        x = self.conv2d_13(x)  # 解码1：卷积细化
        x = self.cbam_5(x)  # 解码1：注意力增强

        # 解码 2：上采样后与 x2 concat（通道数 256 + 128 = 384）
        x = self.up_sampling_2(x)  # 解码2：上采样到与 x2 同分辨率
        x = torch.concat([x, x2], dim=1)  # 解码2：concat skip(x2)

        x = self.conv2d_14(x)  # 解码2：卷积融合并压缩通道
        x = self.conv2d_15(x)  # 解码2：卷积细化
        x = self.cbam_6(x)  # 解码2：注意力增强

        # 解码 3：上采样后与 x1 concat（通道数 128 + 64 = 192）
        x = self.up_sampling_3(x)  # 解码3：上采样到与 x1 同分辨率
        x = torch.concat([x, x1], dim=1)  # 解码3：concat skip(x1)

        x = self.conv2d_16(x)  # 解码3：卷积融合并压缩通道
        x = self.conv2d_17(x)  # 解码3：卷积细化
        x = self.cbam_7(x)  # 解码3：注意力增强
        x = self.conv2d_18(x)  # 输出头：映射到 3 通道输出

        # Sigmoid：把输出约束到 [0,1]，便于作为概率/热力图使用
        x = torch.sigmoid(x)  # Sigmoid：将输出转换为 0-1 概率/热力图

        return x  # 返回最终输出张量
