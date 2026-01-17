# TemPose: a new skeleton-based transformer model designed for fine-grained motion recognition in badminton
# (2023/08) https://ieeexplore.ieee.org/document/10208321
# Authors: Magnus Ibh, Stella Grasshof, Dan Witzner, Pascal Madeleine

# Modified by Jing-Yuan Chang

import torch  # 导入模块，供后续使用
from torch import nn, Tensor  # 从模块导入符号，供后续调用
from positional_encodings.torch_encodings import PositionalEncoding1D  # 从模块导入符号，供后续调用
from torchinfo import summary  # 从模块导入符号，供后续调用
from torch.utils.flop_counter import FlopCounterMode  # 从模块导入符号，供后续调用


class MLP(nn.Module):  # 定义类（封装数据与行为）
    '''Same as MLP_Block in TemPose paper.'''  # 执行当前语句（保持与上文逻辑一致）
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:  # 定义函数（封装可复用逻辑）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        self.mlp = nn.Sequential(  # 给对象属性 self.mlp 赋值/初始化（来自当前语句右侧表达式）
            nn.Linear(in_dim, hd_dim),  # 执行当前语句（保持与上文逻辑一致）
            nn.GELU(),  # 执行当前语句（保持与上文逻辑一致）
            nn.Dropout(drop_p, inplace=True),  # 执行当前语句（保持与上文逻辑一致）
            nn.Linear(hd_dim, out_dim)  # 调用函数/方法执行某个动作或计算
        )  # 执行当前语句（保持与上文逻辑一致）

    def forward(self, x: Tensor):  # 定义函数（封装可复用逻辑）
        return self.mlp(x)  # 从函数返回结果


class MLP_Head(nn.Module):  # 定义类（封装数据与行为）
    '''Same as MLP_Head in TemPose.'''  # 执行当前语句（保持与上文逻辑一致）
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:  # 定义函数（封装可复用逻辑）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        self.layer_norm = nn.LayerNorm(in_dim)  # 给对象属性 self.layer_norm 赋值/初始化（来自当前语句右侧表达式）
        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)  # 给对象属性 self.mlp 赋值/初始化（来自当前语句右侧表达式）

    def forward(self, x: Tensor):  # 定义函数（封装可复用逻辑）
        x = self.layer_norm(x)  # 将 x 设为一次调用/构造的返回值
        x = self.mlp(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class FeedForward(nn.Module):  # 定义类（封装数据与行为）
    '''Same as FeedForward in TemPose.'''  # 执行当前语句（保持与上文逻辑一致）
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:  # 定义函数（封装可复用逻辑）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)  # 给对象属性 self.mlp 赋值/初始化（来自当前语句右侧表达式）
        self.dropout = nn.Dropout(drop_p, inplace=True)  # 给对象属性 self.dropout 赋值/初始化（来自当前语句右侧表达式）

    def forward(self, x: Tensor):  # 定义函数（封装可复用逻辑）
        x = self.mlp(x)  # 将 x 设为一次调用/构造的返回值
        x = self.dropout(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class MultiHeadAttention(nn.Module):  # 定义类（封装数据与行为）
    '''Same as Attention in TemPose.'''  # 执行当前语句（保持与上文逻辑一致）
    def __init__(self, d_model, d_head, n_head, drop_p) -> None:  # 定义函数（封装可复用逻辑）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        d_cat = d_head * n_head  # 将表达式计算结果赋给变量 d_cat

        self.h = n_head  # 给对象属性 self.h 赋值/初始化（来自当前语句右侧表达式）
        self.to_qkv = nn.Linear(d_model, d_cat * 3, bias=False)  # 给对象属性 self.to_qkv 赋值/初始化（来自当前语句右侧表达式）
        self.scale = d_head**-0.5  # 给对象属性 self.scale 赋值/初始化（来自当前语句右侧表达式）

        self.attend = nn.Sequential(  # 给对象属性 self.attend 赋值/初始化（来自当前语句右侧表达式）
            nn.Softmax(dim=-1),  # 执行当前语句（保持与上文逻辑一致）
            nn.Dropout(drop_p)  # This shouldn't be inplace.
        )  # 执行当前语句（保持与上文逻辑一致）
        
        self.tail = nn.Sequential(  # 给对象属性 self.tail 赋值/初始化（来自当前语句右侧表达式）
            nn.Linear(d_cat, d_model),  # 执行当前语句（保持与上文逻辑一致）
            nn.Dropout(drop_p, inplace=True)  # 调用函数/方法执行某个动作或计算
        ) if n_head != 1 or d_cat != d_model else nn.Identity()  # 调用函数/方法执行某个动作或计算

    def forward(self, x: Tensor, mask: Tensor = None):  # 定义函数（封装可复用逻辑）
        # x: (b*n, t, d_model)
        bn, t, _ = x.shape  # 执行当前语句（保持与上文逻辑一致）

        qkv: Tensor = self.to_qkv(x)  # 调用函数/方法执行某个动作或计算
        qkv = qkv.view(bn, t, self.h, -1).chunk(3, dim=-1)  # 将 qkv 设为一次调用/构造的返回值
        q, k, v = map(lambda ts: ts.transpose(1, 2), qkv)  # 调用函数/方法执行某个动作或计算
        # q, k, v: (bn, h, t, d_head)

        dots: Tensor = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale  # 执行当前语句（保持与上文逻辑一致）
        # dots: (bn, h, t, t)
        if mask is not None:  # 条件分支判断并选择执行路径
            # mask: (bn, t)
            mask = mask.view(bn, 1, 1, t)  # 将 mask 设为一次调用/构造的返回值
            dots = dots.masked_fill(mask == 0.0, -torch.inf)  # 将 dots 设为一次调用/构造的返回值
        
        coef = self.attend(dots)  # 将 coef 设为一次调用/构造的返回值
        attension: Tensor = coef @ v.contiguous()  # 调用函数/方法执行某个动作或计算
        # attension: (bn, h, t, d_head)
        
        out = attension.transpose(1, 2).reshape(bn, t, -1)  # 将 out 设为一次调用/构造的返回值
        # out: (bn, t, h*d_head)
        out = self.tail(out)  # 将 out 设为一次调用/构造的返回值
        return out  # (bn, t, d_model)


class TransformerLayer(nn.Module):  # 定义类（封装数据与行为）
    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p) -> None:  # 定义函数（封装可复用逻辑）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        self.layer_norm1 = nn.LayerNorm(d_model)  # 给对象属性 self.layer_norm1 赋值/初始化（来自当前语句右侧表达式）
        self.attn = MultiHeadAttention(d_model, d_head, n_head, drop_p)  # 给对象属性 self.attn 赋值/初始化（来自当前语句右侧表达式）
        self.layer_norm2 = nn.LayerNorm(d_model)  # 给对象属性 self.layer_norm2 赋值/初始化（来自当前语句右侧表达式）
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)  # 给对象属性 self.ff 赋值/初始化（来自当前语句右侧表达式）

    def forward(self, x: Tensor, mask=None):  # 定义函数（封装可复用逻辑）
        z = self.layer_norm1(x)  # 将 z 设为一次调用/构造的返回值
        x = self.attn(z, mask) + x  # 将 x 设为一次调用/构造的返回值
        z = self.layer_norm2(x)  # 将 z 设为一次调用/构造的返回值
        x = self.ff(z) + x  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class TransformerEncoder(nn.Module):  # 定义类（封装数据与行为）
    '''Same as Transformer in TemPose.'''  # 执行当前语句（保持与上文逻辑一致）
    def __init__(self, d_model, d_head, n_head, depth, hd_mlp, drop_p) -> None:  # 定义函数（封装可复用逻辑）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        self.layers = nn.ModuleList(  # 给对象属性 self.layers 赋值/初始化（来自当前语句右侧表达式）
            [TransformerLayer(d_model, d_head, n_head, hd_mlp, drop_p)  # 调用函数/方法执行某个动作或计算
             for _ in range(depth)]  # 循环遍历序列/迭代器
        )  # 执行当前语句（保持与上文逻辑一致）

    def forward(self, x: Tensor, mask=None):  # 定义函数（封装可复用逻辑）
        for layer in self.layers:  # 循环遍历序列/迭代器
            x = layer(x, mask)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class TCN(nn.Module):  # 定义类（封装数据与行为）
    '''Same as TCN in TemPose. There is a bit different from the original TCN.'''  # 执行当前语句（保持与上文逻辑一致）
    def __init__(self, in_channel, channels: list[int], kernel_size=5, drop_p=0.3) -> None:  # 定义函数（封装可复用逻辑）
        '''`kernel_size` should be an odd number, so the output sequence length can remain the same as input.'''  # 执行当前语句（保持与上文逻辑一致）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        layers = []  # 初始化变量 layers 为一个容器/表达式结果
        for i in range(len(channels)):  # 循环遍历序列/迭代器
            in_ch = in_channel if i == 0 else channels[i-1]  # 将表达式计算结果赋给变量 in_ch
            out_ch = channels[i]  # 将表达式计算结果赋给变量 out_ch
            
            dilation = i * 2 + 1  # 将表达式计算结果赋给变量 dilation
            padding = (kernel_size - 1) * dilation // 2  # 初始化变量 padding 为一个容器/表达式结果
            layers += [  # 执行当前语句（保持与上文逻辑一致）
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),  # 执行当前语句（保持与上文逻辑一致）
                nn.BatchNorm1d(out_ch),  # 执行当前语句（保持与上文逻辑一致）
                nn.GELU(),  # 执行当前语句（保持与上文逻辑一致）
                nn.Dropout(drop_p, inplace=True)  # 调用函数/方法执行某个动作或计算
            ]  # 执行当前语句（保持与上文逻辑一致）
        self.net = nn.Sequential(*layers)  # 给对象属性 self.net 赋值/初始化（来自当前语句右侧表达式）
    
    def forward(self, x: Tensor):  # 定义函数（封装可复用逻辑）
        return self.net(x)  # 从函数返回结果


class TemPose_V(nn.Module):  # 定义类（封装数据与行为）
    '''Similar to TemPose_TF in TemPose.'''  # 执行当前语句（保持与上文逻辑一致）
    def __init__(  # 定义函数（封装可复用逻辑）
        self, in_dim, seq_len, n_class=35, n_people=2,  # 执行当前语句（保持与上文逻辑一致）
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,  # 将表达式计算结果赋给变量 d_model
        drop_p=0.3, mlp_d_scale=4  # 将表达式计算结果赋给变量 drop_p
    ):  # 执行当前语句（保持与上文逻辑一致）
        super().__init__()  # 调用函数/方法执行某个动作或计算

        self.project = nn.Linear(in_dim, d_model)  # 给对象属性 self.project 赋值/初始化（来自当前语句右侧表达式）

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_tem 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people, 1+seq_len, d_model))  # 给对象属性 self.embedding_tem 赋值/初始化（来自当前语句右侧表达式）
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)  # 给对象属性 self.pre_dropout 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_tem 赋值/初始化（来自当前语句右侧表达式）

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_inter 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people, d_model))  # 给对象属性 self.embedding_inter 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_inter 赋值/初始化（来自当前语句右侧表达式）

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.mlp_head 赋值/初始化（来自当前语句右侧表达式）

        self.d_model = d_model  # 给对象属性 self.d_model 赋值/初始化（来自当前语句右侧表达式）

        self.init_weights()  # 调用函数/方法执行某个动作或计算

    @torch.no_grad()  # 装饰器：修改/包装下方函数或类的行为
    def init_weights(self):  # 定义函数（封装可复用逻辑）
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)  # 将 p_enc_1d_model 设为一次调用/构造的返回值
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))  # 调用函数/方法执行某个动作或计算
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))  # 调用函数/方法执行某个动作或计算

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)  # 调用函数/方法执行某个动作或计算
        self.embedding_inter.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)  # 调用函数/方法执行某个动作或计算
        nn.init.normal_(self.learned_token_inter, std=0.02)  # 调用函数/方法执行某个动作或计算

        self.apply(self.init_weights_recursive)  # 调用函数/方法执行某个动作或计算

    def init_weights_recursive(self, m):  # 定义函数（封装可复用逻辑）
        # Same as TemPose
        if isinstance(m, nn.Linear):  # 条件分支判断并选择执行路径
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)  # 调用函数/方法执行某个动作或计算
            if m.bias is not None:  # 条件分支判断并选择执行路径
                nn.init.constant_(m.bias, 0)  # 调用函数/方法执行某个动作或计算
        elif isinstance(m, nn.Conv1d):  # 条件分支判断并选择执行路径
            nn.init.xavier_normal_(m.weight)  # 调用函数/方法执行某个动作或计算

    def forward(  # 定义函数（封装可复用逻辑）
        self,  # 执行当前语句（保持与上文逻辑一致）
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        video_len: Tensor  # video_len: (b)
    ):  # 执行当前语句（保持与上文逻辑一致）
        JnB = JnB.transpose(1, 2).contiguous()  # 将 JnB 设为一次调用/构造的返回值
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)  # 将 x 设为一次调用/构造的返回值
        b, n, t, d = x.shape  # 执行当前语句（保持与上文逻辑一致）

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)  # 将 class_token_tem 设为一次调用/构造的返回值
        x = torch.cat((class_token_tem, x), dim=2)  # 将 x 设为一次调用/构造的返回值
        t += 1  # 执行当前语句（保持与上文逻辑一致）

        # Temporal embedding
        x = x + self.embedding_tem  # 将表达式计算结果赋给变量 x
        x: Tensor = self.pre_dropout(x)  # 调用函数/方法执行某个动作或计算

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)  # 将 x 设为一次调用/构造的返回值

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)  # 将 range_t 设为一次调用/构造的返回值
        video_len = video_len.unsqueeze(-1)  # 将 video_len 设为一次调用/构造的返回值
        mask = range_t < (1 + video_len)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)  # 将 x 设为一次调用/构造的返回值
        x = x[:, 0].view(b, n, d)  # 将 x 设为一次调用/构造的返回值

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)  # 将 class_token_inter 设为一次调用/构造的返回值
        x = torch.cat((class_token_inter, x), dim=1)  # 将 x 设为一次调用/构造的返回值
        n += 1  # 执行当前语句（保持与上文逻辑一致）

        # Interactional embedding
        x = x + self.embedding_inter  # 将表达式计算结果赋给变量 x

        # Interactional TransformerLayers
        x = self.encoder_inter(x)  # 将 x 设为一次调用/构造的返回值
        x = x[:, 0].contiguous()  # 将 x 设为一次调用/构造的返回值

        x = self.mlp_head(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class TemPose_PF(nn.Module):  # 定义类（封装数据与行为）
    '''For ablation studies.

    Equal to TemPose_TF without the shuttlecock trajectory
    or TemPose_V with the player positions.
    '''
    def __init__(  # 定义函数（封装可复用逻辑）
        self, in_dim, seq_len, n_class=35, n_people=2,  # 执行当前语句（保持与上文逻辑一致）
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,  # 将表达式计算结果赋给变量 d_model
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5  # 将表达式计算结果赋给变量 drop_p
    ):  # 执行当前语句（保持与上文逻辑一致）
        '''`d_model` should be an even number.'''  # 执行当前语句（保持与上文逻辑一致）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        if n_people > 2:  # 条件分支判断并选择执行路径
            raise NotImplementedError  # 执行当前语句（保持与上文逻辑一致）

        self.project = nn.Linear(in_dim, d_model)  # 给对象属性 self.project 赋值/初始化（来自当前语句右侧表达式）

        # TCNs
        tcn_channels = [d_model // 2, d_model]  # 初始化变量 tcn_channels 为一个容器/表达式结果
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_top 赋值/初始化（来自当前语句右侧表达式）
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_bottom 赋值/初始化（来自当前语句右侧表达式）

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_tem 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+2, 1+seq_len, d_model))  # 给对象属性 self.embedding_tem 赋值/初始化（来自当前语句右侧表达式）
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)  # 给对象属性 self.pre_dropout 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_tem 赋值/初始化（来自当前语句右侧表达式）

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_inter 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+2, d_model))  # 给对象属性 self.embedding_inter 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_inter 赋值/初始化（来自当前语句右侧表达式）

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.mlp_head 赋值/初始化（来自当前语句右侧表达式）

        self.d_model = d_model  # 给对象属性 self.d_model 赋值/初始化（来自当前语句右侧表达式）

        self.init_weights()  # 调用函数/方法执行某个动作或计算

    @torch.no_grad()  # 装饰器：修改/包装下方函数或类的行为
    def init_weights(self):  # 定义函数（封装可复用逻辑）
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)  # 将 p_enc_1d_model 设为一次调用/构造的返回值
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))  # 调用函数/方法执行某个动作或计算
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))  # 调用函数/方法执行某个动作或计算

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)  # 调用函数/方法执行某个动作或计算
        self.embedding_inter.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)  # 调用函数/方法执行某个动作或计算
        nn.init.normal_(self.learned_token_inter, std=0.02)  # 调用函数/方法执行某个动作或计算

        self.apply(self.init_weights_recursive)  # 调用函数/方法执行某个动作或计算

    def init_weights_recursive(self, m):  # 定义函数（封装可复用逻辑）
        # Same as TemPose
        if isinstance(m, nn.Linear):  # 条件分支判断并选择执行路径
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)  # 调用函数/方法执行某个动作或计算
            if m.bias is not None:  # 条件分支判断并选择执行路径
                nn.init.constant_(m.bias, 0)  # 调用函数/方法执行某个动作或计算
        elif isinstance(m, nn.Conv1d):  # 条件分支判断并选择执行路径
            nn.init.xavier_normal_(m.weight)  # 调用函数/方法执行某个动作或计算

    def forward(  # 定义函数（封装可复用逻辑）
        self,  # 执行当前语句（保持与上文逻辑一致）
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        pos: Tensor,  # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):  # 执行当前语句（保持与上文逻辑一致）
        JnB = JnB.transpose(1, 2).contiguous()  # 将 JnB 设为一次调用/构造的返回值
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)  # 将 x 设为一次调用/构造的返回值
        b, n, t, d = x.shape  # 执行当前语句（保持与上文逻辑一致）

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()  # 将 pos_top 设为一次调用/构造的返回值
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()  # 将 pos_bottom 设为一次调用/构造的返回值
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)  # 调用函数/方法执行某个动作或计算
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)  # 调用函数/方法执行某个动作或计算
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)

        pos_top = pos_top.transpose(1, 2)  # 将 pos_top 设为一次调用/构造的返回值
        pos_bottom = pos_bottom.transpose(1, 2)  # 将 pos_bottom 设为一次调用/构造的返回值
        x_additional = torch.stack((pos_top, pos_bottom), dim=1)  # 将 x_additional 设为一次调用/构造的返回值
        # x_additional: (b, 2, t, d)

        # Positions Fusion (PF)
        x = torch.cat((x, x_additional), dim=1)  # 将 x 设为一次调用/构造的返回值
        n += 2  # 执行当前语句（保持与上文逻辑一致）

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)  # 将 class_token_tem 设为一次调用/构造的返回值
        x = torch.cat((class_token_tem, x), dim=2)  # 将 x 设为一次调用/构造的返回值
        t += 1  # 执行当前语句（保持与上文逻辑一致）

        # Temporal embedding
        x = x + self.embedding_tem  # 将表达式计算结果赋给变量 x
        x: Tensor = self.pre_dropout(x)  # 调用函数/方法执行某个动作或计算

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)  # 将 x 设为一次调用/构造的返回值

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)  # 将 range_t 设为一次调用/构造的返回值
        video_len = video_len.unsqueeze(-1)  # 将 video_len 设为一次调用/构造的返回值
        mask = range_t < (1 + video_len)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)  # 将 x 设为一次调用/构造的返回值
        x = x[:, 0].view(b, n, d)  # 将 x 设为一次调用/构造的返回值

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)  # 将 class_token_inter 设为一次调用/构造的返回值
        x = torch.cat((class_token_inter, x), dim=1)  # 将 x 设为一次调用/构造的返回值
        n += 1  # 执行当前语句（保持与上文逻辑一致）

        # Interactional embedding
        x = x + self.embedding_inter  # 将表达式计算结果赋给变量 x

        # Interactional TransformerLayers
        x = self.encoder_inter(x)  # 将 x 设为一次调用/构造的返回值
        x = x[:, 0].contiguous()  # 将 x 设为一次调用/构造的返回值

        x = self.mlp_head(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class TemPose_SF(nn.Module):  # 定义类（封装数据与行为）
    '''For ablation studies.

    Equal to TemPose_TF without the player positions
    or TemPose_V with the shuttlecock trajectory.
    '''
    def __init__(  # 定义函数（封装可复用逻辑）
        self, in_dim, seq_len, n_class=35, n_people=2,  # 执行当前语句（保持与上文逻辑一致）
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,  # 将表达式计算结果赋给变量 d_model
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5  # 将表达式计算结果赋给变量 drop_p
    ):  # 执行当前语句（保持与上文逻辑一致）
        '''`d_model` should be an even number.'''  # 执行当前语句（保持与上文逻辑一致）
        super().__init__()  # 调用函数/方法执行某个动作或计算

        self.project = nn.Linear(in_dim, d_model)  # 给对象属性 self.project 赋值/初始化（来自当前语句右侧表达式）

        # TCNs
        tcn_channels = [d_model // 2, d_model]  # 初始化变量 tcn_channels 为一个容器/表达式结果
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_shuttle 赋值/初始化（来自当前语句右侧表达式）

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_tem 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+1, 1+seq_len, d_model))  # 给对象属性 self.embedding_tem 赋值/初始化（来自当前语句右侧表达式）
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)  # 给对象属性 self.pre_dropout 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_tem 赋值/初始化（来自当前语句右侧表达式）

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_inter 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+1, d_model))  # 给对象属性 self.embedding_inter 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_inter 赋值/初始化（来自当前语句右侧表达式）

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.mlp_head 赋值/初始化（来自当前语句右侧表达式）

        self.d_model = d_model  # 给对象属性 self.d_model 赋值/初始化（来自当前语句右侧表达式）

        self.init_weights()  # 调用函数/方法执行某个动作或计算

    @torch.no_grad()  # 装饰器：修改/包装下方函数或类的行为
    def init_weights(self):  # 定义函数（封装可复用逻辑）
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)  # 将 p_enc_1d_model 设为一次调用/构造的返回值
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))  # 调用函数/方法执行某个动作或计算
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))  # 调用函数/方法执行某个动作或计算

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)  # 调用函数/方法执行某个动作或计算
        self.embedding_inter.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)  # 调用函数/方法执行某个动作或计算
        nn.init.normal_(self.learned_token_inter, std=0.02)  # 调用函数/方法执行某个动作或计算

        self.apply(self.init_weights_recursive)  # 调用函数/方法执行某个动作或计算

    def init_weights_recursive(self, m):  # 定义函数（封装可复用逻辑）
        # Same as TemPose
        if isinstance(m, nn.Linear):  # 条件分支判断并选择执行路径
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)  # 调用函数/方法执行某个动作或计算
            if m.bias is not None:  # 条件分支判断并选择执行路径
                nn.init.constant_(m.bias, 0)  # 调用函数/方法执行某个动作或计算
        elif isinstance(m, nn.Conv1d):  # 条件分支判断并选择执行路径
            nn.init.xavier_normal_(m.weight)  # 调用函数/方法执行某个动作或计算

    def forward(  # 定义函数（封装可复用逻辑）
        self,  # 执行当前语句（保持与上文逻辑一致）
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):  # 执行当前语句（保持与上文逻辑一致）
        JnB = JnB.transpose(1, 2).contiguous()  # 将 JnB 设为一次调用/构造的返回值
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)  # 将 x 设为一次调用/构造的返回值
        b, n, t, d = x.shape  # 执行当前语句（保持与上文逻辑一致）

        shuttle = shuttle.transpose(1, 2).contiguous()  # 将 shuttle 设为一次调用/构造的返回值
        # shuttle: (b, 2, t)

        # TCN
        shuttle: Tensor = self.tcn_shuttle(shuttle)  # 调用函数/方法执行某个动作或计算
        # shuttle: (b, d, t)

        shuttle = shuttle.transpose(1, 2).contiguous()  # 将 shuttle 设为一次调用/构造的返回值
        x_additional = shuttle.unsqueeze(1)  # 将 x_additional 设为一次调用/构造的返回值
        # x_additional: (b, 1, t, d)

        # Shuttlecock Fusion (SF)
        x = torch.cat((x, x_additional), dim=1)  # 将 x 设为一次调用/构造的返回值
        n += 1  # 执行当前语句（保持与上文逻辑一致）

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)  # 将 class_token_tem 设为一次调用/构造的返回值
        x = torch.cat((class_token_tem, x), dim=2)  # 将 x 设为一次调用/构造的返回值
        t += 1  # 执行当前语句（保持与上文逻辑一致）

        # Temporal embedding
        x = x + self.embedding_tem  # 将表达式计算结果赋给变量 x
        x: Tensor = self.pre_dropout(x)  # 调用函数/方法执行某个动作或计算

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)  # 将 x 设为一次调用/构造的返回值

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)  # 将 range_t 设为一次调用/构造的返回值
        video_len = video_len.unsqueeze(-1)  # 将 video_len 设为一次调用/构造的返回值
        mask = range_t < (1 + video_len)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)  # 将 x 设为一次调用/构造的返回值
        x = x[:, 0].view(b, n, d)  # 将 x 设为一次调用/构造的返回值

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)  # 将 class_token_inter 设为一次调用/构造的返回值
        x = torch.cat((class_token_inter, x), dim=1)  # 将 x 设为一次调用/构造的返回值
        n += 1  # 执行当前语句（保持与上文逻辑一致）

        # Interactional embedding
        x = x + self.embedding_inter  # 将表达式计算结果赋给变量 x

        # Interactional TransformerLayers
        x = self.encoder_inter(x)  # 将 x 设为一次调用/构造的返回值
        x = x[:, 0].contiguous()  # 将 x 设为一次调用/构造的返回值

        x = self.mlp_head(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class TemPose_TF(nn.Module):  # 定义类（封装数据与行为）
    '''Similar to TemPose_TF in TemPose.'''  # 执行当前语句（保持与上文逻辑一致）
    def __init__(  # 定义函数（封装可复用逻辑）
        self, in_dim, seq_len, n_class=35, n_people=2,  # 执行当前语句（保持与上文逻辑一致）
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,  # 将表达式计算结果赋给变量 d_model
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5  # 将表达式计算结果赋给变量 drop_p
    ):  # 执行当前语句（保持与上文逻辑一致）
        '''`d_model` should be an even number.'''  # 执行当前语句（保持与上文逻辑一致）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        if n_people > 2:  # 条件分支判断并选择执行路径
            raise NotImplementedError  # 执行当前语句（保持与上文逻辑一致）

        self.project = nn.Linear(in_dim, d_model)  # 给对象属性 self.project 赋值/初始化（来自当前语句右侧表达式）

        # TCNs
        tcn_channels = [d_model // 2, d_model]  # 初始化变量 tcn_channels 为一个容器/表达式结果
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_top 赋值/初始化（来自当前语句右侧表达式）
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_bottom 赋值/初始化（来自当前语句右侧表达式）
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_shuttle 赋值/初始化（来自当前语句右侧表达式）

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_tem 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+3, 1+seq_len, d_model))  # 给对象属性 self.embedding_tem 赋值/初始化（来自当前语句右侧表达式）
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)  # 给对象属性 self.pre_dropout 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_tem 赋值/初始化（来自当前语句右侧表达式）

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_inter 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+3, d_model))  # 给对象属性 self.embedding_inter 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_inter 赋值/初始化（来自当前语句右侧表达式）

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.mlp_head 赋值/初始化（来自当前语句右侧表达式）

        self.d_model = d_model  # 给对象属性 self.d_model 赋值/初始化（来自当前语句右侧表达式）

        self.init_weights()  # 调用函数/方法执行某个动作或计算

    @torch.no_grad()  # 装饰器：修改/包装下方函数或类的行为
    def init_weights(self):  # 定义函数（封装可复用逻辑）
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)  # 将 p_enc_1d_model 设为一次调用/构造的返回值
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))  # 调用函数/方法执行某个动作或计算
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))  # 调用函数/方法执行某个动作或计算

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)  # 调用函数/方法执行某个动作或计算
        self.embedding_inter.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)  # 调用函数/方法执行某个动作或计算
        nn.init.normal_(self.learned_token_inter, std=0.02)  # 调用函数/方法执行某个动作或计算

        self.apply(self.init_weights_recursive)  # 调用函数/方法执行某个动作或计算

    def init_weights_recursive(self, m):  # 定义函数（封装可复用逻辑）
        # Same as TemPose
        if isinstance(m, nn.Linear):  # 条件分支判断并选择执行路径
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)  # 调用函数/方法执行某个动作或计算
            if m.bias is not None:  # 条件分支判断并选择执行路径
                nn.init.constant_(m.bias, 0)  # 调用函数/方法执行某个动作或计算
        elif isinstance(m, nn.Conv1d):  # 条件分支判断并选择执行路径
            nn.init.xavier_normal_(m.weight)  # 调用函数/方法执行某个动作或计算

    def forward(  # 定义函数（封装可复用逻辑）
        self,  # 执行当前语句（保持与上文逻辑一致）
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        pos: Tensor,  # pos: (b, t, n, 2)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):  # 执行当前语句（保持与上文逻辑一致）
        JnB = JnB.transpose(1, 2).contiguous()  # 将 JnB 设为一次调用/构造的返回值
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)  # 将 x 设为一次调用/构造的返回值
        b, n, t, d = x.shape  # 执行当前语句（保持与上文逻辑一致）

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()  # 将 pos_top 设为一次调用/构造的返回值
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()  # 将 pos_bottom 设为一次调用/构造的返回值
        shuttle = shuttle.transpose(1, 2).contiguous()  # 将 shuttle 设为一次调用/构造的返回值
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)
        # shuttle: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)  # 调用函数/方法执行某个动作或计算
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)  # 调用函数/方法执行某个动作或计算
        shuttle: Tensor = self.tcn_shuttle(shuttle)  # 调用函数/方法执行某个动作或计算
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)
        # shuttle: (b, d, t)

        pos_top = pos_top.transpose(1, 2)  # 将 pos_top 设为一次调用/构造的返回值
        pos_bottom = pos_bottom.transpose(1, 2)  # 将 pos_bottom 设为一次调用/构造的返回值
        shuttle = shuttle.transpose(1, 2)  # 将 shuttle 设为一次调用/构造的返回值
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)  # 将 x_additional 设为一次调用/构造的返回值
        # x_additional: (b, 3, t, d)

        # Temporal Fusion (TF)
        x = torch.cat((x, x_additional), dim=1)  # 将 x 设为一次调用/构造的返回值
        n += 3  # 执行当前语句（保持与上文逻辑一致）

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)  # 将 class_token_tem 设为一次调用/构造的返回值
        x = torch.cat((class_token_tem, x), dim=2)  # 将 x 设为一次调用/构造的返回值
        t += 1  # 执行当前语句（保持与上文逻辑一致）

        # Temporal embedding
        x = x + self.embedding_tem  # 将表达式计算结果赋给变量 x
        x: Tensor = self.pre_dropout(x)  # 调用函数/方法执行某个动作或计算

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)  # 将 x 设为一次调用/构造的返回值

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)  # 将 range_t 设为一次调用/构造的返回值
        video_len = video_len.unsqueeze(-1)  # 将 video_len 设为一次调用/构造的返回值
        mask = range_t < (1 + video_len)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)  # 将 x 设为一次调用/构造的返回值
        x = x[:, 0].view(b, n, d)  # 将 x 设为一次调用/构造的返回值

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)  # 将 class_token_inter 设为一次调用/构造的返回值
        x = torch.cat((class_token_inter, x), dim=1)  # 将 x 设为一次调用/构造的返回值
        n += 1  # 执行当前语句（保持与上文逻辑一致）

        # Interactional embedding
        x = x + self.embedding_inter  # 将表达式计算结果赋给变量 x

        # Interactional TransformerLayers
        x = self.encoder_inter(x)  # 将 x 设为一次调用/构造的返回值
        x = x[:, 0].contiguous()  # 将 x 设为一次调用/构造的返回值

        x = self.mlp_head(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


if __name__ == '__main__':  # 条件分支判断并选择执行路径
    b, t, n = 1, 100, 2  # 执行当前语句（保持与上文逻辑一致）
    n_features = (17 + 19 * 1) * n  # 初始化变量 n_features 为一个容器/表达式结果
    pose = torch.randn((b, t, n, n_features), dtype=torch.float)  # 将 pose 设为一次调用/构造的返回值
    pos = torch.randn((b, t, n, 2), dtype=torch.float)  # 将 pos 设为一次调用/构造的返回值
    shuttle = torch.randn((b, t, 2), dtype=torch.float)  # 将 shuttle 设为一次调用/构造的返回值
    videos_len = torch.tensor([t], dtype=torch.long).repeat(b)  # 将 videos_len 设为一次调用/构造的返回值
    input_data = [pose, pos, shuttle, videos_len]  # 初始化变量 input_data 为一个容器/表达式结果
    model = TemPose_TF(  # 将表达式计算结果赋给变量 model
        in_dim=n_features,  # 将表达式计算结果赋给变量 in_dim
        seq_len=t,  # 将表达式计算结果赋给变量 seq_len
        n_class=25,  # 将表达式计算结果赋给变量 n_class
        d_model=100  # 将表达式计算结果赋给变量 d_model
    )  # 执行当前语句（保持与上文逻辑一致）
    # summary(model, input_data=input_data, depth=4, device='cpu')

    # Count FLOPs
    flop_counter = FlopCounterMode(display=False)  # 将 flop_counter 设为一次调用/构造的返回值
    with flop_counter:  # 上下文管理：确保资源正确释放
        output = model(*input_data)  # 将 output 设为一次调用/构造的返回值
    flops_per_forward = flop_counter.get_total_flops()  # 将 flops_per_forward 设为一次调用/构造的返回值
    print(f"FLOPs (per forward pass): {flops_per_forward / 1e9:.2f} GFLOPS")  # 调用函数/方法执行某个动作或计算
    
    n_epochs_about = 350  # 将表达式计算结果赋给变量 n_epochs_about
    # on ShuttleSet
    n_training_samples = 25741  # 将表达式计算结果赋给变量 n_training_samples
    n_validate_samples = 4241  # 将表达式计算结果赋给变量 n_validate_samples
    n_testing_samples = 3499  # 将表达式计算结果赋给变量 n_testing_samples

    training_flops = flops_per_forward * n_training_samples * n_epochs_about * 3  # 将表达式计算结果赋给变量 training_flops
    validate_flops = flops_per_forward * n_validate_samples * n_epochs_about  # 将表达式计算结果赋给变量 validate_flops
    testing_flops = flops_per_forward * n_testing_samples  # 将表达式计算结果赋给变量 testing_flops
    print(f"Training FLOPs: {training_flops / 1e15:.2f} PFLOPs")  # 调用函数/方法执行某个动作或计算
    print(f"Validating FLOPs: {validate_flops / 1e15:.2f} PFLOPs")  # 调用函数/方法执行某个动作或计算
    print(f"Testing FLOPs (per 1000 instances): {flops_per_forward * 1000 / 1e12:.2f} TFLOPs")  # 调用函数/方法执行某个动作或计算
    print(f"Testing FLOPs: {testing_flops / 1e12:.2f} TFLOPs")  # 调用函数/方法执行某个动作或计算
    total_flops = training_flops + validate_flops + testing_flops  # 将表达式计算结果赋给变量 total_flops
    print(f"Total FLOPs: {total_flops / 1e15:.2f} PFLOPs")  # 调用函数/方法执行某个动作或计算
