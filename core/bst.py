# Writen by Jing-Yuan Chang

import torch  # 导入模块，供后续使用
from torch import nn, Tensor  # 从模块导入符号，供后续调用
from positional_encodings.torch_encodings import PositionalEncoding1D  # 从模块导入符号，供后续调用
from torchinfo import summary  # 从模块导入符号，供后续调用
from torch.utils.flop_counter import FlopCounterMode  # 从模块导入符号，供后续调用

import sys  # 导入模块，供后续使用
import os  # 导入模块，供后续使用
if __name__ == '__main__':  # 条件分支判断并选择执行路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 调用函数/方法执行某个动作或计算

from .tempose import TCN, FeedForward, MLP, MLP_Head, TransformerEncoder  # 从模块导入符号，供后续调用


class MultiHeadCrossAttention(nn.Module):  # 定义类（封装数据与行为）
    def __init__(self, d_model, d_head, n_head, drop_p) -> None:  # 定义函数（封装可复用逻辑）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        d_cat = d_head * n_head  # 将表达式计算结果赋给变量 d_cat

        self.h = n_head  # 给对象属性 self.h 赋值/初始化（来自当前语句右侧表达式）
        self.to_q = nn.Linear(d_model, d_cat, bias=False)  # 给对象属性 self.to_q 赋值/初始化（来自当前语句右侧表达式）
        self.to_kv = nn.Linear(d_model, d_cat * 2, bias=False)  # 给对象属性 self.to_kv 赋值/初始化（来自当前语句右侧表达式）
        self.scale = d_head**-0.5  # 给对象属性 self.scale 赋值/初始化（来自当前语句右侧表达式）

        self.attend = nn.Sequential(  # 给对象属性 self.attend 赋值/初始化（来自当前语句右侧表达式）
            nn.Softmax(dim=-1),  # 执行当前语句（保持与上文逻辑一致）
            nn.Dropout(drop_p)  # This shouldn't be inplace.
        )  # 执行当前语句（保持与上文逻辑一致）
        
        self.tail = nn.Sequential(  # 给对象属性 self.tail 赋值/初始化（来自当前语句右侧表达式）
            nn.Linear(d_cat, d_model),  # 执行当前语句（保持与上文逻辑一致）
            nn.Dropout(drop_p, inplace=True)  # 调用函数/方法执行某个动作或计算
        ) if n_head != 1 or d_cat != d_model else nn.Identity()  # 调用函数/方法执行某个动作或计算

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor = None):  # 定义函数（封装可复用逻辑）
        # x1, x2: (b, t, d_model)
        q: Tensor = self.to_q(x1)  # 调用函数/方法执行某个动作或计算
        kv: Tensor = self.to_kv(x2)  # 调用函数/方法执行某个动作或计算
        b, t, _ = q.shape  # 执行当前语句（保持与上文逻辑一致）

        q = q.view(b, t, self.h, -1).transpose(1, 2)  # 将 q 设为一次调用/构造的返回值
        kv = kv.view(b, t, self.h, -1).chunk(2, dim=-1)  # 将 kv 设为一次调用/构造的返回值
        k, v = map(lambda ts: ts.transpose(1, 2), kv)  # 调用函数/方法执行某个动作或计算
        # q, k, v: (b, h, t, d_head)

        dots: Tensor = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale  # 执行当前语句（保持与上文逻辑一致）
        # dots: (b, h, t, t)
        if mask is not None:  # 条件分支判断并选择执行路径
            # mask: (b, t)
            mask = mask.view(b, 1, 1, t)  # 将 mask 设为一次调用/构造的返回值
            dots = dots.masked_fill(mask == 0.0, -torch.inf)  # 将 dots 设为一次调用/构造的返回值
        
        coef = self.attend(dots)  # 将 coef 设为一次调用/构造的返回值
        attension: Tensor = coef @ v.contiguous()  # 调用函数/方法执行某个动作或计算
        # attension: (b, h, t, d_head)

        out = attension.transpose(1, 2).reshape(b, t, -1)  # 将 out 设为一次调用/构造的返回值
        # out: (b, t, h*d_head)
        out = self.tail(out)  # 将 out 设为一次调用/构造的返回值
        return out  # (b, t, d_model)


class CrossTransformerLayer(nn.Module):  # 定义类（封装数据与行为）
    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p) -> None:  # 定义函数（封装可复用逻辑）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        self.layer_norm1_x1 = nn.LayerNorm(d_model)  # 给对象属性 self.layer_norm1_x1 赋值/初始化（来自当前语句右侧表达式）
        self.layer_norm1_x2 = nn.LayerNorm(d_model)  # 给对象属性 self.layer_norm1_x2 赋值/初始化（来自当前语句右侧表达式）
        self.cross_attn = MultiHeadCrossAttention(d_model, d_head, n_head, drop_p)  # 给对象属性 self.cross_attn 赋值/初始化（来自当前语句右侧表达式）
        self.layer_norm2 = nn.LayerNorm(d_model)  # 给对象属性 self.layer_norm2 赋值/初始化（来自当前语句右侧表达式）
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)  # 给对象属性 self.ff 赋值/初始化（来自当前语句右侧表达式）

    def forward(self, x1: Tensor, x2: Tensor, mask=None):  # 定义函数（封装可复用逻辑）
        x1 = self.layer_norm1_x1(x1)  # 将 x1 设为一次调用/构造的返回值
        x2 = self.layer_norm1_x2(x2)  # 将 x2 设为一次调用/构造的返回值
        x = self.cross_attn(x1, x2, mask)  # 将 x 设为一次调用/构造的返回值
        z = self.layer_norm2(x)  # 将 z 设为一次调用/构造的返回值
        x = self.ff(z) + x  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class BST_0(nn.Module):  # 定义类（封装数据与行为）
    '''BST-backbone'''  # 执行当前语句（保持与上文逻辑一致）
    def __init__(  # 定义函数（封装可复用逻辑）
        self, in_dim, seq_len, n_class=35, n_people=2,  # 执行当前语句（保持与上文逻辑一致）
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,  # 将表达式计算结果赋给变量 d_model
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5  # 将表达式计算结果赋给变量 drop_p
    ):  # 执行当前语句（保持与上文逻辑一致）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        if n_people > 2:  # 条件分支判断并选择执行路径
            raise NotImplementedError  # 执行当前语句（保持与上文逻辑一致）

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_pose 赋值/初始化（来自当前语句右侧表达式）
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_shuttle 赋值/初始化（来自当前语句右侧表达式）

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_tem 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 给对象属性 self.embedding_tem 赋值/初始化（来自当前语句右侧表达式）
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)  # 给对象属性 self.pre_dropout 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_tem 赋值/初始化（来自当前语句右侧表达式）

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))  # 给对象属性 self.embedding_cross 赋值/初始化（来自当前语句右侧表达式）
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.cross_trans 赋值/初始化（来自当前语句右侧表达式）

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_inter 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 给对象属性 self.embedding_inter 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_inter 赋值/初始化（来自当前语句右侧表达式）
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.mlp_head 赋值/初始化（来自当前语句右侧表达式）

        self.d_model = d_model  # 给对象属性 self.d_model 赋值/初始化（来自当前语句右侧表达式）

        self.init_weights()  # 调用函数/方法执行某个动作或计算

    @torch.no_grad()  # 装饰器：修改/包装下方函数或类的行为
    def init_weights(self):  # 定义函数（封装可复用逻辑）
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)  # 将 p_enc_1d_model 设为一次调用/构造的返回值
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)  # 调用函数/方法执行某个动作或计算
        self.embedding_tem.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)  # 调用函数/方法执行某个动作或计算
        self.embedding_cross.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

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
        b, t, n, in_dim = JnB.shape  # 执行当前语句（保持与上文逻辑一致）
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)  # 将 JnB 设为一次调用/构造的返回值
        JnB = self.tcn_pose(JnB)  # 将 JnB 设为一次调用/构造的返回值
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)  # 将 JnB 设为一次调用/构造的返回值

        shuttle = shuttle.transpose(1, 2).contiguous()  # 将 shuttle 设为一次调用/构造的返回值
        shuttle = self.tcn_shuttle(shuttle)  # 将 shuttle 设为一次调用/构造的返回值
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)  # 将 shuttle 设为一次调用/构造的返回值
        
        x = torch.cat((JnB, shuttle), dim=1)  # 将 x 设为一次调用/构造的返回值
        _, n, _, d = x.shape  # 执行当前语句（保持与上文逻辑一致）

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)  # 将 class_token_tem 设为一次调用/构造的返回值
        x = x.view(b*n, t, d)  # 将 x 设为一次调用/构造的返回值
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem  # 将 x 设为一次调用/构造的返回值

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)  # 将 range_t 设为一次调用/构造的返回值
        video_len = video_len.unsqueeze(-1)  # 将 video_len 设为一次调用/构造的返回值
        mask = range_t < (1 + video_len)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)  # 将 mask_n 设为一次调用/构造的返回值
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)  # 调用函数/方法执行某个动作或计算
        x = self.encoder_tem(x, mask_n)  # 将 x 设为一次调用/构造的返回值
        x = x.view(b, n, 1+t, d)  # 将 x 设为一次调用/构造的返回值

        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))  # 调用函数/方法执行某个动作或计算
        
        p1_cls, p2_cls, shuttle_cls = (  # 执行当前语句（保持与上文逻辑一致）
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()  # 调用函数/方法执行某个动作或计算
        )  # 执行当前语句（保持与上文逻辑一致）

        p1 = p1[:, 1:].contiguous() + self.embedding_cross  # 将 p1 设为一次调用/构造的返回值
        p2 = p2[:, 1:].contiguous() + self.embedding_cross  # 将 p2 设为一次调用/构造的返回值
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross  # 将 shuttle 设为一次调用/构造的返回值

        cross_mask = mask[:, 1:].contiguous()  # 将 cross_mask 设为一次调用/构造的返回值
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)  # 将 p1_shuttle 设为一次调用/构造的返回值
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)  # 将 p2_shuttle 设为一次调用/构造的返回值

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)  # 将 class_token_inter 设为一次调用/构造的返回值
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter  # 将 p1_shuttle 设为一次调用/构造的返回值
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter  # 将 p2_shuttle 设为一次调用/构造的返回值

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)  # 调用函数/方法执行某个动作或计算
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)  # 调用函数/方法执行某个动作或计算

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()  # 将 p1_shuttle_cls 设为一次调用/构造的返回值
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()  # 将 p2_shuttle_cls 设为一次调用/构造的返回值

        p1_conclusion = p1_cls + p1_shuttle_cls  # 将表达式计算结果赋给变量 p1_conclusion
        p2_conclusion = p2_cls + p2_shuttle_cls  # 将表达式计算结果赋给变量 p2_conclusion

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)  # 将 x 设为一次调用/构造的返回值
        x = self.mlp_head(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class BST(nn.Module):  # 定义类（封装数据与行为）
    '''BST
    - PPF: Pose Position Fusion
    '''
    def __init__(  # 定义函数（封装可复用逻辑）
        self, in_dim, seq_len, n_class=35, n_people=2,  # 执行当前语句（保持与上文逻辑一致）
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,  # 将表达式计算结果赋给变量 d_model
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5  # 将表达式计算结果赋给变量 drop_p
    ):  # 执行当前语句（保持与上文逻辑一致）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        if n_people > 2:  # 条件分支判断并选择执行路径
            raise NotImplementedError  # 执行当前语句（保持与上文逻辑一致）

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)  # 给对象属性 self.mlp_positions 赋值/初始化（来自当前语句右侧表达式）

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_pose 赋值/初始化（来自当前语句右侧表达式）
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_shuttle 赋值/初始化（来自当前语句右侧表达式）

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_tem 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 给对象属性 self.embedding_tem 赋值/初始化（来自当前语句右侧表达式）
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)  # 给对象属性 self.pre_dropout 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_tem 赋值/初始化（来自当前语句右侧表达式）

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))  # 给对象属性 self.embedding_cross 赋值/初始化（来自当前语句右侧表达式）
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.cross_trans 赋值/初始化（来自当前语句右侧表达式）

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_inter 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 给对象属性 self.embedding_inter 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_inter 赋值/初始化（来自当前语句右侧表达式）
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.mlp_head 赋值/初始化（来自当前语句右侧表达式）

        self.d_model = d_model  # 给对象属性 self.d_model 赋值/初始化（来自当前语句右侧表达式）

        self.init_weights()  # 调用函数/方法执行某个动作或计算

    @torch.no_grad()  # 装饰器：修改/包装下方函数或类的行为
    def init_weights(self):  # 定义函数（封装可复用逻辑）
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)  # 将 p_enc_1d_model 设为一次调用/构造的返回值
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)  # 调用函数/方法执行某个动作或计算
        self.embedding_tem.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)  # 调用函数/方法执行某个动作或计算
        self.embedding_cross.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

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
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):  # 执行当前语句（保持与上文逻辑一致）
        b, t, n, in_dim = JnB.shape  # 执行当前语句（保持与上文逻辑一致）
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)  # 将 JnB 设为一次调用/构造的返回值
        
        pos = self.mlp_positions(pos)  # 将 pos 设为一次调用/构造的返回值
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)  # 将 pos_impact 设为一次调用/构造的返回值

        JnB = JnB * pos_impact + JnB  # 将表达式计算结果赋给变量 JnB

        JnB = self.tcn_pose(JnB)  # 将 JnB 设为一次调用/构造的返回值
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)  # 将 JnB 设为一次调用/构造的返回值

        shuttle = shuttle.transpose(1, 2).contiguous()  # 将 shuttle 设为一次调用/构造的返回值
        shuttle = self.tcn_shuttle(shuttle)  # 将 shuttle 设为一次调用/构造的返回值
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)  # 将 shuttle 设为一次调用/构造的返回值
        
        x = torch.cat((JnB, shuttle), dim=1)  # 将 x 设为一次调用/构造的返回值
        _, n, _, d = x.shape  # 执行当前语句（保持与上文逻辑一致）

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)  # 将 class_token_tem 设为一次调用/构造的返回值
        x = x.view(b*n, t, d)  # 将 x 设为一次调用/构造的返回值
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem  # 将 x 设为一次调用/构造的返回值

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)  # 将 range_t 设为一次调用/构造的返回值
        video_len = video_len.unsqueeze(-1)  # 将 video_len 设为一次调用/构造的返回值
        mask = range_t < (1 + video_len)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)  # 将 mask_n 设为一次调用/构造的返回值
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)  # 调用函数/方法执行某个动作或计算
        x = self.encoder_tem(x, mask_n)  # 将 x 设为一次调用/构造的返回值
        x = x.view(b, n, 1+t, d)  # 将 x 设为一次调用/构造的返回值

        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))  # 调用函数/方法执行某个动作或计算
        
        p1_cls, p2_cls, shuttle_cls = (  # 执行当前语句（保持与上文逻辑一致）
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()  # 调用函数/方法执行某个动作或计算
        )  # 执行当前语句（保持与上文逻辑一致）

        p1 = p1[:, 1:].contiguous() + self.embedding_cross  # 将 p1 设为一次调用/构造的返回值
        p2 = p2[:, 1:].contiguous() + self.embedding_cross  # 将 p2 设为一次调用/构造的返回值
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross  # 将 shuttle 设为一次调用/构造的返回值

        cross_mask = mask[:, 1:].contiguous()  # 将 cross_mask 设为一次调用/构造的返回值
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)  # 将 p1_shuttle 设为一次调用/构造的返回值
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)  # 将 p2_shuttle 设为一次调用/构造的返回值

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)  # 将 class_token_inter 设为一次调用/构造的返回值
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter  # 将 p1_shuttle 设为一次调用/构造的返回值
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter  # 将 p2_shuttle 设为一次调用/构造的返回值

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)  # 调用函数/方法执行某个动作或计算
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)  # 调用函数/方法执行某个动作或计算

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()  # 将 p1_shuttle_cls 设为一次调用/构造的返回值
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()  # 将 p2_shuttle_cls 设为一次调用/构造的返回值

        p1_conclusion = p1_cls + p1_shuttle_cls  # 将表达式计算结果赋给变量 p1_conclusion
        p2_conclusion = p2_cls + p2_shuttle_cls  # 将表达式计算结果赋给变量 p2_conclusion

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)  # 将 x 设为一次调用/构造的返回值
        x = self.mlp_head(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class BST_CG(nn.Module):  # 定义类（封装数据与行为）
    '''BST
    - PPF: Pose Position Fusion
    - Adding Clean Gate
    '''
    def __init__(  # 定义函数（封装可复用逻辑）
        self, in_dim, seq_len, n_class=35, n_people=2,  # 执行当前语句（保持与上文逻辑一致）
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,  # 将表达式计算结果赋给变量 d_model
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5  # 将表达式计算结果赋给变量 drop_p
    ):  # 执行当前语句（保持与上文逻辑一致）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        if n_people > 2:  # 条件分支判断并选择执行路径
            raise NotImplementedError  # 执行当前语句（保持与上文逻辑一致）

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)  # 给对象属性 self.mlp_positions 赋值/初始化（来自当前语句右侧表达式）

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_pose 赋值/初始化（来自当前语句右侧表达式）
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_shuttle 赋值/初始化（来自当前语句右侧表达式）

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_tem 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 给对象属性 self.embedding_tem 赋值/初始化（来自当前语句右侧表达式）
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)  # 给对象属性 self.pre_dropout 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_tem 赋值/初始化（来自当前语句右侧表达式）

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))  # 给对象属性 self.embedding_cross 赋值/初始化（来自当前语句右侧表达式）
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.cross_trans 赋值/初始化（来自当前语句右侧表达式）

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_inter 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 给对象属性 self.embedding_inter 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_inter 赋值/初始化（来自当前语句右侧表达式）
        
        # Clean Gate
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)  # 给对象属性 self.mlp_clean 赋值/初始化（来自当前语句右侧表达式）

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.mlp_head 赋值/初始化（来自当前语句右侧表达式）

        self.d_model = d_model  # 给对象属性 self.d_model 赋值/初始化（来自当前语句右侧表达式）

        self.init_weights()  # 调用函数/方法执行某个动作或计算

    @torch.no_grad()  # 装饰器：修改/包装下方函数或类的行为
    def init_weights(self):  # 定义函数（封装可复用逻辑）
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)  # 将 p_enc_1d_model 设为一次调用/构造的返回值
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)  # 调用函数/方法执行某个动作或计算
        self.embedding_tem.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)  # 调用函数/方法执行某个动作或计算
        self.embedding_cross.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

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
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):  # 执行当前语句（保持与上文逻辑一致）
        b, t, n, in_dim = JnB.shape  # 执行当前语句（保持与上文逻辑一致）
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)  # 将 JnB 设为一次调用/构造的返回值
        
        pos = self.mlp_positions(pos)  # 将 pos 设为一次调用/构造的返回值
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)  # 将 pos_impact 设为一次调用/构造的返回值

        JnB = JnB * pos_impact + JnB  # 将表达式计算结果赋给变量 JnB

        JnB = self.tcn_pose(JnB)  # 将 JnB 设为一次调用/构造的返回值
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)  # 将 JnB 设为一次调用/构造的返回值

        shuttle = shuttle.transpose(1, 2).contiguous()  # 将 shuttle 设为一次调用/构造的返回值
        shuttle = self.tcn_shuttle(shuttle)  # 将 shuttle 设为一次调用/构造的返回值
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)  # 将 shuttle 设为一次调用/构造的返回值
        
        x = torch.cat((JnB, shuttle), dim=1)  # 将 x 设为一次调用/构造的返回值
        _, n, _, d = x.shape  # 执行当前语句（保持与上文逻辑一致）

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)  # 将 class_token_tem 设为一次调用/构造的返回值
        x = x.view(b*n, t, d)  # 将 x 设为一次调用/构造的返回值
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem  # 将 x 设为一次调用/构造的返回值

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)  # 将 range_t 设为一次调用/构造的返回值
        video_len = video_len.unsqueeze(-1)  # 将 video_len 设为一次调用/构造的返回值
        mask = range_t < (1 + video_len)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)  # 将 mask_n 设为一次调用/构造的返回值
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)  # 调用函数/方法执行某个动作或计算
        x = self.encoder_tem(x, mask_n)  # 将 x 设为一次调用/构造的返回值
        x = x.view(b, n, 1+t, d)  # 将 x 设为一次调用/构造的返回值

        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))  # 调用函数/方法执行某个动作或计算
        
        p1_cls, p2_cls, shuttle_cls = (  # 执行当前语句（保持与上文逻辑一致）
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()  # 调用函数/方法执行某个动作或计算
        )  # 执行当前语句（保持与上文逻辑一致）

        p1 = p1[:, 1:].contiguous() + self.embedding_cross  # 将 p1 设为一次调用/构造的返回值
        p2 = p2[:, 1:].contiguous() + self.embedding_cross  # 将 p2 设为一次调用/构造的返回值
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross  # 将 shuttle 设为一次调用/构造的返回值

        cross_mask = mask[:, 1:].contiguous()  # 将 cross_mask 设为一次调用/构造的返回值
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)  # 将 p1_shuttle 设为一次调用/构造的返回值
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)  # 将 p2_shuttle 设为一次调用/构造的返回值

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)  # 将 class_token_inter 设为一次调用/构造的返回值
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter  # 将 p1_shuttle 设为一次调用/构造的返回值
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter  # 将 p2_shuttle 设为一次调用/构造的返回值

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)  # 调用函数/方法执行某个动作或计算
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)  # 调用函数/方法执行某个动作或计算

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()  # 将 p1_shuttle_cls 设为一次调用/构造的返回值
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()  # 将 p2_shuttle_cls 设为一次调用/构造的返回值

        # Clean Gate
        info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)  # 将 info_need_clean 设为一次调用/构造的返回值
        dirt = self.mlp_clean(info_need_clean)  # 将 dirt 设为一次调用/构造的返回值
        shuttle_cls = shuttle_cls - dirt  # 将表达式计算结果赋给变量 shuttle_cls

        p1_conclusion = p1_cls + p1_shuttle_cls  # 将表达式计算结果赋给变量 p1_conclusion
        p2_conclusion = p2_cls + p2_shuttle_cls  # 将表达式计算结果赋给变量 p2_conclusion

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)  # 将 x 设为一次调用/构造的返回值
        x = self.mlp_head(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class BST_AP(nn.Module):  # 定义类（封装数据与行为）
    '''BST_AimPlayer
    - PPF: Pose Position Fusion
    - Adding Cosine Simularity to determine alpha
    '''
    def __init__(  # 定义函数（封装可复用逻辑）
        self, in_dim, seq_len, n_class=35, n_people=2,  # 执行当前语句（保持与上文逻辑一致）
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,  # 将表达式计算结果赋给变量 d_model
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5  # 将表达式计算结果赋给变量 drop_p
    ):  # 执行当前语句（保持与上文逻辑一致）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        if n_people > 2:  # 条件分支判断并选择执行路径
            raise NotImplementedError  # 执行当前语句（保持与上文逻辑一致）

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)  # 给对象属性 self.mlp_positions 赋值/初始化（来自当前语句右侧表达式）

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_pose 赋值/初始化（来自当前语句右侧表达式）
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_shuttle 赋值/初始化（来自当前语句右侧表达式）

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_tem 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 给对象属性 self.embedding_tem 赋值/初始化（来自当前语句右侧表达式）
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)  # 给对象属性 self.pre_dropout 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_tem 赋值/初始化（来自当前语句右侧表达式）

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))  # 给对象属性 self.embedding_cross 赋值/初始化（来自当前语句右侧表达式）
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.cross_trans 赋值/初始化（来自当前语句右侧表达式）

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_inter 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 给对象属性 self.embedding_inter 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_inter 赋值/初始化（来自当前语句右侧表达式）
        
        # Cosine Simularity
        self.cos_sim = nn.CosineSimilarity()  # 给对象属性 self.cos_sim 赋值/初始化（来自当前语句右侧表达式）

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.mlp_head 赋值/初始化（来自当前语句右侧表达式）

        self.d_model = d_model  # 给对象属性 self.d_model 赋值/初始化（来自当前语句右侧表达式）

        self.init_weights()  # 调用函数/方法执行某个动作或计算

    @torch.no_grad()  # 装饰器：修改/包装下方函数或类的行为
    def init_weights(self):  # 定义函数（封装可复用逻辑）
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)  # 将 p_enc_1d_model 设为一次调用/构造的返回值
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)  # 调用函数/方法执行某个动作或计算
        self.embedding_tem.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)  # 调用函数/方法执行某个动作或计算
        self.embedding_cross.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

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
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):  # 执行当前语句（保持与上文逻辑一致）
        b, t, n, in_dim = JnB.shape  # 执行当前语句（保持与上文逻辑一致）
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)  # 将 JnB 设为一次调用/构造的返回值
        
        pos = self.mlp_positions(pos)  # 将 pos 设为一次调用/构造的返回值
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)  # 将 pos_impact 设为一次调用/构造的返回值

        JnB = JnB * pos_impact + JnB  # 将表达式计算结果赋给变量 JnB

        JnB = self.tcn_pose(JnB)  # 将 JnB 设为一次调用/构造的返回值
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)  # 将 JnB 设为一次调用/构造的返回值

        shuttle = shuttle.transpose(1, 2).contiguous()  # 将 shuttle 设为一次调用/构造的返回值
        shuttle = self.tcn_shuttle(shuttle)  # 将 shuttle 设为一次调用/构造的返回值
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)  # 将 shuttle 设为一次调用/构造的返回值
        
        x = torch.cat((JnB, shuttle), dim=1)  # 将 x 设为一次调用/构造的返回值
        _, n, _, d = x.shape  # 执行当前语句（保持与上文逻辑一致）

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)  # 将 class_token_tem 设为一次调用/构造的返回值
        x = x.view(b*n, t, d)  # 将 x 设为一次调用/构造的返回值
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem  # 将 x 设为一次调用/构造的返回值

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)  # 将 range_t 设为一次调用/构造的返回值
        video_len = video_len.unsqueeze(-1)  # 将 video_len 设为一次调用/构造的返回值
        mask = range_t < (1 + video_len)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)  # 将 mask_n 设为一次调用/构造的返回值
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)  # 调用函数/方法执行某个动作或计算
        x = self.encoder_tem(x, mask_n)  # 将 x 设为一次调用/构造的返回值
        x = x.view(b, n, 1+t, d)  # 将 x 设为一次调用/构造的返回值

        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))  # 调用函数/方法执行某个动作或计算
        
        p1_cls, p2_cls, shuttle_cls = (  # 执行当前语句（保持与上文逻辑一致）
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()  # 调用函数/方法执行某个动作或计算
        )  # 执行当前语句（保持与上文逻辑一致）

        p1 = p1[:, 1:].contiguous() + self.embedding_cross  # 将 p1 设为一次调用/构造的返回值
        p2 = p2[:, 1:].contiguous() + self.embedding_cross  # 将 p2 设为一次调用/构造的返回值
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross  # 将 shuttle 设为一次调用/构造的返回值

        cross_mask = mask[:, 1:].contiguous()  # 将 cross_mask 设为一次调用/构造的返回值
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)  # 将 p1_shuttle 设为一次调用/构造的返回值
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)  # 将 p2_shuttle 设为一次调用/构造的返回值

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)  # 将 class_token_inter 设为一次调用/构造的返回值
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter  # 将 p1_shuttle 设为一次调用/构造的返回值
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter  # 将 p2_shuttle 设为一次调用/构造的返回值

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)  # 调用函数/方法执行某个动作或计算
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)  # 调用函数/方法执行某个动作或计算

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()  # 将 p1_shuttle_cls 设为一次调用/构造的返回值
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()  # 将 p2_shuttle_cls 设为一次调用/构造的返回值

        p1_conclusion = p1_cls + p1_shuttle_cls  # 将表达式计算结果赋给变量 p1_conclusion
        p2_conclusion = p2_cls + p2_shuttle_cls  # 将表达式计算结果赋给变量 p2_conclusion

        # Compute Cosine Simularities
        p1_shuttle_sim = self.cos_sim(p1_shuttle_cls, shuttle_cls)  # 将 p1_shuttle_sim 设为一次调用/构造的返回值
        p2_shuttle_sim = self.cos_sim(p2_shuttle_cls, shuttle_cls)  # 将 p2_shuttle_sim 设为一次调用/构造的返回值
        alpha: Tensor = (p1_shuttle_sim - p2_shuttle_sim + 2) / 4  # 执行当前语句（保持与上文逻辑一致）
        alpha = alpha.unsqueeze(1)  # 将 alpha 设为一次调用/构造的返回值

        p1_conclusion = alpha * p1_conclusion  # 将表达式计算结果赋给变量 p1_conclusion
        p2_conclusion = (1-alpha) * p2_conclusion  # 初始化变量 p2_conclusion 为一个容器/表达式结果

        x = torch.cat((p1_conclusion, p2_conclusion), dim=1)  # 将 x 设为一次调用/构造的返回值
        x = self.mlp_head(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


class BST_CG_AP(nn.Module):  # 定义类（封装数据与行为）
    '''BST_CleanGate_AimPlayer
    - PPF: Pose Position Fusion
    - Adding Clean Gate and Cosine Simularity
    '''
    def __init__(  # 定义函数（封装可复用逻辑）
        self, in_dim, seq_len, n_class=35, n_people=2,  # 执行当前语句（保持与上文逻辑一致）
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,  # 将表达式计算结果赋给变量 d_model
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5  # 将表达式计算结果赋给变量 drop_p
    ):  # 执行当前语句（保持与上文逻辑一致）
        super().__init__()  # 调用函数/方法执行某个动作或计算
        if n_people > 2:  # 条件分支判断并选择执行路径
            raise NotImplementedError  # 执行当前语句（保持与上文逻辑一致）

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)  # 给对象属性 self.mlp_positions 赋值/初始化（来自当前语句右侧表达式）

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_pose 赋值/初始化（来自当前语句右侧表达式）
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)  # 给对象属性 self.tcn_shuttle 赋值/初始化（来自当前语句右侧表达式）

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_tem 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 给对象属性 self.embedding_tem 赋值/初始化（来自当前语句右侧表达式）
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)  # 给对象属性 self.pre_dropout 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_tem 赋值/初始化（来自当前语句右侧表达式）

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))  # 给对象属性 self.embedding_cross 赋值/初始化（来自当前语句右侧表达式）
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.cross_trans 赋值/初始化（来自当前语句右侧表达式）

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))  # 给对象属性 self.learned_token_inter 赋值/初始化（来自当前语句右侧表达式）
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))  # 给对象属性 self.embedding_inter 赋值/初始化（来自当前语句右侧表达式）
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.encoder_inter 赋值/初始化（来自当前语句右侧表达式）
        
        # Cosine Simularity
        self.cos_sim = nn.CosineSimilarity()  # 给对象属性 self.cos_sim 赋值/初始化（来自当前语句右侧表达式）

        # Clean Gate
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)  # 给对象属性 self.mlp_clean 赋值/初始化（来自当前语句右侧表达式）

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)  # 给对象属性 self.mlp_head 赋值/初始化（来自当前语句右侧表达式）

        self.d_model = d_model  # 给对象属性 self.d_model 赋值/初始化（来自当前语句右侧表达式）

        self.init_weights()  # 调用函数/方法执行某个动作或计算

    @torch.no_grad()  # 装饰器：修改/包装下方函数或类的行为
    def init_weights(self):  # 定义函数（封装可复用逻辑）
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)  # 将 p_enc_1d_model 设为一次调用/构造的返回值
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)  # 调用函数/方法执行某个动作或计算
        self.embedding_tem.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)  # 调用函数/方法执行某个动作或计算
        self.embedding_cross.copy_(pos_encoding)  # 调用函数/方法执行某个动作或计算

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
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):  # 执行当前语句（保持与上文逻辑一致）
        b, t, n, in_dim = JnB.shape  # 执行当前语句（保持与上文逻辑一致）
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)  # 将 JnB 设为一次调用/构造的返回值
        
        pos = self.mlp_positions(pos)  # 将 pos 设为一次调用/构造的返回值
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)  # 将 pos_impact 设为一次调用/构造的返回值

        JnB = JnB * pos_impact + JnB  # 将表达式计算结果赋给变量 JnB

        JnB = self.tcn_pose(JnB)  # 将 JnB 设为一次调用/构造的返回值
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)  # 将 JnB 设为一次调用/构造的返回值

        shuttle = shuttle.transpose(1, 2).contiguous()  # 将 shuttle 设为一次调用/构造的返回值
        shuttle = self.tcn_shuttle(shuttle)  # 将 shuttle 设为一次调用/构造的返回值
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)  # 将 shuttle 设为一次调用/构造的返回值
        
        x = torch.cat((JnB, shuttle), dim=1)  # 将 x 设为一次调用/构造的返回值
        _, n, _, d = x.shape  # 执行当前语句（保持与上文逻辑一致）

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)  # 将 class_token_tem 设为一次调用/构造的返回值
        x = x.view(b*n, t, d)  # 将 x 设为一次调用/构造的返回值
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem  # 将 x 设为一次调用/构造的返回值

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)  # 将 range_t 设为一次调用/构造的返回值
        video_len = video_len.unsqueeze(-1)  # 将 video_len 设为一次调用/构造的返回值
        mask = range_t < (1 + video_len)  # 将 mask 设为一次调用/构造的返回值
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)  # 将 mask_n 设为一次调用/构造的返回值
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)  # 调用函数/方法执行某个动作或计算
        x = self.encoder_tem(x, mask_n)  # 将 x 设为一次调用/构造的返回值
        x = x.view(b, n, 1+t, d)  # 将 x 设为一次调用/构造的返回值

        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))  # 调用函数/方法执行某个动作或计算
        
        p1_cls, p2_cls, shuttle_cls = (  # 执行当前语句（保持与上文逻辑一致）
            p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()  # 调用函数/方法执行某个动作或计算
        )  # 执行当前语句（保持与上文逻辑一致）

        p1 = p1[:, 1:].contiguous() + self.embedding_cross  # 将 p1 设为一次调用/构造的返回值
        p2 = p2[:, 1:].contiguous() + self.embedding_cross  # 将 p2 设为一次调用/构造的返回值
        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross  # 将 shuttle 设为一次调用/构造的返回值

        cross_mask = mask[:, 1:].contiguous()  # 将 cross_mask 设为一次调用/构造的返回值
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)  # 将 p1_shuttle 设为一次调用/构造的返回值
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)  # 将 p2_shuttle 设为一次调用/构造的返回值

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)  # 将 class_token_inter 设为一次调用/构造的返回值
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter  # 将 p1_shuttle 设为一次调用/构造的返回值
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter  # 将 p2_shuttle 设为一次调用/构造的返回值

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)  # 调用函数/方法执行某个动作或计算
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)  # 调用函数/方法执行某个动作或计算

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()  # 将 p1_shuttle_cls 设为一次调用/构造的返回值
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()  # 将 p2_shuttle_cls 设为一次调用/构造的返回值

        # Compute Cosine Simularities
        p1_shuttle_sim = self.cos_sim(p1_shuttle_cls, shuttle_cls)  # 将 p1_shuttle_sim 设为一次调用/构造的返回值
        p2_shuttle_sim = self.cos_sim(p2_shuttle_cls, shuttle_cls)  # 将 p2_shuttle_sim 设为一次调用/构造的返回值
        alpha: Tensor = (p1_shuttle_sim - p2_shuttle_sim + 2) / 4  # 执行当前语句（保持与上文逻辑一致）
        alpha = alpha.unsqueeze(1)  # 将 alpha 设为一次调用/构造的返回值

        p1_conclusion = p1_cls + p1_shuttle_cls  # 将表达式计算结果赋给变量 p1_conclusion
        p2_conclusion = p2_cls + p2_shuttle_cls  # 将表达式计算结果赋给变量 p2_conclusion

        p1_conclusion = alpha * p1_conclusion  # 将表达式计算结果赋给变量 p1_conclusion
        p2_conclusion = (1-alpha) * p2_conclusion  # 初始化变量 p2_conclusion 为一个容器/表达式结果

        # Clean Gate
        info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)  # 将 info_need_clean 设为一次调用/构造的返回值
        dirt = self.mlp_clean(info_need_clean)  # 将 dirt 设为一次调用/构造的返回值
        shuttle_cls = shuttle_cls - dirt  # 将表达式计算结果赋给变量 shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)  # 将 x 设为一次调用/构造的返回值
        x = self.mlp_head(x)  # 将 x 设为一次调用/构造的返回值
        return x  # 从函数返回结果


if __name__ == '__main__':  # 条件分支判断并选择执行路径
    b, t, n = 1, 100, 2  # 执行当前语句（保持与上文逻辑一致）
    n_features = (17 + 19 * 1) * n  # 初始化变量 n_features 为一个容器/表达式结果
    pose = torch.randn((b, t, n, n_features), dtype=torch.float)  # 将 pose 设为一次调用/构造的返回值
    shuttle = torch.randn((b, t, 2), dtype=torch.float)  # 将 shuttle 设为一次调用/构造的返回值
    pos = torch.randn((b, t, n, 2), dtype=torch.float)  # 将 pos 设为一次调用/构造的返回值
    videos_len = torch.tensor([t], dtype=torch.long).repeat(b)  # 将 videos_len 设为一次调用/构造的返回值
    input_data = [pose, shuttle, pos, videos_len]  # 初始化变量 input_data 为一个容器/表达式结果
    model = BST_CG_AP(  # 将表达式计算结果赋给变量 model
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
