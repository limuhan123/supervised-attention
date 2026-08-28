from .model_tool import *

class YOLOv8_small_self_branch(nn.Module):
    def __init__(self, nc=80, scales=(0.33, 0.25, 1024)):
        super().__init__()
        depth, width, max_channels = scales
        
        def ch(channels):  # helper for channels
            return min(int(channels * width), max_channels)
            # return max_channels

        def dep(n):
            return max(round(n * depth), 1)

        # Backbone
        self.backbone = nn.ModuleList([
            Conv(3, ch(64), 3, 2),  # 0-P1/2
            Conv(ch(64), ch(128), 3, 2),  # 1-P2/4
            C2f(ch(128), ch(128), n=dep(3 * depth), shortcut=True),  # 2
            Conv(ch(128), ch(256), 3, 2),  # 3-P3/8
            C2f(ch(256), ch(256), n=dep(6 * depth), shortcut=True),  # 4
            Conv(ch(256), ch(512), 3, 2),  # 5-P4/16
            C2f(ch(512), ch(512), n=dep(6 * depth), shortcut=True),  # 6
            SPPF(ch(512), ch(512), 5),  # 7
            C2f(ch(512), ch(512), n=dep(3 * depth), shortcut=False),  # 8
        ])

        self.attention_down = Conv(ch(256),ch(512),3,2)
        self.attention_heatmap = nn.Conv2d(ch(512),1,kernel_size=1)
        self.branch_scale = nn.Parameter(torch.tensor(0.5))

        # Neck
        self.head = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),  # 9
            Concat(dimension=1),  # 10
            C2f(ch(512) + ch(256), ch(256), n=dep(3 * depth), shortcut=False),  # 11 (P3/8-small)
            
            Conv(ch(256), ch(256), 3, 2),  # 12
            Concat(dimension=1),  # 13
            C2f(ch(256) + ch(512), ch(512), n=dep(3 * depth), shortcut=False),  # 14 (P4/16-medium)
        ])

        # Detect head
        self.detect = Detect(nc, (ch(256), ch(512)))
        # 初始化stride和bias
        self.stride = torch.tensor([8., 16.])  # 根据实际特征图下采样率设置
        self.detect.stride = self.stride #设置采样率
        self.detect.bias_init()  # 初始化检测头的偏置

    def forward(self, x, return_loss=False):
        """Forward pass through YOLOv8 model."""
        # -------------------- backbone --------------------
        backbone_features = []
        for i, module in enumerate(self.backbone):
            x = module(x)
            if i == 4 :
                # 从中间单独拿出一部分进行采样
                attention_feature = self.attention_down(x)
                branch = self.attention_heatmap(attention_feature)
                # att_mask = torch.sigmoid(branch)
                # x = x * (1.0 + self.branch_scale * att_mask)

            elif i == 5 :
                # 此处为p4,尺寸为1/16
                att_mask = torch.sigmoid(branch)
                x = x * (1.0 + self.branch_scale * att_mask)

            if i in [4, 8]:  # 保存P3, P4
                backbone_features.append(x)
        p3, p4 = backbone_features  # 更直观命名

        # -------------------- head --------------------
        # P4 -> P3
        x = self.head[0](p4)  # upsample
        x = self.head[1]([x, p3])  # concat with backbone P3
        p3_out = self.head[2](x)   # C2f

        # P3 -> P4
        x = self.head[3](p3_out)  # upsample
        x = self.head[4]([x, p4]) # concat with backbone P3
        p4_out = self.head[5](x)  # C2f (P3/8-small)

        # -------------------- detect --------------------
        return self.detect([p3_out, p4_out], return_loss=return_loss),branch


class Net_learn_self_branch(nn.Module):
    """YOLOv8 object detection model."""
    def __init__(self, nc=80, scales=(0.33, 0.25, 1024)):
        super().__init__()
        depth, width, max_channels = scales
        
        def ch(channels):  # helper for channels
            return min(int(channels * width), max_channels)
            # return max_channels

        self.attention_down = Conv(ch(256),ch(512),3,2)
        self.attention_heatmap = nn.Conv2d(ch(512),1,kernel_size=1)
        self.branch_scale = nn.Parameter(torch.tensor(0.5))
    
        # Backbone
        self.backbone = nn.ModuleList([
            Conv(3, ch(64), 3, 2),  # 0-P1/2
            Conv(ch(64), ch(128), 3, 2),  # 1-P2/4
            C2f(ch(128), ch(128), n=int(3 * depth), shortcut=True),  # 2
            Conv(ch(128), ch(256), 3, 2),  # 3-P3/8
            C2f(ch(256), ch(256), n=int(6 * depth), shortcut=True),  # 4
            Conv(ch(256), ch(512), 3, 2),  # 5-P4/16
            C2f(ch(512), ch(512), n=int(6 * depth), shortcut=True),  # 6
            Conv(ch(512), ch(1024), 3, 2),  # 7-P5/32
            C2f(ch(1024), ch(1024), n=int(3 * depth), shortcut=True),  # 8
            SPPF(ch(1024), ch(1024), 5),  # 9
        ])
        
        # Head
        self.head = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),  # 10
            Concat(dimension=1),  # 11
            C2f(ch(1024 + 512), ch(512), n=int(3 * depth), shortcut=False),  # 12
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # 13
            Concat(dimension=1),  # 14
            C2f(ch(512 + 256), ch(256), n=int(3 * depth), shortcut=False),  # 15 (P3/8-small)
            
            Conv(ch(256), ch(256), 3, 2),  # 16
            Concat(dimension=1),  # 17
            C2f(ch(256 + 512), ch(512), n=int(3 * depth), shortcut=False),  # 18 (P4/16-medium)
            
            Conv(ch(512), ch(512), 3, 2),  # 19
            Concat(dimension=1),  # 20
            C2f(ch(512 + 1024), ch(1024), n=int(3 * depth), shortcut=False),  # 21 (P5/32-large)
        ])
        
        # Detect head
        self.detect = Detect(nc, (ch(256), ch(512), ch(1024)))
        # 初始化stride和bias
        self.stride = torch.tensor([8., 16., 32.])  # 根据实际特征图下采样率设置
        self.detect.stride = self.stride #设置采样率
        self.detect.bias_init()  # 初始化检测头的偏置

    def forward(self, x, return_loss=False):
        """Forward pass through YOLOv8 model."""
        # -------------------- backbone --------------------
        backbone_features = []
        for i, module in enumerate(self.backbone):
            x = module(x)

            if i == 4 :
                # 从中间单独拿出一部分进行采样
                attention_feature = self.attention_down(x)
                branch = self.attention_heatmap(attention_feature)

            elif i == 5 :
                # 此处为p4,尺寸为1/16
                att_mask = torch.sigmoid(branch)
                x = x * (1.0 + self.branch_scale * att_mask)

            if i in [4, 6, 9]:  # 保存P3, P4, P5
                backbone_features.append(x)
        p3, p4, p5 = backbone_features  # 更直观命名

        # -------------------- head --------------------
        # P5 -> P4
        x = self.head[0](p5)  # upsample
        x = self.head[1]([x, p4])  # concat with backbone P4
        p4_out = self.head[2](x)   # C2f

        # P4 -> P3
        x = self.head[3](p4_out)  # upsample
        x = self.head[4]([x, p3]) # concat with backbone P3
        p3_out = self.head[5](x)  # C2f (P3/8-small)

        # P3 -> P4 (下采样)
        x = self.head[6](p3_out)      # conv
        x = self.head[7]([x, p4_out]) # concat with head P4
        p4_out = self.head[8](x)     # C2f (P4/16-medium)

        # P4 -> P5 (下采样)
        x = self.head[9](p4_out)     # conv
        x = self.head[10]([x, p5])    # concat with backbone P5
        p5_out = self.head[11](x)     # C2f (P5/32-large)

        # -------------------- detect --------------------
        return self.detect([p3_out, p4_out, p5_out], return_loss=return_loss),branch


class Net_learn_branch(nn.Module):
    def __init__(self, nc=1, scales=(0.33, 0.25, 1024)):
        super().__init__()
        depth, width, max_channels = scales
        
        def ch(channels):  # helper for channels
            return min(int(channels * width), max_channels)
            # return max_channels
        
        # Attension
        #主干提取部分，用来特征提取
        self.attention = nn.Sequential(
            Conv(3,ch(32),3,2), #1/2
            Conv(ch(32),ch(64),3,2), #1/4
            Conv(ch(64),ch(128),3,2), #1/8
            Conv(ch(128),ch(256),3,2), #1/16  
        )
        self.attention_heatmap = nn.Conv2d(ch(256), 1, 1)

        # Backbone
        self.backbone = nn.ModuleList([
            Conv(3, ch(64), 3, 2),  # 0-P1/2
            Conv(ch(64), ch(128), 3, 2),  # 1-P2/4
            C2f(ch(128), ch(128), n=int(3 * depth), shortcut=True),  # 2
            Conv(ch(128), ch(256), 3, 2),  # 3-P3/8
            C2f(ch(256), ch(256), n=int(6 * depth), shortcut=True),  # 4
            Conv(ch(256), ch(512), 3, 2),  # 5-P4/16
            C2f(ch(512), ch(512), n=int(6 * depth), shortcut=True),  # 6
            Conv(ch(512), ch(1024), 3, 2),  # 7-P5/32
            C2f(ch(1024), ch(1024), n=int(3 * depth), shortcut=True),  # 8
            SPPF(ch(1024), ch(1024), 5),  # 9
        ])
        
        # Head
        self.head = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),  # 10
            Concat(dimension=1),  # 11
            C2f(ch(1024 + 512), ch(512), n=int(3 * depth), shortcut=False),  # 12
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # 13
            Concat(dimension=1),  # 14
            C2f(ch(512 + 256), ch(256), n=int(3 * depth), shortcut=False),  # 15 (P3/8-small)
            
            Conv(ch(256), ch(256), 3, 2),  # 16
            Concat(dimension=1),  # 17
            C2f(ch(256 + 512), ch(512), n=int(3 * depth), shortcut=False),  # 18 (P4/16-medium)
            
            Conv(ch(512), ch(512), 3, 2),  # 19
            Concat(dimension=1),  # 20
            C2f(ch(512 + 1024), ch(1024), n=int(3 * depth), shortcut=False),  # 21 (P5/32-large)
        ])
        
        # Detect head
        self.detect = Detect(nc, (ch(256), ch(512), ch(1024)))
        # 初始化stride和bias
        self.stride = torch.tensor([8., 16., 32.])  # 根据实际特征图下采样率设置
        self.detect.stride = self.stride #设置采样率
        self.detect.bias_init()  # 初始化检测头的偏置
        self.right_proj = Conv(256, ch(512), 1, 1)
        self.branch_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x,return_loss=False):
        """Forward pass through YOLOv8 model."""
        # -------------------- backbone --------------------
        right = self.attention(x)
        branch = self.attention_heatmap(right)
        backbone_features = []
        for i, module in enumerate(self.backbone):
            x = module(x)
            if i in [4, 6, 9]:  # 保存P3, P4, P5
                backbone_features.append(x)
            
            # 将上下文特征注入主干，分支权重先固定
            if i == 5:
                # 将 1 通道的热力图过 sigmoid 并缩放到与 x 尺寸一致
                att_mask = torch.sigmoid(branch) 
                # 使用残差乘法：不仅保留原特征，还在高响应区域加强
                x = x * (1.0 + self.branch_scale * att_mask)
            
        p3, p4, p5 = backbone_features  

        # -------------------- head --------------------
        # P5 -> P4
        x = self.head[0](p5)  # upsample
        x = self.head[1]([x, p4])  # concat with backbone P4
        p4_out = self.head[2](x)   # C2f

        # P4 -> P3
        x = self.head[3](p4_out)  # upsample
        x = self.head[4]([x, p3]) # concat with backbone P3
        p3_out = self.head[5](x)  # C2f (P3/8-small)

        # P3 -> P4 (下采样)
        x = self.head[6](p3_out)      # conv
        x = self.head[7]([x, p4_out]) # concat with head P4
        p4_out = self.head[8](x)     # C2f (P4/16-medium)

        # P4 -> P5 (下采样)
        x = self.head[9](p4_out)     # conv
        x = self.head[10]([x, p5])    # concat with backbone P5
        p5_out = self.head[11](x)     # C2f (P5/32-large)

        # -------------------- detect --------------------
        return self.detect([p3_out, p4_out, p5_out],return_loss=return_loss),branch


class Net_learn_branch_film(nn.Module):
    def __init__(self, nc=1, scales=(0.33, 0.25, 1024)):
        super().__init__()
        depth, width, max_channels = scales
        
        def ch(channels):  
            return min(int(channels * width), max_channels)
        
        # 旁路特征提取
        self.attention = nn.Sequential(
            Conv(3, ch(32), 3, 2),   # 1/2
            Conv(ch(32), ch(64), 3, 2),  # 1/4
            Conv(ch(64), ch(128), 3, 2), # 1/8
            Conv(ch(128), ch(256), 3, 2),# 1/16  
        )
        self.attention_heatmap = nn.Sequential(
            Conv(ch(256), ch(128), 3, 1),
            nn.Conv2d(ch(128), 1, kernel_size=1, stride=1, padding=0),
        )
        
        # ---------------------------------------------------------
        # 目标层 i=5 时的通道数
        target_channels = ch(512)
        
        # 生成 gamma 和 beta，输出通道数是 target_channels 的 2 倍
        self.film_generator = nn.Sequential(

            Conv(ch(256), ch(256), 3, 1), 
            # 最后用 1x1 卷积输出双倍通道，不加激活函数
            nn.Conv2d(ch(256), target_channels * 2, kernel_size=1, stride=1, padding=0)
        )
        
        # 初始化保护因子
        self.film_scale = nn.Parameter(torch.zeros(1))
        # ---------------------------------------------------------

        # Backbone 
        self.backbone = nn.ModuleList([
            Conv(3, ch(64), 3, 2),  # 0
            Conv(ch(64), ch(128), 3, 2),  # 1
            C2f(ch(128), ch(128), n=int(3 * depth), shortcut=True),  # 2
            Conv(ch(128), ch(256), 3, 2),  # 3
            C2f(ch(256), ch(256), n=int(6 * depth), shortcut=True),  # 4
            Conv(ch(256), ch(512), 3, 2),  # 5-P4/16
            C2f(ch(512), ch(512), n=int(6 * depth), shortcut=True),  # 6
            Conv(ch(512), ch(1024), 3, 2),  # 7
            C2f(ch(1024), ch(1024), n=int(3 * depth), shortcut=True),  # 8
            SPPF(ch(1024), ch(1024), 5),  # 9
        ])
        
        # Head & Detect 
        self.head = nn.ModuleList([
        nn.Upsample(scale_factor=2, mode='nearest'),  # 10
        Concat(dimension=1),  # 11
        C2f(ch(1024 + 512), ch(512), n=int(3 * depth), shortcut=False),  # 12
        
        nn.Upsample(scale_factor=2, mode='nearest'),  # 13
        Concat(dimension=1),  # 14
        C2f(ch(512 + 256), ch(256), n=int(3 * depth), shortcut=False),  # 15 (P3/8-small)
        
        Conv(ch(256), ch(256), 3, 2),  # 16
        Concat(dimension=1),  # 17
        C2f(ch(256 + 512), ch(512), n=int(3 * depth), shortcut=False),  # 18 (P4/16-medium)
        
        Conv(ch(512), ch(512), 3, 2),  # 19
        Concat(dimension=1),  # 20
        C2f(ch(512 + 1024), ch(1024), n=int(3 * depth), shortcut=False),  # 21 (P5/32-large)
    ])
    
        # Detect head
        self.detect = Detect(nc, (ch(256), ch(512), ch(1024)))
        # 初始化stride和bias
        self.stride = torch.tensor([8., 16., 32.])  # 根据实际特征图下采样率设置
        self.detect.stride = self.stride #设置采样率
        self.detect.bias_init()  # 初始化检测头的偏置
        self.right_proj = Conv(256, ch(512), 1, 1)
        self.branch_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_loss=False):
        # -------------------- 旁路处理 --------------------
        right = self.attention(x)
        att_logits = self.attention_heatmap(right)
        
        # 生成 FiLM 混合特征图 [B, ch(512)*2, H/16, W/16]
        film_params = self.film_generator(right) 
        
        #  沿通道维度(dim=1)切分为两半，分别得到 gamma_raw 和 beta_raw
        # [B, ch(512), H/16, W/16]
        gamma_raw, beta_raw = torch.chunk(film_params, 2, dim=1)
        
        # 3. 构造安全的缩放与平移参数 (依赖 self.film_scale)
        # 训练初期 scale 为 0，所以 gamma 恒为 1，beta 恒为 0
        gamma = 1.0 + self.film_scale * gamma_raw
        beta = self.film_scale * beta_raw

        # -------------------- 主干网络 --------------------
        backbone_features = []
        for i, module in enumerate(self.backbone):
            x = module(x)
            if i in [4, 6, 9]:
                backbone_features.append(x)
            
            # 在 i=5 时注入 FiLM 调制
            if i == 5:
                # 仿射变换计算: out = gamma * x + beta
                x = gamma * x + beta
            
        p3, p4, p5 = backbone_features  

        # -------------------- 颈部与检测头 --------------------
        # P5 -> P4
        x = self.head[0](p5)  # upsample
        x = self.head[1]([x, p4])  # concat with backbone P4
        p4_out = self.head[2](x)   # C2f

        # P4 -> P3
        x = self.head[3](p4_out)  # upsample
        x = self.head[4]([x, p3]) # concat with backbone P3
        p3_out = self.head[5](x)  # C2f (P3/8-small)

        # P3 -> P4 (下采样)
        x = self.head[6](p3_out)      # conv
        x = self.head[7]([x, p4_out]) # concat with head P4
        p4_out = self.head[8](x)     # C2f (P4/16-medium)

        # P4 -> P5 (下采样)
        x = self.head[9](p4_out)     # conv
        x = self.head[10]([x, p5])    # concat with backbone P5
        p5_out = self.head[11](x)     # C2f (P5/32-large)

        # -------------------- detect --------------------
        
        return self.detect([p3_out, p4_out, p5_out], return_loss=return_loss), att_logits

class self_attention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        hidden_channels = max(channels // 4, 16)

        self.head = nn.Sequential(
            # 深度卷积：提取空间结构，计算量较小
            nn.Conv2d(channels,channels,kernel_size=3,stride=2,padding=autopad(3, None, 1),groups=channels,bias=False,),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),

            # 压缩通道
            nn.Conv2d(channels,hidden_channels,kernel_size=1,bias=False,),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),

            # 输出单通道注意力 logits
            nn.Conv2d(hidden_channels,1,kernel_size=1,bias=True,),)

    def forward(self, x):
        return self.head(x)

class self_res(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,padding=1,groups=channels,bias=False,),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels,channels,kernel_size=1,bias=False,),
            nn.BatchNorm2d(channels),
        )

        # 初始为0，开始训练时等价于原始网络
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, att_mask):
        gated_feature = x * att_mask
        enhanced_feature = self.refine(gated_feature)

        return x + torch.tanh(self.alpha) * enhanced_feature

class YOLOv8_self_res(nn.Module):
    """YOLOv8 object detection model."""
    def __init__(self, nc=80, scales=(0.33, 0.25, 1024)):
        super().__init__()
        depth, width, max_channels = scales
        
        def ch(channels):  # helper for channels
            return min(int(channels * width), max_channels)
            # return max_channels
        
        # Backbone
        self.backbone = nn.ModuleList([
            Conv(3, ch(64), 3, 2),  # 0-P1/2
            Conv(ch(64), ch(128), 3, 2),  # 1-P2/4
            C2f(ch(128), ch(128), n=int(3 * depth), shortcut=True),  # 2
            Conv(ch(128), ch(256), 3, 2),  # 3-P3/8
            C2f(ch(256), ch(256), n=int(6 * depth), shortcut=True),  # 4
            Conv(ch(256), ch(512), 3, 2),  # 5-P4/16
            C2f(ch(512), ch(512), n=int(6 * depth), shortcut=True),  # 6
            Conv(ch(512), ch(1024), 3, 2),  # 7-P5/32
            C2f(ch(1024), ch(1024), n=int(3 * depth), shortcut=True),  # 8
            SPPF(ch(1024), ch(1024), 5),  # 9
        ])
        self.attention = self_attention(ch(256))
        self.res = self_res(ch(512))
        
        # Head
        self.head = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),  # 10
            Concat(dimension=1),  # 11
            C2f(ch(1024 + 512), ch(512), n=int(3 * depth), shortcut=False),  # 12
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # 13
            Concat(dimension=1),  # 14
            C2f(ch(512 + 256), ch(256), n=int(3 * depth), shortcut=False),  # 15 (P3/8-small)
            
            Conv(ch(256), ch(256), 3, 2),  # 16
            Concat(dimension=1),  # 17
            C2f(ch(256 + 512), ch(512), n=int(3 * depth), shortcut=False),  # 18 (P4/16-medium)
            
            Conv(ch(512), ch(512), 3, 2),  # 19
            Concat(dimension=1),  # 20
            C2f(ch(512 + 1024), ch(1024), n=int(3 * depth), shortcut=False),  # 21 (P5/32-large)
        ])
        
        # Detect head
        self.detect = Detect(nc, (ch(256), ch(512), ch(1024)))
        # 初始化stride和bias
        self.stride = torch.tensor([8., 16., 32.])  # 根据实际特征图下采样率设置
        self.detect.stride = self.stride #设置采样率
        self.detect.bias_init()  # 初始化检测头的偏置

    def forward(self, x, return_loss=False):
        """Forward pass through YOLOv8 model."""
        # -------------------- backbone --------------------
        backbone_features = []
        for i, module in enumerate(self.backbone):
            x = module(x)
            
            if i == 4:
                branch = self.attention(x)
            
            if i == 5:
                x = self.res(x,branch)

            if i in [4, 6, 9]:  # 保存P3, P4, P5
                backbone_features.append(x)
        p3, p4, p5 = backbone_features  # 更直观命名

        # -------------------- head --------------------
        # P5 -> P4
        x = self.head[0](p5)  # upsample
        x = self.head[1]([x, p4])  # concat with backbone P4
        p4_out = self.head[2](x)   # C2f

        # P4 -> P3
        x = self.head[3](p4_out)  # upsample
        x = self.head[4]([x, p3]) # concat with backbone P3
        p3_out = self.head[5](x)  # C2f (P3/8-small)

        # P3 -> P4 (下采样)
        x = self.head[6](p3_out)      # conv
        x = self.head[7]([x, p4_out]) # concat with head P4
        p4_out = self.head[8](x)     # C2f (P4/16-medium)

        # P4 -> P5 (下采样)
        x = self.head[9](p4_out)     # conv
        x = self.head[10]([x, p5])    # concat with backbone P5
        p5_out = self.head[11](x)     # C2f (P5/32-large)

        # -------------------- detect --------------------
        return self.detect([p3_out, p4_out, p5_out], return_loss=return_loss),branch
