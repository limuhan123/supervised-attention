from .model_tool import *


class YOLOv8_small(nn.Module):
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
        return self.detect([p3_out, p4_out], return_loss=return_loss)


class YOLOv8(nn.Module):
    """YOLOv8 object detection model."""
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
            Conv(ch(512), ch(1024), 3, 2),  # 7-P5/32
            C2f(ch(1024), ch(1024), n=dep(3 * depth), shortcut=True),  # 8
            SPPF(ch(1024), ch(1024), 5),  # 9
        ])
        
        # Head
        self.head = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),  # 10
            Concat(dimension=1),  # 11
            C2f(ch(1024 + 512), ch(512), n=dep(3 * depth), shortcut=False),  # 12
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # 13
            Concat(dimension=1),  # 14
            C2f(ch(512 + 256), ch(256), n=dep(3 * depth), shortcut=False),  # 15 (P3/8-small)
            
            Conv(ch(256), ch(256), 3, 2),  # 16
            Concat(dimension=1),  # 17
            C2f(ch(256 + 512), ch(512), n=dep(3 * depth), shortcut=False),  # 18 (P4/16-medium)
            
            Conv(ch(512), ch(512), 3, 2),  # 19
            Concat(dimension=1),  # 20
            C2f(ch(512 + 1024), ch(1024), n=dep(3 * depth), shortcut=False),  # 21 (P5/32-large)
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
        return self.detect([p3_out, p4_out, p5_out], return_loss=return_loss)

class YOLOv8_cbam(nn.Module):
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
        self.cbam = CBAM(channels=ch(512),reduction=16,spatial_kernel_size=7,)
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

            # backbone[5] 输出初始 P4/16 特征
            # 与你的场坪监督注意力处于相同位置
            if i == 5:
                x = self.cbam(x)

            if i in [4, 6, 9]:
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
        return self.detect([p3_out, p4_out, p5_out], return_loss=return_loss)

class YOLOv8_se(nn.Module):
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

        self.se = SEBlock(channels=ch(512),reduction=16,)
        
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
            # backbone[5] 输出 P4/16 初始特征
            if i == 5:
                x = self.se(x)
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
        return self.detect([p3_out, p4_out, p5_out], return_loss=return_loss)


if __name__ == "__main__":
    x = torch.rand(1,3,640,640)
    model = YOLOv8()
    model = model.eval()
    result = model(x)
    for i in result:
        print(i.shape)