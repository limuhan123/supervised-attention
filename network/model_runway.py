import torch.nn as nn 
import torch

#自动padding填充，控制图像结构
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

#单层残差块
class Res_block(nn.Module):
    def __init__(self,in_ch,out_ch,kernal_size=3,stride=1,padding=1):
        super().__init__()

        #一层卷积
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernal_size,stride,autopad(kernal_size,None,1))
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        #二层卷积
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernal_size,1,autopad(kernal_size,None,1))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

        #通道对齐
        self.shortcut = nn.Sequential()
        if stride !=1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,1,stride,autopad(1,None,1)),
                nn.BatchNorm2d(out_ch)
            )
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x):

        #一层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        #加入dropout层
        out = self.dropout(out)

        #二层卷积
        out = self.conv2(out)
        out = self.bn2(out)

        #残差链接
        out = self.relu2(self.shortcut(x) + out)

        return out

#注意力模块,学习通道间的关系
class attension_block(nn.Module):
    def __init__(self,channels,reduction_ratio=8):
        super().__init__()

        #池化后特征图全连接
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #(B,C,W,H) -> (B,C,1,1),平均空间特征
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)  

        self.dropout = nn.Dropout(0.5)  
        self.relu = nn.ReLU(inplace=True)  
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        batch_size, channels, height, width = x.size()  
        out = self.avg_pool(x).view(batch_size, channels)  #(B,C,W,H) -> (B,C,1,1) -> (B,C)
        out = self.fc1(out)  

        out = self.dropout(out)

        out = self.relu(out)  
        out = self.fc2(out)  #得到每个通道的权重值，并将通道权重值还原到每个通道的图像像素中
        out = self.sigmoid(out).view(batch_size, channels, 1, 1)  
        out = out.expand_as(x)  
        out = out * x  #增强活跃的通道，抑制低活跃的通道
        return out     

#简单卷积神经网络
class Conv(nn.Module):
    def __init__(self,in_ch,out_ch,k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(in_ch,out_ch,k,s,autopad(k,p,d),groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

#创建机场跑道判定网络
class Runway_detect(nn.Module):
    def __init__(self,num):
        super().__init__()
    
        #卷积神经网络搭建浅层网络
        self.backbone = nn.Sequential(
            Conv(3,32,3,2),
            Conv(32,64,3,2),
            Conv(64,128,3,2)
        )

        #残差块构建深层网络
        self.res = nn.Sequential(
            Res_block(128,128),
            Res_block(128,128),
            Res_block(128,128),
            Res_block(128,256,stride=2),
            Res_block(256,256),
            Res_block(256,512,stride=2),
            Res_block(512,512)
        )

        #自注意力块
        self.attention = attension_block(512)

        #分类
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512,num)
        )

    def forward(self, x):
        # stem部分
        out = self.backbone(x)
        # 残差块序列部分
        out = self.res(out)
        # 自注意力块部分
        out = self.attention(out)
        # 分类器部分
        out = self.classifier(out)
        return out

if __name__ == "__main__":

    x = torch.randn(1, 3, 320, 320)
    model = Runway_detect(2)
    out = model(x)

    print("输出", out)  