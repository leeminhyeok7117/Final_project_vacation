import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

# =============================================================================
# [최종 수정] 체크포인트 구조(3 Layers, Linear Head, PE 2048) 완벽 일치 버전
# =============================================================================

class ACTVisionPolicy(nn.Module):
    def __init__(
        self,
        state_dim=6,
        action_dim=6,
        seq_len=8,
        chunk_size=4,
        dim_feedforward=512,  
        backbone="simple",    
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        
        # 1. 비전 백본 설정
        if backbone == "resnet18":
            self.vision = ResNetBackbone() 
            feature_dim = 512 
        else:
            self.vision = SimpleCNN()
            feature_dim = 128

        # 2. 트랜스포머 설정
        hidden_dim = 256
        nhead = 4
        
        # [수정 1] num_layers를 4에서 3으로 변경 (에러로그: layers.3 없음 -> 0,1,2만 존재)
        num_layers = 3 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=0.1, 
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 프로젝션
        self.vision.proj = nn.Linear(feature_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        
        # [수정 2] Positional Encoding max_len을 2048로 변경 (에러로그: shape mismatch 2048 vs 9)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=2048)
        
        # [수정 3] Head를 MLP(Sequential)에서 단순 Linear로 변경 (에러로그: head.weight 존재, head.0 없음)
        self.head = nn.Linear(hidden_dim, action_dim * chunk_size)

    def forward(self, img, state_seq):
        B = img.shape[0]
        vision_feat = self.vision(img)
        img_emb = self.vision.proj(vision_feat).unsqueeze(1)
        state_emb = self.state_proj(state_seq)
        
        token_seq = torch.cat([img_emb, state_emb], dim=1)
        token_seq = self.pos_enc(token_seq)
        out_seq = self.encoder(token_seq)
        
        cls_out = out_seq[:, 0, :]
        
        # 단순 Linear 통과
        pred_flat = self.head(cls_out)
        return pred_flat.view(B, self.chunk_size, self.action_dim)


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )

    def forward(self, x):
        x = self.backbone(x)
        return x.flatten(1)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    def forward(self, x):
        return self.net(x).flatten(1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 입력 길이만큼만 잘라서 사용
        return x + self.pe[:, :x.size(1)]