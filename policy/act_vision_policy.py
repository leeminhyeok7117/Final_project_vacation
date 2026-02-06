# import torch
# import torch.nn as nn
# import numpy as np
# from torchvision.models import resnet18, ResNet18_Weights

# # =============================================================================
# # [최종 수정] 체크포인트 구조(3 Layers, Linear Head, PE 2048) 완벽 일치 버전
# # =============================================================================

# class ACTVisionPolicy(nn.Module):
#     def __init__(
#         self,
#         state_dim=6,
#         action_dim=6,
#         seq_len=8,
#         chunk_size=4,
#         dim_feedforward=512,  
#         backbone="simple",    
#     ):
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.seq_len = seq_len
#         self.chunk_size = chunk_size
        
#         # 1. 비전 백본 설정
#         if backbone == "resnet18":
#             self.vision = ResNetBackbone() 
#             feature_dim = 512 
#         else:
#             self.vision = SimpleCNN()
#             feature_dim = 128

#         # 2. 트랜스포머 설정
#         hidden_dim = 256
#         nhead = 4
        
#         # [수정 1] num_layers를 4에서 3으로 변경 (에러로그: layers.3 없음 -> 0,1,2만 존재)
#         num_layers = 3 
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim, 
#             nhead=nhead, 
#             dim_feedforward=dim_feedforward,
#             dropout=0.1, 
#             batch_first=True,
#             norm_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # 3. 프로젝션
#         self.vision.proj = nn.Linear(feature_dim, hidden_dim)
#         self.state_proj = nn.Linear(state_dim, hidden_dim)
        
#         # [수정 2] Positional Encoding max_len을 2048로 변경 (에러로그: shape mismatch 2048 vs 9)
#         self.pos_enc = PositionalEncoding(hidden_dim, max_len=2048)
        
#         # [수정 3] Head를 MLP(Sequential)에서 단순 Linear로 변경 (에러로그: head.weight 존재, head.0 없음)
#         self.head = nn.Linear(hidden_dim, action_dim * chunk_size)

#     def forward(self, img, state_seq):
#         B = img.shape[0]
#         vision_feat = self.vision(img)
#         img_emb = self.vision.proj(vision_feat).unsqueeze(1)
#         state_emb = self.state_proj(state_seq)
        
#         token_seq = torch.cat([img_emb, state_emb], dim=1)
#         token_seq = self.pos_enc(token_seq)
#         out_seq = self.encoder(token_seq)
        
#         cls_out = out_seq[:, 0, :]
        
#         # 단순 Linear 통과
#         pred_flat = self.head(cls_out)
#         return pred_flat.view(B, self.chunk_size, self.action_dim)


# class ResNetBackbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#         self.backbone = nn.Sequential(
#             resnet.conv1,
#             resnet.bn1,
#             resnet.relu,
#             resnet.maxpool,
#             resnet.layer1,
#             resnet.layer2,
#             resnet.layer3,
#             resnet.layer4,
#             resnet.avgpool
#         )

#     def forward(self, x):
#         x = self.backbone(x)
#         return x.flatten(1)

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#     def forward(self, x):
#         return self.net(x).flatten(1)

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=50):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         # 입력 길이만큼만 잘라서 사용
#         return x + self.pe[:, :x.size(1)]

########cvae######################################################################################
# 현재 방식
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models

# 관절 키 (기존 설정 유지)
JOINT_KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ACTVisionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len, chunk_size, latent_dim=32, d_model=256):
        super().__init__()
        self.chunk_size, self.action_dim, self.latent_dim = chunk_size, action_dim, latent_dim

        # 1. 멀티모달 인코더 (ResNet-18 백본)
        resnet = models.resnet18(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_proj = nn.Linear(512, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        
        # 2. CVAE (Style Encoder)
        self.style_encoder = nn.Linear(action_dim * chunk_size, d_model)
        self.latent_head = nn.Linear(d_model, latent_dim * 2) 
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # 3. Transformer
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.action_head = nn.Linear(d_model, chunk_size * action_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img, state_seq, action_chunk=None):
        batch_size = img.size(0)
        v_feat = self.vision_proj(self.backbone(img).flatten(1)).unsqueeze(1) 
        s_feat = self.state_proj(state_seq) 

        if action_chunk is not None:
            style_feat = torch.relu(self.style_encoder(action_chunk.flatten(1)))
            latent_info = self.latent_head(style_feat)
            mu, logvar = latent_info[:, :self.latent_dim], latent_info[:, self.latent_dim:]
            z = self.reparameterize(mu, logvar)
        else:
            mu = logvar = None
            z = torch.zeros((batch_size, self.latent_dim), device=img.device)

        z_feat = self.latent_proj(z).unsqueeze(1)
        x = torch.cat([z_feat, v_feat, s_feat], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        out = self.action_head(x[:, 0, :])
        return out.view(-1, self.chunk_size, self.action_dim), mu, logvar

# 에러가 발생한 클래스 이름을 정확히 선언해야 합니다.
class ACTVisionDatasetFromPickle(Dataset):
    def __init__(self, pkl_files, seq_len, chunk_size, is_train=True, stats=None):
        self.seq_len, self.chunk_size, self.is_train = seq_len, chunk_size, is_train
        self.obs_trajs, self.act_trajs, self.index = [], [], []
        
        # LeRobot 공식 사양과 유사한 증강
        self.augmentor = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if is_train else transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for traj_id, fp in enumerate(pkl_files):
            with open(fp, "rb") as f:
                obj = pickle.load(f)
            self.obs_trajs.append(obj["observations"])
            self.act_trajs.append(obj["actions"])
            for t in range(seq_len - 1, len(obj["observations"]) - chunk_size + 1):
                self.index.append((traj_id, t))
        
        if stats is None:
            all_states = [ [float(obs[k]) for k in JOINT_KEYS] for traj in self.obs_trajs for obs in traj ]
            all_states = np.array(all_states)
            self.stats = {
                "min": torch.tensor(all_states.min(axis=0), dtype=torch.float32),
                "max": torch.tensor(all_states.max(axis=0), dtype=torch.float32)
            }
        else:
            self.stats = stats

    def _norm(self, x):
        return 2 * (x - self.stats["min"]) / (self.stats["max"] - self.stats["min"] + 1e-5) - 1

    def __getitem__(self, i):
        traj_id, t = self.index[i]
        obs_list, act_list = self.obs_trajs[traj_id], self.act_trajs[traj_id]
        img = np.asarray(obs_list[t]["camera"])
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
        img_t = self.augmentor(img_t)

        s_seq = torch.stack([self._norm(torch.tensor([float(obs_list[tt][k]) for k in JOINT_KEYS])) 
                            for tt in range(t - self.seq_len + 1, t + 1)])
        a_chunk = torch.stack([self._norm(torch.tensor([float(act_list[tt][k]) for k in JOINT_KEYS])) 
                              for tt in range(t, t + self.chunk_size)])
        return img_t, s_seq, a_chunk

    def __len__(self):
        return len(self.index)