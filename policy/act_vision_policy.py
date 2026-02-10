import math
from dataclasses import dataclass
from collections import deque
from itertools import chain
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# =========================
# LeRobot 스타일 key 상수
# =========================
OBS_STATE = "observation.state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
ACTION = "action"
ACTION_IS_PAD = "action_is_pad"


# =========================
# Positional embedding 유틸
# =========================
def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    def get_position_angle_vec(position: int):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])  # (1,H,W)
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1,C,H,W)
        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


# =========================
# Config (LeRobot ACTConfig의 핵심 subset)
# =========================
@dataclass
class ACTVisionConfig:
    # IO
    state_dim: int
    action_dim: int
    chunk_size: int
    n_action_steps: int = 1  # LeRobot temporal ensemble 쓰려면 1이어야 함(여기서는 기본 1로)

    # Backbone/Transformer
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1

    # VAE
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Loss
    kl_weight: float = 10.0
    dropout: float = 0.1


# =========================
# Transformer blocks (LeRobot 구조)
# =========================
class ACTEncoder(nn.Module):
    def __init__(self, cfg: ACTVisionConfig, is_vae_encoder: bool = False):
        super().__init__()
        num_layers = cfg.n_vae_encoder_layers if is_vae_encoder else cfg.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(cfg) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(cfg.dim_model) if cfg.pre_norm else nn.Identity()
        self.is_vae_encoder = is_vae_encoder

    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        return self.norm(x)


class ACTEncoderLayer(nn.Module):
    def __init__(self, cfg: ACTVisionConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)

        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)

        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.activation = get_activation_fn(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm

    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)

        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, cfg: ACTVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)])
        self.norm = nn.LayerNorm(cfg.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        return self.norm(x)


class ACTDecoderLayer(nn.Module):
    def __init__(self, cfg: ACTVisionConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)
        self.multihead_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)

        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)

        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)
        self.norm3 = nn.LayerNorm(cfg.dim_model)

        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)

        self.activation = get_activation_fn(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm

    @staticmethod
    def maybe_add_pos_embed(t: Tensor, pos: Tensor | None) -> Tensor:
        return t if pos is None else t + pos

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)

        if not self.pre_norm:
            x = self.norm3(x)
        return x


# =========================
# ACT 본체 (LeRobot 구조)
# =========================
class LeRobotStyleACT(nn.Module):
    def __init__(self, cfg: ACTVisionConfig):
        super().__init__()
        self.cfg = cfg

        # ----- VAE encoder -----
        if cfg.use_vae:
            self.vae_encoder = ACTEncoder(cfg, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, cfg.dim_model)
            self.vae_encoder_robot_state_input_proj = nn.Linear(cfg.state_dim, cfg.dim_model)
            self.vae_encoder_action_input_proj = nn.Linear(cfg.action_dim, cfg.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(cfg.dim_model, cfg.latent_dim * 2)

            num_input_token_encoder = 2 + cfg.chunk_size  # [cls, state, actions...]
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, cfg.dim_model).unsqueeze(0),
            )

        # ----- Vision backbone (ResNet layer4 feature map) -----
        backbone_model = getattr(torchvision.models, cfg.vision_backbone)(
            replace_stride_with_dilation=[False, False, cfg.replace_final_stride_with_dilation],
            weights=cfg.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # ----- Main transformer -----
        self.encoder = ACTEncoder(cfg)
        self.decoder = ACTDecoder(cfg)

        self.encoder_latent_input_proj = nn.Linear(cfg.latent_dim, cfg.dim_model)
        self.encoder_robot_state_input_proj = nn.Linear(cfg.state_dim, cfg.dim_model)
        self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, cfg.dim_model, kernel_size=1)

        # Positional embeddings
        self.encoder_1d_feature_pos_embed = nn.Embedding(2, cfg.dim_model)  # [latent, state]
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(cfg.dim_model // 2)

        # Decoder queries
        self.decoder_pos_embed = nn.Embedding(cfg.chunk_size, cfg.dim_model)

        # Action head
        self.action_head = nn.Linear(cfg.dim_model, cfg.action_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _reparameterize(mu: Tensor, log_sigma_x2: Tensor) -> Tensor:
        return mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """
        batch keys:
          - observation.image: (B,3,H,W)  OR
          - observation.images: list/tuple of (B,3,H,W) (여러 카메라면 첫 번째 사용)
          - observation.state: (B,state_dim)
          - action: (B,chunk,action_dim)  (train 시)
          - action_is_pad: (B,chunk)      (옵션)
        """
        # 이미지 추출
        if OBS_IMAGES in batch:
            img = batch[OBS_IMAGES][0]
        else:
            img = batch[OBS_IMAGE]
        state = batch[OBS_STATE]
        B = img.shape[0]

        # ----- latent 준비 -----
        if self.cfg.use_vae and self.training and (ACTION in batch):
            action_chunk = batch[ACTION]  # (B,S,A)

            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=B)
            state_embed = self.vae_encoder_robot_state_input_proj(state).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(action_chunk)

            vae_in = torch.cat([cls_embed, state_embed, action_embed], dim=1)  # (B,S+2,D)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()              # (1,S+2,D)

            if ACTION_IS_PAD in batch:
                action_is_pad = batch[ACTION_IS_PAD].to(torch.bool)
            else:
                action_is_pad = torch.zeros((B, self.cfg.chunk_size), dtype=torch.bool, device=img.device)

            cls_state_is_pad = torch.zeros((B, 2), dtype=torch.bool, device=img.device)
            key_padding_mask = torch.cat([cls_state_is_pad, action_is_pad], dim=1)

            cls_token_out = self.vae_encoder(
                vae_in.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # (B,D)

            latent_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_params[:, : self.cfg.latent_dim]
            log_sigma_x2 = latent_params[:, self.cfg.latent_dim :]
            latent_sample = self._reparameterize(mu, log_sigma_x2)
        else:
            mu = None
            log_sigma_x2 = None
            latent_sample = torch.zeros((B, self.cfg.latent_dim), dtype=torch.float32, device=img.device)

        # ----- encoder tokens: [latent, state, img_pixels...] -----
        encoder_in_tokens = [
            self.encoder_latent_input_proj(latent_sample),           # (B,D)
            self.encoder_robot_state_input_proj(state),              # (B,D)
        ]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))  # (2,1,D)

        cam_features = self.backbone(img)["feature_map"]  # (B,C',H',W')
        cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
        cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B,D,H',W')

        cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")  # (HW,B,D)
        cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")  # (HW,B,D)

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)  # (2,B,D)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)  # (2,1,D)

        encoder_in_tokens = torch.cat([encoder_in_tokens, cam_features], dim=0)
        encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, cam_pos_embed], dim=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # ----- decoder queries -> actions -----
        decoder_in = torch.zeros(
            (self.cfg.chunk_size, B, self.cfg.dim_model),
            dtype=encoder_in_tokens.dtype,
            device=encoder_in_tokens.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)  # (B,S,D)
        actions = self.action_head(decoder_out)    # (B,S,A)

        return actions, (mu, log_sigma_x2)


# =========================
# LeRobot ACTPolicy와 동일한 "형태"의 Policy
# =========================
class ACTVisionPolicy(nn.Module):
    """
    LeRobot의 ACTPolicy와 동일 인터페이스:
      - forward(batch) -> (loss, loss_dict)
      - predict_action_chunk(batch) -> actions
      - select_action(batch) -> action (queue 사용)
      - reset()
    """

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, n_action_steps: int = 1, d_model: int = 512):
        super().__init__()
        self.cfg = ACTVisionConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            dim_model=d_model,
        )
        self.model = LeRobotStyleACT(self.cfg)
        self.reset()

    def reset(self):
        self._action_queue = deque([], maxlen=self.cfg.n_action_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        actions, _ = self.model(batch)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        LeRobot ACTPolicy.select_action 형태.
        n_action_steps > 1이면 queue 사용.
        """
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.cfg.n_action_steps]  # (B,n,A)
            self._action_queue.extend(actions.transpose(0, 1))  # (n,B,A)
        return self._action_queue.popleft()  # (B,A)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        LeRobot ACTPolicy.forward 형태:
          - actions_hat, (mu, log_sigma_x2) = self.model(batch)
          - l1_loss (+ pad mask)
          - kld_loss (use_vae일 때)
          - return loss, loss_dict
        """
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        if ACTION not in batch:
            raise ValueError("training forward(batch)에는 'action'이 포함되어야 합니다.")

        if ACTION_IS_PAD in batch:
            action_is_pad = batch[ACTION_IS_PAD].to(torch.bool)
        else:
            action_is_pad = torch.zeros(
                (actions_hat.shape[0], actions_hat.shape[1]),
                dtype=torch.bool,
                device=actions_hat.device,
            )

        l1 = F.l1_loss(batch[ACTION], actions_hat, reduction="none")  # (B,S,A)
        l1_loss = (l1 * (~action_is_pad).unsqueeze(-1)).mean()

        loss_dict = {"l1_loss": float(l1_loss.item())}

        if self.cfg.use_vae and (mu_hat is not None) and (log_sigma_x2_hat is not None):
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = float(mean_kld.item())
            loss = l1_loss + mean_kld * self.cfg.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


# 관절 키 (기존 pickle 포맷 기준)
JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# =========================
class ACTVisionDatasetFromPickle(Dataset):
    """
    pickle 파일 포맷:
      obj["observations"][t]["camera"]  -> HxWx3 uint8
      obj["observations"][t][JOINT_KEYS...] -> float
      obj["actions"][t][JOINT_KEYS...] -> float
    반환:
      img_t: (3, 128, 128) float, ImageNet normalize 적용
      s_seq: (seq_len, state_dim) [-1,1] min-max 정규화
      a_chunk: (chunk_size, action_dim) [-1,1] min-max 정규화
    """

    def __init__(self, pkl_files, seq_len, chunk_size, is_train=True, stats=None):
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.is_train = is_train

        self.obs_trajs = []
        self.act_trajs = []
        self.index = []

        # 이미지 augmentation/normalize
        self.augmentor = (
            transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            if is_train
            else transforms.Compose(
                [
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        )

        # load trajectories + build indices
        for traj_id, fp in enumerate(pkl_files):
            with open(fp, "rb") as f:
                try:
                    obj = pickle.load(f)
                except UnicodeDecodeError:
                    obj = pickle.load(f, encoding="latin1")
            self.obs_trajs.append(obj["observations"])
            self.act_trajs.append(obj["actions"])

            # t는 "현재 시점" 인덱스. state_seq는 [t-seq_len+1 ... t], action_chunk는 [t ... t+chunk_size-1]
            for t in range(seq_len - 1, len(obj["observations"]) - chunk_size + 1):
                self.index.append((traj_id, t))

        # stats 계산 (state 기준 min/max)
        if stats is None:
            all_states = [
                [float(obs[k]) for k in JOINT_KEYS]
                for traj in self.obs_trajs
                for obs in traj
            ]
            all_states = np.array(all_states)
            self.stats = {
                "min": torch.tensor(all_states.min(axis=0), dtype=torch.float32),
                "max": torch.tensor(all_states.max(axis=0), dtype=torch.float32),
            }
        else:
            self.stats = stats

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # [-1,1] min-max
        return 2 * (x - self.stats["min"]) / (self.stats["max"] - self.stats["min"] + 1e-5) - 1

    def __getitem__(self, i):
        traj_id, t = self.index[i]
        obs_list = self.obs_trajs[traj_id]
        act_list = self.act_trajs[traj_id]

        # image
        img = np.asarray(obs_list[t]["camera"])
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(
            img_t.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        img_t = self.augmentor(img_t)

        # state sequence (seq_len, state_dim)
        s_seq = torch.stack(
            [
                self._norm(torch.tensor([float(obs_list[tt][k]) for k in JOINT_KEYS], dtype=torch.float32))
                for tt in range(t - self.seq_len + 1, t + 1)
            ]
        )

        # action chunk (chunk_size, action_dim)
        a_chunk = torch.stack(
            [
                self._norm(torch.tensor([float(act_list[tt][k]) for k in JOINT_KEYS], dtype=torch.float32))
                for tt in range(t, t + self.chunk_size)
            ]
        )

        return img_t, s_seq, a_chunk

    def __len__(self):
        return len(self.index)
# =========================


# =========================
# (중요) Dataset (img, state_seq, action_chunk) -> LeRobot batch dict 변환 함수
# =========================
def to_lerobot_batch(img: Tensor, state_seq: Tensor, action_chunk: Tensor, device: torch.device, chunk_size: int) -> dict:
    """
    기존 pickle dataset 출력 형태를 LeRobot 학습 예제 batch 형태로 변환.
      - state_seq: (B,seq_len,state_dim) 이면 마지막 스텝만 사용(LeRobot ACT n_obs_steps=1)
    """
    img = img.to(device, non_blocking=True)
    state_seq = state_seq.to(device, non_blocking=True)
    action_chunk = action_chunk.to(device, non_blocking=True)

    state = state_seq[:, -1, :] if state_seq.dim() == 3 else state_seq
    action_is_pad = torch.zeros((img.shape[0], chunk_size), dtype=torch.bool, device=device)

    return {
        OBS_IMAGE: img,
        OBS_STATE: state,
        ACTION: action_chunk,
        ACTION_IS_PAD: action_is_pad,
    }