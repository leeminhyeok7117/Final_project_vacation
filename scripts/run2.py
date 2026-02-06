#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from pathlib import Path
from torchvision import transforms, models
import dynamixel_sdk as dxl

# 상위 경로 추가 (robots 모듈용)
sys.path.insert(0, str(Path(__file__).parent.parent))
from robots import SimpleRobot 

# =========================================================
# 1. 설정값 (CVAE 모델에 맞춤)
# =========================================================
MODEL_PATH = "examples/models/act_cvae_epoch_450.pth"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# 모델 하이퍼파라미터 (보내주신 코드 기준)
SEQ_LEN = 8
CHUNK_SIZE = 4
STATE_DIM = 6
ACTION_DIM = 6
LATENT_DIM = 32     # CVAE 잠재 변수 차원
D_MODEL = 256       # 트랜스포머 히든 차원

CONTROL_HZ = 30.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 기어비 설정
GEAR_RATIOS = {1: 2.0, 11: 2.0}

# =========================================================
# 2. 모델 클래스 정의 (보내주신 CVAE 코드 그대로 삽입)
# =========================================================
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
        resnet = models.resnet18(weights=None) # 추론 시엔 가중치 로드 불필요 (ckpt에서 덮어씀)
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
        # 이미지 특징 추출
        v_feat = self.vision_proj(self.backbone(img).flatten(1)).unsqueeze(1) 
        # 관절 상태 특징 추출
        s_feat = self.state_proj(state_seq) 

        # CVAE 로직: Inference 시 action_chunk가 없으면 z=0 (Mean Prior) 사용
        if action_chunk is not None:
            style_feat = torch.relu(self.style_encoder(action_chunk.flatten(1)))
            latent_info = self.latent_head(style_feat)
            mu, logvar = latent_info[:, :self.latent_dim], latent_info[:, self.latent_dim:]
            z = self.reparameterize(mu, logvar)
        else:
            mu = logvar = None
            # ★ Inference: 잠재 변수 z를 0으로 설정 (가장 평균적인 행동 생성)
            z = torch.zeros((batch_size, self.latent_dim), device=img.device)

        z_feat = self.latent_proj(z).unsqueeze(1)
        
        # [z, vision, state] 순서로 결합
        x = torch.cat([z_feat, v_feat, s_feat], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        out = self.action_head(x[:, 0, :])
        return out.view(-1, self.chunk_size, self.action_dim), mu, logvar

# =========================================================
# 3. 추론 클래스
# =========================================================
class ACTInference:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        print(f"[INFO] CVAE 체크포인트 로드 중: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.stats = checkpoint['stats']
        
        # 통계값 처리
        self.use_min_max = "min" in self.stats
        if not self.use_min_max:
            if "qpos_mean" in self.stats:
                self.stats["mean"] = self.stats["qpos_mean"]
                self.stats["std"] = self.stats["qpos_std"]

        # ★ CVAE 모델 초기화
        self.model = ACTVisionPolicy(
            state_dim=STATE_DIM, 
            action_dim=ACTION_DIM, 
            seq_len=SEQ_LEN, 
            chunk_size=CHUNK_SIZE,
            latent_dim=LATENT_DIM,
            d_model=D_MODEL
        ).to(device)
        
        # 가중치 로드 (이제 구조가 같으므로 바로 로드 가능)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # 혹시 모를 prefix 문제 처리 ("module." 등)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '') # DDP로 학습된 경우 대비
            new_state_dict[new_k] = v
            
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"[INFO] 모델 로드 완료: {msg}")
        
        self.model.eval()
        
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _normalize_state(self, state):
        if self.use_min_max:
            s_min = self.stats["min"].to(self.device)
            s_max = self.stats["max"].to(self.device)
            return 2 * (state - s_min) / (s_max - s_min + 1e-5) - 1
        else:
            mean = self.stats["mean"].to(self.device)
            std = self.stats["std"].to(self.device)
            return (state - mean) / std

    def _unnormalize_action(self, action_norm):
        if self.use_min_max:
            a_min = self.stats["min"].to(self.device)
            a_max = self.stats["max"].to(self.device)
            return (action_norm + 1) * (a_max - a_min) / 2 + a_min
        else:
            mean = self.stats["mean"].to(self.device)
            std = self.stats["std"].to(self.device)
            return action_norm * std + mean

    @torch.no_grad()
    def step(self, image_np, state_history):
        # 이미지 전처리
        img_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
        img_t = self.img_transform(img_t).unsqueeze(0).to(self.device)

        # 관절 상태 전처리
        state_t = torch.tensor(np.array(state_history), dtype=torch.float32).to(self.device)
        state_norm = self._normalize_state(state_t).unsqueeze(0)

        # ★ 추론 실행 (action_chunk=None -> z=0 자동 설정)
        pred_chunk, _, _ = self.model(img_t, state_norm, action_chunk=None)
        
        action_real_chunk = self._unnormalize_action(pred_chunk.squeeze(0))
        return action_real_chunk.cpu().numpy()

# =========================================================
# 4. 하드웨어 제어 함수 (이전 성공 코드 유지)
# =========================================================
def manual_raw_write(bus, motor_name, raw_value):
    if motor_name not in bus.motors: return
    motor = bus.motors[motor_name]
    handler, table, _, _ = bus.get_target_info(motor.id)
    addr, size = table["Goal_Position"]
    
    raw_int = int(raw_value)
    
    if size == 4: # XL (Extended)
        packet_val = raw_int & 0xFFFFFFFF
        handler.write4ByteTxRx(bus.port_handler, motor.id, addr, packet_val)
    else: # AX
        packet_val = raw_int & 0xFFFF
        handler.write2ByteTxRx(bus.port_handler, motor.id, addr, packet_val)

def main():
    import pyrealsense2 as rs
    print("[Info] RealSense 장치 리셋 시도...")
    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            dev.hardware_reset()
        time.sleep(2.0)
    except Exception:
        pass

    # 1. 모델 로드
    try:
        inference = ACTInference(MODEL_PATH, device=DEVICE)
    except Exception as e:
        print(f"[Error] 모델 초기화 실패: {e}")
        sys.exit(1)
    
    print("[INFO] CVAE Model Loaded Successfully")
    
    # 2. 로봇 연결
    try:
        robot = SimpleRobot(
            port="/dev/ttyUSB0",
            motor_ids=[10, 11, 12, 13, 14, 15], 
            robot_id="follower",
            is_leader=False,       
            camera_index=0,        
            camera_type="realsense"
        )
        robot.connect(calibrate=False)
        bus = robot.bus 
        if hasattr(bus, 'enable_torque'):
            bus.enable_torque()
    except Exception as e:
        print(f"[Error] 로봇 연결 실패: {e}")
        sys.exit(1)

    state_buffer = deque(maxlen=SEQ_LEN)
    print("\n[Start] Robot Control Loop (CVAE Inference)...")
    
    try:
        while True:
            loop_start = time.time()
            obs = robot.get_observation() 
            if "camera" not in obs or obs["camera"] is None: continue

            current_joints = [obs[f"{name}.pos"] for name in JOINT_NAMES]
            if len(state_buffer) == 0:
                for _ in range(SEQ_LEN): state_buffer.append(current_joints)
            else:
                state_buffer.append(current_joints)
            
            # 추론
            action_chunk = inference.step(obs["camera"], list(state_buffer))
            next_action = action_chunk[0] 

            ai_raw_list = []
            target_norm_list = []
            debug_raw_list = []

            for i, name in enumerate(JOINT_NAMES):
                pred_val = float(next_action[i]) 
                ai_raw_list.append(f"{pred_val:.2f}")

                motor_id = None
                if name in bus.motors:
                    motor_id = bus.motors[name].id

                # ========================================================
                # 11번 모터: 부호 반전(-) 로직 유지
                # ========================================================
                if motor_id == 11:
                    pred_val_inverted = -pred_val # 부호 반전
                    pred_val_geared = pred_val_inverted * GEAR_RATIOS[11]
                    calc_val = (pred_val_geared * 30.0) + 2048
                    raw_val = int(calc_val)
                    target_norm_list.append(f"{pred_val_geared:.1f}")

                # ========================================================
                # 1번 모터
                # ========================================================
                elif motor_id == 1:
                    geared_val = pred_val * GEAR_RATIOS[1]
                    raw_val = bus.denormalize(name, geared_val)
                    target_norm_list.append(f"{geared_val:.1f}")

                # ========================================================
                # 나머지 모터
                # ========================================================
                else:
                    raw_val = bus.denormalize(name, pred_val)
                    target_norm_list.append(f"{pred_val:.1f}")

                debug_raw_list.append(raw_val)
                manual_raw_write(bus, name, raw_val)

            print(f"\r[AI Out] {ai_raw_list} | [RawInt] {debug_raw_list}", end="")

            elapsed = time.time() - loop_start
            sleep_time = (1.0 / CONTROL_HZ) - elapsed
            if sleep_time > 0: time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Stop]")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()