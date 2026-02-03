#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
from pathlib import Path
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from robots import SimpleRobot 
from policy.act_vision_policy import ACTVisionPolicy

# =========================================================
# 설정
# =========================================================
MODEL_PATH = "examples/models/policy_epoch_1000.pth"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SEQ_LEN = 8
CHUNK_SIZE = 4
STATE_DIM = 6
ACTION_DIM = 6
IMAGE_SIZE = (128, 128)
CONTROL_HZ = 30.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ★ 중요: SimpleRobot과 동일하게 기어비 설정 (1번 Leader, 11번 Follower)
GEAR_RATIOS = {1: 2.0, 11: 2.0}

# =========================================================
# ACT Inference 클래스
# =========================================================
class ACTInference:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        
        print(f"[INFO] 체크포인트 로드 중: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.stats = checkpoint['stats']
        
        # 정규화 방식 자동 감지
        self.use_min_max = "min" in self.stats
        if not self.use_min_max:
            if "qpos_mean" in self.stats:
                self.stats["mean"] = self.stats["qpos_mean"]
                self.stats["std"] = self.stats["qpos_std"]

        self.model = ACTVisionPolicy(
            state_dim=STATE_DIM, 
            action_dim=ACTION_DIM, 
            seq_len=SEQ_LEN, 
            chunk_size=CHUNK_SIZE,
            dim_feedforward=1024,
            backbone="resnet18"
        ).to(device)
        
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
            
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
        img_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
        img_t = self.img_transform(img_t).unsqueeze(0).to(self.device)

        state_t = torch.tensor(np.array(state_history), dtype=torch.float32).to(self.device)
        state_norm = self._normalize_state(state_t).unsqueeze(0)

        pred_chunk = self.model(img_t, state_norm)
        action_real_chunk = self._unnormalize_action(pred_chunk.squeeze(0))
        
        return action_real_chunk.cpu().numpy()

def manual_raw_write(bus, motor_name, raw_value):
    """
    파이썬의 정수(음수 포함)를 다이내믹셀이 이해하는 비트열로 변환하여 전송
    """
    if motor_name not in bus.motors: return
    motor = bus.motors[motor_name]
    handler, table, _, _ = bus.get_target_info(motor.id)
    addr, size = table["Goal_Position"]
    
    raw_int = int(raw_value)
    
    if size == 4:
        # Extended Position (4bytes)
        packet_val = raw_int & 0xFFFFFFFF
        handler.write4ByteTxRx(bus.port_handler, motor.id, addr, packet_val)
    else:
        # Position (2bytes)
        packet_val = raw_int & 0xFFFF
        handler.write2ByteTxRx(bus.port_handler, motor.id, addr, packet_val)

def main():
    # 0. 리얼센스 강제 리셋 (USB 에러 방지용)
    import pyrealsense2 as rs
    print("[Info] RealSense 장치 리셋 시도...")
    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            dev.hardware_reset()
        time.sleep(2.0)
    except Exception:
        pass

    try:
        inference = ACTInference(MODEL_PATH, device=DEVICE)
    except Exception as e:
        print(f"[Error] 모델 초기화 실패: {e}")
        sys.exit(1)
    
    print("[INFO] Model & Stats Loaded Successfully")
    
    try:
        # User Configuration에 맞춰 Motor ID 확인 필요
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
        
        # 안전장치: 토크 켜기
        if hasattr(bus, 'enable_torque'):
            bus.enable_torque()
            print("[Info] 토크 활성화 확인")
            
    except Exception as e:
        print(f"[Error] 로봇 연결 실패: {e}")
        sys.exit(1)

    state_buffer = deque(maxlen=SEQ_LEN)
    print("\n[Start] Robot Control Loop...")
    
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
            
            action_chunk = inference.step(obs["camera"], list(state_buffer))
            next_action = action_chunk[0] 

            target_norm_list = []
            debug_raw_list = [] # 디버깅용 리스트

            for i, name in enumerate(JOINT_NAMES):
                pred_val = float(next_action[i]) 

                # 기어비 보정
                if name in bus.motors:
                    motor_id = bus.motors[name].id
                    if motor_id in GEAR_RATIOS:
                        pred_val = pred_val * GEAR_RATIOS[motor_id]

                target_norm_list.append(f"{pred_val:.1f}")

                # 1. Raw 값 변환
                raw_val = bus.denormalize(name, pred_val)
                
                # ★★★ [중요] 실제로 모터에 들어가는 정수값 확인 ★★★
                raw_int = int(raw_val)
                debug_raw_list.append(raw_int)

                # 2. 쓰기
                manual_raw_write(bus, name, raw_val)

            # [출력] Target 값과 함께 실제 모터에 들어가는 Raw 값(정수)도 같이 출력
            print(f"\r[Target] {target_norm_list} | [RawInt] {debug_raw_list}", end="")

            elapsed = time.time() - loop_start
            sleep_time = (1.0 / CONTROL_HZ) - elapsed
            if sleep_time > 0: time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Stop]")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()