#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from pathlib import Path
from torchvision import transforms

# =========================================================
# 0. 경로 및 임포트 설정
# =========================================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent 

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from robots import SimpleRobot 

try:
    from policy.act_vision_policy import ACTVisionPolicy, OBS_IMAGE, OBS_STATE
except ImportError as e:
    print(f"[Error] act_vision_policy.py를 찾을 수 없습니다. 에러: {e}")
    sys.exit(1)

# =========================================================
# 1. 설정값
# =========================================================
MODEL_FILE_NAME = "checkpoint_epoch_200.pth"
MODEL_PATH = project_root / "examples" / "models" / MODEL_FILE_NAME
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

SEQ_LEN = 8
CHUNK_SIZE = 8
STATE_DIM = 6
ACTION_DIM = 6
D_MODEL = 512       
CONTROL_HZ = 30.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEAR_RATIOS = {1: 2.0, 11: 2.0}

# =========================================================
# 2. 추론 클래스 (Temporal Ensembling)
# =========================================================
class ACTInference:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        print(f"[INFO] 체크포인트 로드 중: {checkpoint_path}")
        
        # 모델 초기화
        self.policy = ACTVisionPolicy(
            state_dim=STATE_DIM, 
            action_dim=ACTION_DIM, 
            chunk_size=CHUNK_SIZE,
            n_action_steps=CHUNK_SIZE,
            d_model=D_MODEL
        ).to(device)

        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("model.") and "backbone" in k:
                new_k = f"model.{k}"
            else:
                new_k = k
            new_state_dict[new_k] = v

        self.policy.load_state_dict(new_state_dict, strict=False)
        self.policy.eval()

        self.stats = checkpoint['stats']
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Temporal Ensembling 버퍼
        self.action_queue = deque(maxlen=CHUNK_SIZE)
        self.k_weight = 0.2

    def _normalize_state(self, state):
        if "min" in self.stats:
            s_min = self.stats["min"].to(self.device)
            s_max = self.stats["max"].to(self.device)
            return 2 * (state - s_min) / (s_max - s_min + 1e-5) - 1
        elif "qpos_mean" in self.stats:
            mean = self.stats["qpos_mean"].to(self.device)
            std = self.stats["qpos_std"].to(self.device)
            return (state - mean) / (std + 1e-8)
        else:
            return state

    def _unnormalize_action(self, action_norm):
        if "min" in self.stats:
            a_min = self.stats["min"].to(self.device)
            a_max = self.stats["max"].to(self.device)
            return (action_norm + 1) * (a_max - a_min) / 2 + a_min
        elif "qpos_mean" in self.stats:
            mean = self.stats["qpos_mean"].to(self.device)
            std = self.stats["qpos_std"].to(self.device)
            return action_norm * std + mean
        else:
            return action_norm

    @torch.no_grad()
    def step(self, image_np, current_joints):
        # 1. 전처리
        img_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
        img_t = self.img_transform(img_t).unsqueeze(0).to(self.device)
        
        state_t = torch.tensor(np.array(current_joints), dtype=torch.float32).to(self.device)
        state_norm = self._normalize_state(state_t).unsqueeze(0)

        # 2. 추론
        batch = {
            OBS_IMAGE: img_t,
            OBS_STATE: state_norm
        }
        
        pred_chunk = self.policy.predict_action_chunk(batch)
        action_real_chunk = self._unnormalize_action(pred_chunk.squeeze(0)).cpu().numpy()

        # 3. Temporal Ensembling
        self.action_queue.append(action_real_chunk)
        
        actions_for_curr_step = []
        for i, chunk in enumerate(reversed(self.action_queue)):
            if i < len(chunk): 
                actions_for_curr_step.append(chunk[i])
        
        if not actions_for_curr_step:
            return np.zeros(ACTION_DIM)

        weights = np.exp(-self.k_weight * np.arange(len(actions_for_curr_step)))
        weights = weights / weights.sum()
        
        combined_action = np.sum(np.array(actions_for_curr_step) * weights[:, None], axis=0)
        return combined_action

# =========================================================
# 3. 메인 함수
# =========================================================
def manual_raw_write(bus, motor_name, raw_value):
    if motor_name not in bus.motors: return
    motor = bus.motors[motor_name]
    handler, table, _, _ = bus.get_target_info(motor.id)
    addr, size = table["Goal_Position"]
    raw_int = int(round(raw_value))
    
    if size == 4:
        handler.write4ByteTxRx(bus.port_handler, motor.id, addr, raw_int & 0xFFFFFFFF)
    else:
        handler.write2ByteTxRx(bus.port_handler, motor.id, addr, raw_int & 0xFFFF)

def main():
    # ---------------------------------------------------------
    # [수정] 강제 리셋 코드 삭제 -> SimpleRobot이 알아서 하도록 원복
    # ---------------------------------------------------------
    
    # 모델 로드
    try:
        inference = ACTInference(MODEL_PATH, device=DEVICE)
    except Exception as e:
        print(f"[Error] 모델 로드 실패: {e}"); sys.exit(1)

    # 로봇 연결
    try:
        print("[INFO] 로봇 연결 시작...")
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
        try:
            if hasattr(bus, 'enable_torque'): bus.enable_torque()
        except: pass
        
        print("[Info] 카메라 데이터 안정화 대기 (3초)...")
        time.sleep(3.0)
    except Exception as e:
        print(f"[Error] 로봇 연결 실패: {e}"); sys.exit(1)

    print("\n[Start] Control Loop (Ensembling ON)...")

    # FPS 계산을 위한 변수
    loop_times = deque(maxlen=30) # 최근 30번의 루프 시간 저장

    try:
        while True:
            loop_start = time.time()
            
            # 1. 관측값 획득 시간 측정
            obs_start = time.time()
            obs = robot.get_observation() 
            if "camera" not in obs or obs["camera"] is None: continue
            obs_time = time.time() - obs_start

            current_joints = [obs[f"{name}.pos"] for name in JOINT_NAMES]
            
            # 2. 추론(Inference) 시간 측정
            inf_start = time.time()
            next_action = inference.step(obs["camera"], current_joints)
            inf_time = time.time() - inf_start

            # 3. 제어 명령 전송
            for i, name in enumerate(JOINT_NAMES):
                motor_id = bus.motors[name].id
                ratio = GEAR_RATIOS.get(motor_id, 1.0) # test.py의 기어비 반영
                raw_val = bus.denormalize(name, float(next_action[i]) * ratio)
                manual_raw_write(bus, name, raw_val)

            # 전체 루프 소요 시간 계산
            elapsed = time.time() - loop_start
            loop_times.append(elapsed)
            avg_elapsed = sum(loop_times) / len(loop_times)
            current_fps = 1.0 / avg_elapsed if avg_elapsed > 0 else 0

            # 디버깅 정보 출력
            print(f"\r[FPS: {current_fps:4.1f}] Total: {elapsed*1000:4.1f}ms (Obs: {obs_time*1000:3.1f}ms, Inf: {inf_time*1000:3.1f}ms)", end="")

            # 주기(Hz) 맞추기
            sleep_time = (1.0 / CONTROL_HZ) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 처리 시간이 1/30초를 넘었을 경우 경고
                pass
            
    except KeyboardInterrupt:
        print("\n[Stop]")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()


            # for i, name in enumerate(JOINT_NAMES):
            #     # ★ 11번 모터 반전 제거됨 (기본 denormalize)
            #     raw_val = bus.denormalize(name, float(next_action[i]))
            #     manual_raw_write(bus, name, raw_val)
            #     debug_vals.append(int(raw_val))

            # print(f"\r[Act] {debug_vals}", end="")