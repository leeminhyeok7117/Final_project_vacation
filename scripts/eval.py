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

# =========================================================
# 2. 추론 클래스 (Temporal Ensembling 통합)
# =========================================================
class ACTInference:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        print(f"[INFO] 체크포인트 로드 중: {checkpoint_path}")
        
        self.policy = ACTVisionPolicy(
            state_dim=STATE_DIM, 
            action_dim=ACTION_DIM, 
            chunk_size=CHUNK_SIZE,
            n_action_steps=CHUNK_SIZE,
            d_model=D_MODEL
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.policy.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.policy.eval()

        self.stats = checkpoint['stats']
        self.use_min_max = "min" in self.stats
        if not self.use_min_max and "qpos_mean" in self.stats:
            self.stats["mean"] = self.stats["qpos_mean"]
            self.stats["std"] = self.stats["qpos_std"]

        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Temporal Ensembling 버퍼 초기화
        self.all_time_actions = None
        self.step_idx = 0
        self.k = 0.01  # 지수 가중치 (작을수록 과거 예측 중시, 클수록 최신 예측 중시)

    def _normalize_state(self, state):
        if self.use_min_max:
            s_min = self.stats["min"].to(self.device)
            s_max = self.stats["max"].to(self.device)
            return 2 * (state - s_min) / (s_max - s_min + 1e-5) - 1
        else:
            mean = self.stats["mean"].to(self.device)
            std = self.stats["std"].to(self.device)
            return (state - mean) / (std + 1e-8)

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
    def step(self, image_np, current_joints):
        # 1. 전처리
        img_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
        img_t = self.img_transform(img_t).unsqueeze(0).to(self.device)
        state_t = torch.tensor(current_joints, dtype=torch.float32).to(self.device)
        state_norm = self._normalize_state(state_t).unsqueeze(0)

        # 2. 추론 (Chunk 예측)
        batch = {OBS_IMAGE: img_t, OBS_STATE: state_norm}
        pred_chunk = self.policy.predict_action_chunk(batch)
        action_real_chunk = self._unnormalize_action(pred_chunk.squeeze(0)).cpu().numpy()

        # 3. Temporal Ensembling
        curr_t = self.step_idx
        if self.all_time_actions is None:
            # 넉넉한 크기의 버퍼 생성 [최대스텝, 최대스텝+청크, 액션차원]
            self.all_time_actions = np.zeros([20000, 20000 + CHUNK_SIZE, ACTION_DIM])
        
        self.all_time_actions[curr_t, curr_t : curr_t + CHUNK_SIZE] = action_real_chunk
        
        # 현재 시점(curr_t)에 겹치는 모든 과거의 예측값들을 수집
        actions_for_curr_step = self.all_time_actions[: curr_t + 1, curr_t, :]
        # 유효한(0이 아닌) 값만 필터링
        actions_populated = actions_for_curr_step[np.any(actions_for_curr_step != 0, axis=1)]
        
        # 지수 가중치 적용
        num_preds = len(actions_populated)
        weights = np.exp(-self.k * np.arange(num_preds)[::-1])
        weights = weights / weights.sum()
        
        combined_action = np.sum(actions_populated * weights[:, None], axis=0)
        
        self.step_idx += 1
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
    inference = ACTInference(MODEL_PATH, device=DEVICE)
    try:
        robot = SimpleRobot(port="/dev/ttyUSB0", motor_ids=[10, 11, 12, 13, 14, 15], 
                            robot_id="follower", is_leader=False, camera_type="realsense")
        robot.connect(calibrate=False)
        bus = robot.bus 
        if hasattr(bus, 'enable_torque'): bus.enable_torque()
    except Exception as e:
        print(f"[Error] 로봇 연결 실패: {e}"); sys.exit(1)

    print("\n[Start] Control Loop with Temporal Ensembling...")
    try:
        while True:
            loop_start = time.time()
            obs = robot.get_observation() 
            if "camera" not in obs or obs["camera"] is None: continue

            current_joints = [obs[f"{name}.pos"] for name in JOINT_NAMES]
            
            # 앙상블된 액션 1개 받아오기
            next_action = inference.step(obs["camera"], current_joints)

            for i, name in enumerate(JOINT_NAMES):
                raw_val = bus.denormalize(name, float(next_action[i]))
                manual_raw_write(bus, name, raw_val)

            elapsed = time.time() - loop_start
            time.sleep(max(0, (1.0 / CONTROL_HZ) - elapsed))
    except KeyboardInterrupt:
        print("\n[Stop]")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()