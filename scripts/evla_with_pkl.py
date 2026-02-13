#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import pickle
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from collections import deque

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
    print(f"[Error] act_vision_policy.py를 찾을 수 없습니다: {e}")
    sys.exit(1)

# =========================================================
# 1. 설정값
# =========================================================
MODEL_PATH = project_root / "examples" / "models" / "checkpoint_epoch_200.pth"
PKL_DATA_PATH = project_root / "examples" / "data" / "demo_20260130_142757_19.pkl" 

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 30.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 2. 유틸리티 및 추론 클래스
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

class ACTInference:
    def __init__(self, checkpoint_path, joint_names, device):
        self.device = device
        self.joint_names = joint_names
        
        print(f"[INFO] 체크포인트 로드 중: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        
        # Config 처리
        state_dim, action_dim, chunk_size, d_model = 6, 6, 8, 512
        if "config" in ckpt:
            cfg = ckpt["config"]
            if isinstance(cfg, dict):
                state_dim = int(cfg.get("state_dim", 6))
                action_dim = int(cfg.get("action_dim", 6))
                chunk_size = int(cfg.get("chunk_size", 8))
                d_model = int(cfg.get("dim_model", 512))
            else:
                state_dim = getattr(cfg, "state_dim", 6)
                action_dim = getattr(cfg, "action_dim", 6)
                chunk_size = getattr(cfg, "chunk_size", 8)
                d_model = getattr(cfg, "dim_model", 512)

        self.policy = ACTVisionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
            d_model=d_model,
        ).to(self.device)

        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt["model"]
        self.policy.load_state_dict(state_dict, strict=False)
        self.policy.eval()

        self.stats = ckpt.get("stats", {})
        self.use_min_max = "min" in self.stats
        
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.action_queue = deque(maxlen=chunk_size)
        self.k_weight = 0.01

    def _normalize_state(self, state_t):
        if self.use_min_max:
            s_min, s_max = self.stats["min"].to(self.device), self.stats["max"].to(self.device)
            return 2.0 * (state_t - s_min) / (s_max - s_min + 1e-5) - 1.0
        elif "qpos_mean" in self.stats:
            mean, std = self.stats["qpos_mean"].to(self.device), self.stats["qpos_std"].to(self.device)
            return (state_t - mean) / (std + 1e-8)
        else:
            return state_t

    def _unnormalize_action(self, action_norm_t):
        if self.use_min_max:
            a_min, a_max = self.stats["min"].to(self.device), self.stats["max"].to(self.device)
            return (action_norm_t + 1.0) * (a_max - a_min) / 2.0 + a_min
        elif "qpos_mean" in self.stats:
            mean, std = self.stats["qpos_mean"].to(self.device), self.stats["qpos_std"].to(self.device)
            return action_norm_t * std + mean
        else:
            return action_norm_t

    @torch.no_grad()
    def step(self, image_np, obs_dict):
        # 1. 이미지 전처리
        img_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
        img_t = self.img_transform(img_t).unsqueeze(0).to(self.device)

        # 2. 상태 전처리
        state_vec = np.array([obs_dict[f"{n}.pos"] for n in self.joint_names], dtype=np.float32)
        state_norm = self._normalize_state(torch.from_numpy(state_vec).to(self.device)).unsqueeze(0)

        # 3. 추론 (배치 차원 정확히 맞춤: 5차원 에러 방지)
        batch = {
            OBS_IMAGE: img_t, # [1, 3, 128, 128]
            OBS_STATE: state_norm          # [1, 6]
        }
        
        pred_chunk = self.policy.predict_action_chunk(batch)
        action_real_chunk = self._unnormalize_action(pred_chunk.squeeze(0)).cpu().numpy()

        # 4. Ensembling 로직
        self.action_queue.append(action_real_chunk)
        
        actions_for_curr_step = []
        for i, chunk in enumerate(reversed(self.action_queue)):
            if i < len(chunk): 
                actions_for_curr_step.append(chunk[i])
        
        if not actions_for_curr_step:
            return np.zeros(6)

        weights = np.exp(-self.k_weight * np.arange(len(actions_for_curr_step)))
        weights = weights / weights.sum()
        
        combined_action = np.sum(np.array(actions_for_curr_step) * weights[:, None], axis=0)
        return combined_action

# =========================================================
# 3. 메인 실행 루프
# =========================================================
def main():
    if not PKL_DATA_PATH.exists():
        print(f"[Error] PKL 파일을 찾을 수 없습니다: {PKL_DATA_PATH}")
        return
    
    print(f"[INFO] PKL 로딩 중: {PKL_DATA_PATH}")
    with open(PKL_DATA_PATH, 'rb') as f:
        loaded_data = pickle.load(f)
    
    # 딕셔너리 구조 처리
    if isinstance(loaded_data, dict):
        if 'observations' in loaded_data:
            demo_data = loaded_data['observations']
            print(f"[INFO] 딕셔너리 구조 감지됨. ({len(demo_data)} steps)")
        else:
            print(f"[Warn] 'observations' 키 없음. 대체 키 탐색 중...")
            for k, v in loaded_data.items():
                if isinstance(v, list) and len(v) > 0:
                    demo_data = v
                    print(f"[INFO] 대신 '{k}' 키 데이터를 사용합니다.")
                    break
            else:
                print("[Error] 데이터를 찾을 수 없습니다.")
                return
    else:
        demo_data = loaded_data

    # 추론 객체
    inference = ACTInference(MODEL_PATH, JOINT_NAMES, DEVICE)

    # 로봇 연결
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
        if hasattr(bus, "enable_torque"): bus.enable_torque()
    except Exception as e:
        print(f"[Error] 로봇 연결 실패: {e}"); sys.exit(1)

    print("\n[Start] PKL Playback with Temporal Ensembling...")
    try:
        for i, step_data in enumerate(demo_data):
            start_time = time.time()
            
            # 1. 파일 경로 처리 (필요시)
            if isinstance(step_data, str):
                pkl_dir = PKL_DATA_PATH.parent
                full_path = pkl_dir / step_data
                if not full_path.exists(): continue
                with open(full_path, 'rb') as f:
                    step_data = pickle.load(f)

            # ----------------------------------------------------
            # 데이터 추출 (키 이름 호환성 체크)
            # ----------------------------------------------------
            
            # (1) 이미지 찾기
            img = None
            if 'camera' in step_data: img = step_data['camera']
            elif 'observation.images.camera' in step_data: img = step_data['observation.images.camera']
            elif 'images' in step_data and isinstance(step_data['images'], dict):
                img = step_data['images'].get('camera')
            elif 'observation' in step_data and isinstance(step_data['observation'], dict):
                 obs = step_data['observation']
                 if 'images' in obs: img = obs['images'].get('camera')
                 elif 'image' in obs: img = obs['image']

            # (2) 상태(State) 찾기 - ★ 수정된 부분 ★
            raw_state = None
            
            # 먼저 묶음 키 확인
            if 'qpos' in step_data: raw_state = step_data['qpos']
            elif 'observation.state' in step_data: raw_state = step_data['observation.state']
            elif 'state' in step_data: raw_state = step_data['state']
            elif 'observation' in step_data and isinstance(step_data['observation'], dict):
                obs = step_data['observation']
                raw_state = obs.get('state') or obs.get('qpos')
            
            # ★ 묶음 키가 없으면, 개별 관절 키들을 모아서 리스트로 만듦 (사용자 케이스 해결)
            if raw_state is None:
                try:
                    temp_joints = []
                    all_found = True
                    for name in JOINT_NAMES:
                        key = f"{name}.pos"
                        if key in step_data:
                            temp_joints.append(step_data[key])
                        else:
                            all_found = False
                            break
                    if all_found:
                        raw_state = temp_joints
                except:
                    pass

            # 데이터 검증
            if img is None or raw_state is None:
                if i < 3: print(f"[Skip] Step {i}: 데이터 누락 (img={img is not None}, state={raw_state is not None})")
                continue

            # 추론
            obs_dict = {f"{name}.pos": raw_state[idx] for idx, name in enumerate(JOINT_NAMES)}
            action = inference.step(img, obs_dict)

            # 로봇 제어
            for j, name in enumerate(JOINT_NAMES):
                raw_val = bus.denormalize(name, float(action[j]))
                manual_raw_write(bus, name, raw_val)

            print(f"\r[Step {i:4d}] Action sent...", end="", flush=True)

            # 속도 조절
            time.sleep(max(0, (1.0 / CONTROL_HZ) - (time.time() - start_time)))

    except KeyboardInterrupt:
        print("\n[Stop] Stopped by user.")
    except Exception:
        print(f"\n[CRITICAL ERROR] 실행 중 오류 발생!")
        traceback.print_exc()
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()