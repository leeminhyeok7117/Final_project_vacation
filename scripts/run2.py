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
project_root = current_file.parent.parent  # scripts의 상위 디렉토리

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from robots import SimpleRobot 

try:
    from policy.act_vision_policy import ACTVisionPolicy, OBS_IMAGE, OBS_STATE
except ImportError as e:
    print(f"[Error] act_vision_policy.py를 찾을 수 없습니다. 에러: {e}")
    sys.exit(1)

# =========================================================
# 1. 설정값 (학습 코드와 100% 동일하게 맞춤)
# =========================================================
MODEL_FILE_NAME = "best_model.pth"

# 경로 자동 생성
MODEL_PATH = project_root / "examples" / "models" / MODEL_FILE_NAME
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# 학습 코드(train_act_vision_policy.py) 기준 설정
SEQ_LEN = 8
CHUNK_SIZE = 8
STATE_DIM = 6
ACTION_DIM = 6
D_MODEL = 512       

CONTROL_HZ = 30.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 2. 추론 클래스
# =========================================================
class ACTInference:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        print(f"[INFO] 체크포인트 로드 중: {checkpoint_path}")
        
        # 1. 모델 초기화
        self.policy = ACTVisionPolicy(
            state_dim=STATE_DIM, 
            action_dim=ACTION_DIM, 
            chunk_size=CHUNK_SIZE,
            d_model=D_MODEL
        ).to(device)

        # 2. 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        missing_keys, unexpected_keys = self.policy.load_state_dict(state_dict, strict=False)
        
        critical_missing = [k for k in missing_keys if "vae_encoder" not in k]

        if len(critical_missing) > 0:
            print("\n" + "="*60)
            print("[CRITICAL ERROR] 모델의 핵심 가중치가 누락되었습니다!")
            print("이 상태로 실행하면 로봇이 오작동(발작)합니다.")
            print(f"누락된 핵심 키 개수: {len(critical_missing)}개")
            print(f"예시: {critical_missing[:5]} ...")
            print("="*60 + "\n")
            sys.exit(1) # 즉시 종료 (안전 확보)
        
        print("[INFO] 모델 가중치 로드 완료.")
        if len(missing_keys) > 0:
            print(f"       -> 학습용 VAE 가중치 {len(missing_keys)}개는 제외되었습니다. (추론 정상)")
        
        self.policy.eval()

        # 4. 통계값 로드
        self.stats = checkpoint['stats']
        
        # 통계값 전처리
        self.use_min_max = "min" in self.stats
        if not self.use_min_max and "qpos_mean" in self.stats:
            self.stats["mean"] = self.stats["qpos_mean"]
            self.stats["std"] = self.stats["qpos_std"]

        # 5. 이미지 정규화
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
        # 1. 이미지 전처리
        img_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
        img_t = self.img_transform(img_t).unsqueeze(0).to(self.device)

        # 2. 관절 상태 전처리
        state_t = torch.tensor(np.array(state_history[-1]), dtype=torch.float32).to(self.device)
        state_norm = self._normalize_state(state_t).unsqueeze(0)

        # 3. 배치 구성
        batch = {
            OBS_IMAGE: img_t,
            OBS_STATE: state_norm
        }

        # 4. 추론
        pred_chunk = self.policy.predict_action_chunk(batch)
        action_real_chunk = self._unnormalize_action(pred_chunk.squeeze(0))
        return action_real_chunk.cpu().numpy()

# =========================================================
# 3. 메인 실행
# =========================================================
def manual_raw_write(bus, motor_name, raw_value):
    if motor_name not in bus.motors: return
    motor = bus.motors[motor_name]
    handler, table, _, _ = bus.get_target_info(motor.id)
    addr, size = table["Goal_Position"]
    
    raw_int = int(raw_value)
    if size == 4:
        handler.write4ByteTxRx(bus.port_handler, motor.id, addr, raw_int & 0xFFFFFFFF)
    else:
        handler.write2ByteTxRx(bus.port_handler, motor.id, addr, raw_int & 0xFFFF)

def main():
    import pyrealsense2 as rs
    try:
        ctx = rs.context()
        for dev in ctx.query_devices(): dev.hardware_reset()
        time.sleep(2.0)
    except: pass

    # 추론 객체 생성 (여기서 모델 구조가 안 맞으면 바로 에러 발생)
    inference = ACTInference(MODEL_PATH, device=DEVICE)
    print("[INFO] ACT Vision Policy Loaded & Verified")
    
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
        if hasattr(bus, 'enable_torque'): bus.enable_torque()
    except Exception as e:
        print(f"[Error] 로봇 연결 실패: {e}"); sys.exit(1)

    state_buffer = deque(maxlen=SEQ_LEN)
    print("\n[Start] Robot Control Loop...")
    
    try:
        while True:
            loop_start = time.time()
            obs = robot.get_observation() 
            if "camera" not in obs or obs["camera"] is None: continue

            current_joints = [obs[f"{name}.pos"] for name in JOINT_NAMES]
            state_buffer.append(current_joints)
            
            # 버퍼가 최소한 1개는 있어야 추론 시작
            if len(state_buffer) < 1: continue 
            
            # 추론
            action_chunk = inference.step(obs["camera"], list(state_buffer))
            next_action = action_chunk[0] 

            debug_raw_list = []
            for i, name in enumerate(JOINT_NAMES):
                pred_val = float(next_action[i]) 
                
                raw_val = bus.denormalize(name, pred_val)

                debug_raw_list.append(raw_val)
                manual_raw_write(bus, name, raw_val)

            print(f"\r[AI Out] {['%.2f'%f for f in next_action]} | [Raw] {debug_raw_list}", end="")

            elapsed = time.time() - loop_start
            time.sleep(max(0, (1.0 / CONTROL_HZ) - elapsed))

    except KeyboardInterrupt:
        print("\n[Stop]")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()