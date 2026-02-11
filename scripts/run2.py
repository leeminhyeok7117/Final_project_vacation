# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import sys
# import time
# import torch
# import torch.nn.functional as F
# import numpy as np
# from collections import deque
# from pathlib import Path
# from torchvision import transforms

# # =========================================================
# # 0. 경로 및 임포트 설정
# # =========================================================
# current_file = Path(__file__).resolve()
# project_root = current_file.parent.parent  # scripts의 상위 디렉토리

# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

# from robots import SimpleRobot 

# try:
#     from policy.act_vision_policy import ACTVisionPolicy, OBS_IMAGE, OBS_STATE
# except ImportError as e:
#     print(f"[Error] act_vision_policy.py를 찾을 수 없습니다. 에러: {e}")
#     sys.exit(1)

# # =========================================================
# # 1. 설정값 (학습 코드와 100% 동일하게 맞춤)
# # =========================================================
# MODEL_FILE_NAME = "checkpoint_epoch_200.pth"

# # 경로 자동 생성
# MODEL_PATH = project_root / "examples" / "models" / MODEL_FILE_NAME
# JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# # 학습 코드(train_act_vision_policy.py) 기준 설정
# SEQ_LEN = 8
# CHUNK_SIZE = 8
# STATE_DIM = 6
# ACTION_DIM = 6
# D_MODEL = 512       

# CONTROL_HZ = 30.0
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =========================================================
# # 2. 추론 클래스
# # =========================================================
# class ACTInference:
#     def __init__(self, checkpoint_path, device="cuda"):
#         self.device = device
#         print(f"[INFO] 체크포인트 로드 중: {checkpoint_path}")
        
#         # 1. 모델 초기화
#         self.policy = ACTVisionPolicy(
#             state_dim=STATE_DIM, 
#             action_dim=ACTION_DIM, 
#             chunk_size=CHUNK_SIZE,
#             n_action_steps=CHUNK_SIZE,
#             d_model=D_MODEL
#         ).to(device)

#         # 2. 체크포인트 로드
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         state_dict = checkpoint['model_state_dict']

#         missing_keys, unexpected_keys = self.policy.load_state_dict(state_dict)
        
#         critical_missing = [k for k in missing_keys if "vae_encoder" not in k]

#         if len(critical_missing) > 0:
#             print("[CRITICAL ERROR] 모델의 핵심 가중치가 누락되었습니다!")
#             sys.exit(1) # 즉시 종료 (안전 확보)
        
#         print("[INFO] 모델 가중치 로드 완료.")
#         if len(missing_keys) > 0:
#             print(f"       -> 학습용 VAE 가중치 {len(missing_keys)}개는 제외되었습니다. (추론 정상)")
        
#         self.policy.eval()

#         # 4. 통계값 로드
#         self.stats = checkpoint['stats']
        
#         # 통계값 전처리
#         self.use_min_max = "min" in self.stats
#         if not self.use_min_max and "qpos_mean" in self.stats:
#             self.stats["mean"] = self.stats["qpos_mean"]
#             self.stats["std"] = self.stats["qpos_std"]

#         # 5. 이미지 정규화
#         self.img_transform = transforms.Compose([
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def _normalize_state(self, state):
#         if self.use_min_max:
#             s_min = self.stats["min"].to(self.device)
#             s_max = self.stats["max"].to(self.device)
#             return 2 * (state - s_min) / (s_max - s_min + 1e-5) - 1
#         else:
#             mean = self.stats["mean"].to(self.device)
#             std = self.stats["std"].to(self.device)
#             return (state - mean) / std

#     def _unnormalize_action(self, action_norm):
#         if self.use_min_max:
#             a_min = self.stats["min"].to(self.device)
#             a_max = self.stats["max"].to(self.device)
#             return (action_norm + 1) * (a_max - a_min) / 2 + a_min
#         else:
#             mean = self.stats["mean"].to(self.device)
#             std = self.stats["std"].to(self.device)
#             return action_norm * std + mean

#     @torch.no_grad()
#     def step(self, image_np, state_history):
#         # 1. 이미지 전처리
#         img_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
#         img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
#         img_t = self.img_transform(img_t).unsqueeze(0).to(self.device)

#         # 2. 관절 상태 전처리
#         state_t = torch.tensor(np.array(state_history[-1]), dtype=torch.float32).to(self.device)
#         state_norm = self._normalize_state(state_t).unsqueeze(0)

#         # 3. 배치 구성
#         batch = {
#             OBS_IMAGE: img_t,
#             OBS_STATE: state_norm
#         }

#         # 4. 추론
#         pred_chunk = self.policy.predict_action_chunk(batch)
#         action_real_chunk = self._unnormalize_action(pred_chunk.squeeze(0))
#         return action_real_chunk.cpu().numpy()

# # =========================================================
# # 3. 메인 실행
# # =========================================================
# def manual_raw_write(bus, motor_name, raw_value):
#     if motor_name not in bus.motors: return
#     motor = bus.motors[motor_name]
#     handler, table, _, _ = bus.get_target_info(motor.id)
#     addr, size = table["Goal_Position"]
    
#     raw_int = int(raw_value)
#     if size == 4:
#         handler.write4ByteTxRx(bus.port_handler, motor.id, addr, raw_int & 0xFFFFFFFF)
#     else:
#         handler.write2ByteTxRx(bus.port_handler, motor.id, addr, raw_int & 0xFFFF)

# def main():
#     import pyrealsense2 as rs
#     try:
#         ctx = rs.context()
#         for dev in ctx.query_devices(): dev.hardware_reset()
#         time.sleep(2.0)
#     except: pass

#     # 추론 객체 생성 (여기서 모델 구조가 안 맞으면 바로 에러 발생)
#     inference = ACTInference(MODEL_PATH, device=DEVICE)
#     print("[INFO] ACT Vision Policy Loaded & Verified")
    
#     try:
#         robot = SimpleRobot(
#             port="/dev/ttyUSB0",
#             motor_ids=[10, 11, 12, 13, 14, 15], 
#             robot_id="follower",
#             is_leader=False,       
#             camera_index=0,        
#             camera_type="realsense"
#         )
#         robot.connect(calibrate=False)
#         bus = robot.bus 
#         if hasattr(bus, 'enable_torque'): bus.enable_torque()
#     except Exception as e:
#         print(f"[Error] 로봇 연결 실패: {e}"); sys.exit(1)

#     state_buffer = deque(maxlen=SEQ_LEN)
#     print("\n[Start] Robot Control Loop...")
    
#     try:
#         while True:
#             loop_start = time.time()
#             obs = robot.get_observation() 
#             if "camera" not in obs or obs["camera"] is None: continue

#             current_joints = [obs[f"{name}.pos"] for name in JOINT_NAMES]
#             state_buffer.append(current_joints)
            
#             # 버퍼가 최소한 1개는 있어야 추론 시작
#             if len(state_buffer) < 1: continue 
            
#             # 추론
#             action_chunk = inference.step(obs["camera"], list(state_buffer))
#             next_action = action_chunk[0] 

#             debug_raw_list = []
#             for i, name in enumerate(JOINT_NAMES):
#                 pred_val = float(next_action[i]) 
#                 raw_val = bus.denormalize(name, pred_val)

#                 debug_raw_list.append(raw_val)
#                 manual_raw_write(bus, name, raw_val)

#             print(f"\r[AI Out] {['%.2f'%f for f in next_action]} | [Raw] {debug_raw_list}", end="")

#             elapsed = time.time() - loop_start
#             time.sleep(max(0, (1.0 / CONTROL_HZ) - elapsed))

#     except KeyboardInterrupt:
#         print("\n[Stop]")
#     finally:
#         robot.disconnect()

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ACT + RealSense(SimpleRobot) PKL 기반 추론 실행 코드
기능:
1) RealSense 카메라 대신 .pkl 파일에 저장된 데이터(Image, State)를 순차적으로 로드
2) 모델(ACT)에 데이터를 입력하여 다음 Action 추론
3) 추론된 Action을 실제 연결된 로봇(Follower)에게 전송
"""

import sys
import time
import argparse
import pickle
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # scripts의 상위 디렉토리

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# 로봇 및 정책 임포트
from robots import SimpleRobot

try:
    from policy.act_vision_policy import ACTVisionPolicy, OBS_IMAGE, OBS_STATE
except ImportError as e:
    print(f"[Error] policy 폴더 내 act_vision_policy.py를 찾을 수 없습니다: {e}")
    sys.exit(1)

# =========================================================
# 유틸: 로봇 raw write (Goal_Position 직접 쓰기)
# =========================================================
def manual_raw_write(bus, motor_name, raw_value):
    if motor_name not in bus.motors:
        return
    motor = bus.motors[motor_name]
    handler, table, _, _ = bus.get_target_info(motor.id)
    addr, size = table["Goal_Position"]

    raw_int = int(raw_value)
    if size == 4:
        handler.write4ByteTxRx(bus.port_handler, motor.id, addr, raw_int & 0xFFFFFFFF)
    else:
        handler.write2ByteTxRx(bus.port_handler, motor.id, addr, raw_int & 0xFFFF)

# =========================================================
# 추론 클래스
# =========================================================
class ACTInference:
    def __init__(
        self,
        checkpoint_path: Path,
        joint_names,
        device: torch.device,
        chunk_size: int = 8,
        camera_is_rgb: bool = False,
    ):
        self.device = device
        self.joint_names = joint_names
        self.chunk_size = chunk_size
        self.camera_is_rgb = camera_is_rgb

        print(f"[INFO] 체크포인트 로드 중: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")

        cfg = ckpt.get("config", {})
        state_dim = int(cfg.get("state_dim", 6))
        action_dim = int(cfg.get("action_dim", 6))
        d_model = int(cfg.get("dim_model", 512))
        n_action_steps = int(cfg.get("n_action_steps", self.chunk_size))
        chunk_size = int(cfg.get("chunk_size", self.chunk_size))

        self.policy = ACTVisionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            d_model=d_model,
        ).to(self.device)

        state_dict = ckpt["model_state_dict"]
        self.policy.load_state_dict(state_dict, strict=False)
        self.policy.eval()
        self.policy.reset()

        self.stats = ckpt.get("stats", {})
        for k in ["min", "max", "mean", "std", "qpos_mean", "qpos_std"]:
            if k in self.stats and not isinstance(self.stats[k], torch.Tensor):
                self.stats[k] = torch.tensor(self.stats[k], dtype=torch.float32)

        self.use_min_max = ("min" in self.stats) and ("max" in self.stats)
        if not self.use_min_max and ("qpos_mean" in self.stats) and ("qpos_std" in self.stats):
            self.stats["mean"] = self.stats["qpos_mean"]
            self.stats["std"] = self.stats["qpos_std"]

        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _normalize_state(self, state_1d: torch.Tensor) -> torch.Tensor:
        if self.use_min_max:
            s_min = self.stats["min"].to(self.device)
            s_max = self.stats["max"].to(self.device)
            return 2.0 * (state_1d - s_min) / (s_max - s_min + 1e-5) - 1.0
        mean = self.stats["mean"].to(self.device)
        std = self.stats["std"].to(self.device)
        return (state_1d - mean) / (std + 1e-8)

    def _unnormalize_action(self, action_norm_1d: torch.Tensor) -> torch.Tensor:
        if self.use_min_max:
            a_min = self.stats["min"].to(self.device)
            a_max = self.stats["max"].to(self.device)
            return (action_norm_1d + 1.0) * (a_max - a_min) / 2.0 + a_min
        mean = self.stats["mean"].to(self.device)
        std = self.stats["std"].to(self.device)
        return action_norm_1d * std + mean

    @torch.no_grad()
    def step(self, image_np: np.ndarray, obs_dict: dict) -> np.ndarray:
        img = image_np.copy()
        if self.camera_is_rgb:
            img = img[:, :, ::-1].copy() # RGB to BGR swap

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
        img_t = self.img_transform(img_t).unsqueeze(0).to(self.device)

        state_vec = np.array([obs_dict[f"{n}.pos"] for n in self.joint_names], dtype=np.float32)
        state_t = torch.from_numpy(state_vec).to(self.device)
        state_norm = self._normalize_state(state_t).unsqueeze(0)

        batch = {OBS_IMAGE: img_t, OBS_STATE: state_norm}
        action_norm = self.policy.select_action(batch).squeeze(0)
        action_real = self._unnormalize_action(action_norm)
        return action_real.detach().cpu().numpy()

# =========================================================
# 자동 모드 판정
# =========================================================
def decide_write_mode(bus, joint_names, obs, probe_actions=None):
    diffs = []
    rt_ok = 0
    for name in joint_names:
        k = f"{name}.pos"
        if k not in obs: continue
        x = float(obs[k])
        try:
            raw = bus.denormalize(name, x)
            x2 = bus.normalize(name, raw)
            diff = abs(float(x2) - x)
            diffs.append(diff)
            if diff < 1e-2: rt_ok += 1
        except: continue
    
    ok_ratio = rt_ok / len(diffs) if diffs else 0
    if ok_ratio >= 0.8: return "B"
    
    if probe_actions is not None:
        a = np.array(probe_actions)
        frac = np.abs(a - np.round(a))
        if np.median(frac) < 0.05: return "A"
    return "B"

# =========================================================
# 메인 실행부
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="체크포인트(.pth) 경로")
    parser.add_argument("--pkl", type=str, required=True, help="추론에 사용할 데이터(.pkl) 경로")
    parser.add_argument("--control_hz", type=float, default=30.0)
    parser.add_argument("--camera_is_rgb", action="store_true")
    args = parser.parse_args()

    # 1. pkl 데이터 로드
    print(f"[INFO] PKL 로딩 중: {args.pkl}")
    with open(args.pkl, 'rb') as f:
        demo_data = pickle.load(f)
    
    # 2. 모델 및 로봇 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    inference = ACTInference(
        checkpoint_path=Path(args.ckpt),
        joint_names=joint_names,
        device=device,
        camera_is_rgb=args.camera_is_rgb
    )

    try:
        robot = SimpleRobot(
            port="/dev/ttyUSB0",
            motor_ids=[10, 11, 12, 13, 14, 15],
            robot_id="follower",
            is_leader=False,
            camera_index=0,
            camera_type="realsense",
        )
        robot.connect(calibrate=False)
        bus = robot.bus
        if hasattr(bus, "enable_torque"): bus.enable_torque()
        print("[INFO] Robot Connected")
    except Exception as e:
        print(f"[Error] 로봇 연결 실패: {e}"); sys.exit(1)

    mode = None
    action_probe = []

    print("\n[Start] PKL Playback Control... (Press CTRL+C to stop)")
    try:
        # pkl 시퀀스 루프
        for i, data_source in enumerate(demo_data):
            start_time = time.time()

            # --- 수정된 부분: data_source가 문자열(경로)일 경우 로드 ---
            if isinstance(data_source, str):
                # 상대 경로일 수 있으므로 pkl 파일이 포함된 폴더 기준으로 경로 설정
                pkl_dir = Path(args.pkl).parent
                with open(pkl_dir / data_source, 'rb') as f:
                    step_data = pickle.load(f)
            else:
                step_data = data_source
            # -------------------------------------------------------

            # 데이터셋 구조에 따라 키값 추출
            img = step_data.get('observation.images.camera') or step_data.get('camera')
            raw_state = step_data.get('observation.state') or step_data.get('qpos')
            
            if img is None or raw_state is None:
                print(f"\n[Skip] Step {i}: 데이터 누락"); continue

            obs_dict = {f"{name}.pos": raw_state[idx] for idx, name in enumerate(joint_names)}

            # 추론
            action = inference.step(img, obs_dict)

            # 모드 판정 (최초 10스텝)
            if mode is None:
                action_probe.append(action)
                if len(action_probe) >= 10:
                    mode = decide_write_mode(bus, joint_names, obs_dict, action_probe)
                    print(f"\n[Decision] Write Mode: {mode}")

            # 로봇 실행
            debug_raw = []
            for j, name in enumerate(joint_names):
                val = float(action[j])
                raw_val = int(round(val)) if mode == "A" else int(round(bus.denormalize(name, val)))
                debug_raw.append(raw_val)
                manual_raw_write(bus, name, raw_val)

            print(f"\r[Step {i:4d}] Mode: {mode or '?'} | Raw: {debug_raw}", end="")

            # 주기 맞춤
            elapsed = time.time() - start_time
            time.sleep(max(0, (1.0 / args.control_hz) - elapsed))

    except KeyboardInterrupt:
        print("\n[Stop] User Interrupted")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
