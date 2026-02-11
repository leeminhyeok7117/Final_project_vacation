#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from robots import SimpleRobot

try:
    from policy.act_vision_policy import ACTVisionPolicy, OBS_IMAGE, OBS_STATE
except ImportError as e:
    print(f"[Error] act_vision_policy.py 누락: {e}"); sys.exit(1)

# =========================================================
# 유틸리티 및 추론 클래스
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
    def __init__(self, checkpoint_path, joint_names, device, chunk_size=8, camera_is_rgb=False):
        self.device = device
        self.joint_names = joint_names
        self.chunk_size = chunk_size
        self.camera_is_rgb = camera_is_rgb

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        cfg = ckpt.get("config", {})
        
        self.policy = ACTVisionPolicy(
            state_dim=int(cfg.get("state_dim", 6)),
            action_dim=int(cfg.get("action_dim", 6)),
            chunk_size=int(cfg.get("chunk_size", 8)),
            n_action_steps=int(cfg.get("n_action_steps", 8)),
            d_model=int(cfg.get("dim_model", 512)),
        ).to(self.device)

        self.policy.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.policy.eval()

        self.stats = ckpt.get("stats", {})
        self.use_min_max = "min" in self.stats
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Temporal Ensembling 관련
        self.all_time_actions = None
        self.step_idx = 0
        self.k = 0.01

    def _normalize_state(self, state_t):
        if self.use_min_max:
            s_min, s_max = self.stats["min"].to(self.device), self.stats["max"].to(self.device)
            return 2.0 * (state_t - s_min) / (s_max - s_min + 1e-5) - 1.0
        mean, std = self.stats["qpos_mean"].to(self.device), self.stats["qpos_std"].to(self.device)
        return (state_t - mean) / (std + 1e-8)

    def _unnormalize_action(self, action_norm_t):
        if self.use_min_max:
            a_min, a_max = self.stats["min"].to(self.device), self.stats["max"].to(self.device)
            return (action_norm_t + 1.0) * (a_max - a_min) / 2.0 + a_min
        mean, std = self.stats["qpos_mean"].to(self.device), self.stats["qpos_std"].to(self.device)
        return action_norm_t * std + mean

    @torch.no_grad()
    def step(self, image_np, obs_dict):
        img = image_np.copy()
        if self.camera_is_rgb: img = img[:, :, ::-1].copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
        img_t = self.img_transform(img_t).unsqueeze(0).to(self.device)

        state_vec = np.array([obs_dict[f"{n}.pos"] for n in self.joint_names], dtype=np.float32)
        state_norm = self._normalize_state(torch.from_numpy(state_vec).to(self.device)).unsqueeze(0)

        # Chunk 예측
        batch = {OBS_IMAGE: img_t, OBS_STATE: state_norm}
        pred_chunk = self.policy.predict_action_chunk(batch)
        action_real_chunk = self._unnormalize_action(pred_chunk.squeeze(0)).cpu().numpy()

        # Ensembling
        t = self.step_idx
        if self.all_time_actions is None:
            self.all_time_actions = np.zeros([10000, 10000 + self.chunk_size, len(self.joint_names)])
        
        self.all_time_actions[t, t : t + self.chunk_size] = action_real_chunk
        actions_populated = self.all_time_actions[: t + 1, t, :]
        actions_populated = actions_populated[np.any(actions_populated != 0, axis=1)]
        
        weights = np.exp(-self.k * np.arange(len(actions_populated))[::-1])
        weights /= weights.sum()
        
        combined_action = np.sum(actions_populated * weights[:, None], axis=0)
        self.step_idx += 1
        return combined_action

# =========================================================
# 메인 루프
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--pkl", type=str, required=True)
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f: demo_data = pickle.load(f)
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    inference = ACTInference(Path(args.ckpt), joint_names, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        robot = SimpleRobot(port="/dev/ttyUSB0", motor_ids=[10, 11, 12, 13, 14, 15], 
                            robot_id="follower", is_leader=False, camera_type="realsense")
        robot.connect(calibrate=False)
        bus = robot.bus
        if hasattr(bus, "enable_torque"): bus.enable_torque()
    except Exception as e:
        print(f"로봇 연결 실패: {e}"); sys.exit(1)

    try:
        for i, step_data in enumerate(demo_data):
            start_time = time.time()
            img = step_data.get('observation.images.camera') or step_data.get('camera')
            raw_state = step_data.get('observation.state') or step_data.get('qpos')
            if img is None or raw_state is None: continue

            obs_dict = {f"{name}.pos": raw_state[idx] for idx, name in enumerate(joint_names)}
            action = inference.step(img, obs_dict)

            for j, name in enumerate(joint_names):
                raw_val = bus.denormalize(name, float(action[j]))
                manual_raw_write(bus, name, raw_val)

            time.sleep(max(0, (1.0 / 30.0) - (time.time() - start_time)))
    except KeyboardInterrupt: pass
    finally: robot.disconnect()

if __name__ == "__main__": main()