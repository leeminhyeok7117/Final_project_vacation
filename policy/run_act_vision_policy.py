#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from collections import deque
from policy.act_vision_policy import ACTVisionPolicy, JOINT_KEYS

# =========================
# 설정 (학습과 동일해야 함)
# =========================
MODEL_PATH = "examples/models/latest_policy.pth"
SEQ_LEN = 8
CHUNK_SIZE = 4
STATE_DIM = 6
ACTION_DIM = 6
IMAGE_SIZE = (128, 128)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 모델 로드
# =========================
model = ACTVisionPolicy(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    seq_len=SEQ_LEN,
    chunk_size=CHUNK_SIZE,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("[INFO] ACT Vision Policy loaded")

# =========================
# 상태 히스토리 버퍼
# =========================
state_buffer = deque(maxlen=SEQ_LEN)

def preprocess_image(img_np: np.ndarray) -> torch.Tensor:
    """
    img_np: (H,W,3) uint8
    return: (1,3,h,w) float32
    """
    img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    img = torch.nn.functional.interpolate(
        img.unsqueeze(0),
        size=IMAGE_SIZE,
        mode="bilinear",
        align_corners=False,
    )
    return img.to(DEVICE)


def preprocess_state(obs_dict: dict) -> torch.Tensor:
    """
    obs_dict: observation dict
    return: (6,)
    """
    return torch.tensor(
        [float(obs_dict[k]) for k in JOINT_KEYS],
        dtype=torch.float32,
        device=DEVICE,
    )


@torch.no_grad()
def policy_step(image_np: np.ndarray, obs_dict: dict) -> np.ndarray:
    """
    한 timestep 추론
    return: action (6,)
    """
    # state buffer 채우기 (초기에는 마지막 값 반복)
    s = preprocess_state(obs_dict)
    if len(state_buffer) == 0:
        for _ in range(SEQ_LEN):
            state_buffer.append(s)
    else:
        state_buffer.append(s)

    # (1, L, 6)
    state_seq = torch.stack(list(state_buffer), dim=0).unsqueeze(0)

    # (1,3,h,w)
    img = preprocess_image(image_np)

    # (1,K,6)
    pred_chunk = model(img, state_seq)

    # 첫 action만 사용
    action = pred_chunk[0, 0].cpu().numpy()
    return action


# =========================
# 예시 실행 루프
# =========================
def run_robot_loop(robot):
    """
    robot 인터페이스 예시:
      robot.get_observation() -> dict (camera 포함)
      robot.send_action(action: np.ndarray)
    """
    while True:
        obs = robot.get_observation()
        img_np = obs["camera"]          # (H,W,3) uint8

        action = policy_step(img_np, obs)
        robot.send_action(action)


if __name__ == "__main__":
    # print("이 스크립트는 로봇/시뮬레이터 인터페이스에 연결해서 사용하세요.")
    import pickle
    from pathlib import Path
    pkl_path = sorted(Path("examples/data").glob("demo_*.pkl"))[0]
    obj = pickle.load(open(pkl_path, "rb"))

    obs0 = obj["observations"][0]
    img0 = obs0["camera"]

    a = policy_step(img0, obs0)
    print("pred action:", a, "shape:", a.shape)
