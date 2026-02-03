#!/usr/bin/env python3
"""
합성 데이터 생성 스크립트
학습 파이프라인 검증용
"""

import pickle
import numpy as np
from pathlib import Path

def create_synthetic_demo(num_frames=100, fps=30):
    """
    합성 데모 데이터 생성

    Args:
        num_frames: 프레임 수
        fps: 초당 프레임

    Returns:
        dict: 데모 데이터
    """
    observations = []
    actions = []

    # 6개 모터 (간단한 사인파 패턴)
    motor_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper"
    ]

    for i in range(num_frames):
        # 시간에 따른 부드러운 모터 움직임 (사인파)
        t = i / fps

        obs = {}
        action = {}

        for j, name in enumerate(motor_names):
            # 각 모터마다 다른 주기로 움직임
            freq = 0.5 + j * 0.1  # 주파수 (Hz)
            amplitude = 50.0  # 진폭 (-50 ~ 50)
            phase = j * np.pi / 6  # 위상 차이

            value = amplitude * np.sin(2 * np.pi * freq * t + phase)

            obs[f"{name}.pos"] = value
            action[f"{name}.pos"] = value

        # 합성 이미지 생성 (움직이는 패턴)
        # 간단한 그라디언트 + 노이즈
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # 빨강 채널: 시간에 따라 변화
        image[:, :, 0] = int(128 + 127 * np.sin(2 * np.pi * t))

        # 초록 채널: 세로 그라디언트
        image[:, :, 1] = np.linspace(0, 255, 480, dtype=np.uint8).reshape(-1, 1)

        # 파랑 채널: 가로 그라디언트
        image[:, :, 2] = np.linspace(0, 255, 640, dtype=np.uint8)

        # 노이즈 추가
        noise = np.random.randint(-20, 20, (480, 640, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        obs["camera"] = image

        observations.append(obs)
        actions.append(action)

    return {
        "observations": observations,
        "actions": actions,
        "fps": fps,
        "motor_model": "ax-12a",
        "num_frames": num_frames,
    }


def main():
    """메인 함수"""

    print("="*70)
    print("  합성 데이터 생성")
    print("="*70)

    # 프로젝트 루트 경로
    PROJECT_ROOT = Path(__file__).parent.parent
    output_dir = PROJECT_ROOT / "examples" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3개의 데모 생성 (다양한 길이)
    demo_configs = [
        {"num_frames": 150, "fps": 30},  # 5초
        {"num_frames": 120, "fps": 30},  # 4초
        {"num_frames": 180, "fps": 30},  # 6초
    ]

    for i, config in enumerate(demo_configs, 1):
        print(f"\n데모 {i} 생성 중...")
        print(f"  프레임: {config['num_frames']}, FPS: {config['fps']}")

        data = create_synthetic_demo(**config)

        save_path = output_dir / f"demo_{i:03d}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"  저장: {save_path}")
        print(f"  크기: {file_size_mb:.2f} MB")

    print("\n" + "="*70)
    print("  완료!")
    print("="*70)

    # 데이터 검증
    print("\n데이터 검증:")
    all_files = sorted(output_dir.glob("demo_*.pkl"))
    print(f"  총 파일 수: {len(all_files)}")

    total_frames = 0
    for pkl_file in all_files:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        total_frames += data["num_frames"]

    print(f"  총 프레임 수: {total_frames}")
    print(f"  총 시간: {total_frames / 30:.1f}초")

    # 첫 번째 데모 샘플 확인
    print("\n첫 번째 데모 샘플:")
    with open(all_files[0], "rb") as f:
        data = pickle.load(f)

    obs = data["observations"][0]
    action = data["actions"][0]

    print(f"  관측 키: {list(obs.keys())}")
    print(f"  액션 키: {list(action.keys())}")
    print(f"  이미지 shape: {obs['camera'].shape}")
    print(f"  이미지 dtype: {obs['camera'].dtype}")
    print(f"  모터 위치 샘플:")
    for key in sorted(obs.keys()):
        if key.endswith(".pos"):
            print(f"    {key}: {obs[key]:.2f}")

    print(f"\n데이터 경로: {output_dir}")


if __name__ == "__main__":
    main()
