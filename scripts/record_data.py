#!/usr/bin/env python
"""
데이터 녹화 스크립트 (토크 강제 해제 및 디버깅 강화 버전)
"""

import argparse
import sys
import time
import pickle
import cv2
import numpy as np

from pathlib import Path
from datetime import datetime

# 패키지 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from robots import SimpleRobot

# 사용자가 지정한 초기 위치
INITIAL_POSITIONS = {
    "shoulder_pan.pos": -0.67,
    "shoulder_lift.pos": 50.94,
    "elbow_flex.pos": -92.74,
    "wrist_flex.pos": -26.63,
    "wrist_roll.pos": -1.33,
    "gripper.pos": 5.81
}

def force_torque_off(robot_instance):
    """
    SimpleRobot 객체 내부를 뒤져서 어떻게든 토크를 끕니다.
    """
    print("\n" + "!"*50)
    print("[DEBUG] 리더 로봇 내부 속성 탐색 및 토크 해제 시도")
    
    # 1. 먼저 robot_instance 자체가 disable_torque를 가지고 있는지 확인
    if hasattr(robot_instance, 'disable_torque'):
        print("  >> Method 1: robot.disable_torque() 호출")
        robot_instance.disable_torque()
        return True

    # 2. 내부 속성들 중 'disable_torque' 메서드를 가진 객체(Bus)가 있는지 탐색
    # SimpleRobot의 속성들을 리스트로 뽑아봅니다.
    attributes = dir(robot_instance)
    print(f"  [Info] 감지된 속성들: { [a for a in attributes if not a.startswith('__')] }")

    success = False
    
    # 예상되는 변수명 후보군
    candidates = ['robot', 'bus', 'motors', 'motor_bus', 'dxl', 'controller', 'driver']
    
    for attr_name in candidates:
        if hasattr(robot_instance, attr_name):
            obj = getattr(robot_instance, attr_name)
            if hasattr(obj, 'disable_torque'):
                print(f"  >> Method 2: robot.{attr_name}.disable_torque() 호출 성공!")
                obj.disable_torque()
                success = True
                break
    
    if not success:
        print("  [Warn] 명시적인 disable_torque 메서드를 찾지 못했습니다.")
        print("  [Warn] 직접 패킷을 보내는 최후의 수단을 시도합니다.")
        
        # 3. 최후의 수단: motors 딕셔너리를 찾아서 개별 모터 제어 시도
        # 보통 SimpleRobot은 self.motors = {...} 형태를 가집니다.
        if hasattr(robot_instance, 'motors'):
            motors = getattr(robot_instance, 'motors')
            # motors가 딕셔너리면
            if isinstance(motors, dict):
                 print(f"  >> Method 3: 개별 모터 {list(motors.keys())} 토크 해제 시도")
                 # 여기서 실제로 패킷을 보내는 객체를 또 찾아야 하는데...
                 # 만약 bus 객체를 못 찾았다면 이 방법도 힘들 수 있습니다.
                 pass

    print("!"*50 + "\n")
    return success

def main():
    # ========================================================================
    # 1. 인자 파싱 및 설정
    # ========================================================================
    parser = argparse.ArgumentParser(description="데이터 녹화 (Teleop 기반)")
    parser.add_argument("--leader-port", default="/dev/ttyUSB0", help="Leader 포트")
    parser.add_argument("--follower-port", default="/dev/ttyUSB1", help="Follower 포트")
    parser.add_argument("--motor-model", default="ax-12a", help="기본 모터 모델")
    parser.add_argument("--fps", type=int, default=30, help="제어 및 녹화 주파수")
    parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스")
    parser.add_argument("--camera-type", default="realsense", help="카메라 타입 (opencv/realsense)")
    parser.add_argument("--output-dir", default=None, help="출력 디렉토리")
    args = parser.parse_args()

    FPS = args.fps
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "examples" / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    i=20
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = DATA_DIR / f"demo_{timestamp}_{i}.pkl"

    # ========================================================================
    # 2. 로봇 초기화
    # ========================================================================
    print(f"[Init] 로봇 연결 시도 중...")
    
    # Leader: 이동을 위해 is_leader=False로 시작
    leader = SimpleRobot(
        port=args.leader_port,
        motor_ids=[0, 1, 2, 3, 4, 5],
        motor_model=args.motor_model,
        robot_id="leader",
        is_leader=False, 
    )

    follower = SimpleRobot(
        port=args.follower_port,
        motor_ids=[10, 11, 12, 13, 14, 15],
        motor_model=args.motor_model,
        robot_id="follower",
        is_leader=False,
        camera_index=args.camera,
        camera_type=args.camera_type,
    )

    try:
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)
    except Exception as e:
        print(f"[ERROR] 연결 실패: {e}")
        return
    
    # ========================================================================
    # 3. 초기 위치 이동 및 강제 토크 해제
    # ========================================================================
    print("\n" + "-"*50)
    print("[Action] 모든 로봇을 초기 위치로 이동합니다 (3초 대기)...")
    
    follower.send_action(INITIAL_POSITIONS)
    leader.send_action(INITIAL_POSITIONS)
    
    # 이동 시간 충분히 줌
    time.sleep(3.0) 
    
    print("[Action] 리더 암 모드 변경 및 토크 해제 시도")
    
    # 1. 로직 변경: 더 이상 명령을 보내지 않도록 설정
    leader.is_leader = True 
    
    # 2. 아주 짧은 대기 (상태 변경이 반영될 시간)
    time.sleep(0.5)

    # 3. ★ 토크 강제 해제 함수 호출 ★
    force_torque_off(leader)
    
    print("[Ready] 초기화 완료. 리더 암을 손으로 움직여 보세요.")
    print("-" * 50 + "\n")

    # ========================================================================
    # 4. 녹화 루프
    # ========================================================================
    print("="*70)
    print(f"  녹화 준비 완료! 저장 경로: {save_path}")
    print("  - 카메라 창에서 'q'를 누르면 종료 및 저장됩니다.")
    print("="*70)
    input("엔터를 누르면 녹화 시작...")

    window_name = "Robot Observation Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    observations = []
    actions = []
    frame_count = 0

    try:
        while True:
            loop_start = time.perf_counter()

            # (1) 데이터 획득
            leader_obs = leader.get_observation()
            action = {k: v for k, v in leader_obs.items() if k.endswith(".pos")}

            follower.send_action(action)
            observation = follower.get_observation()

            # (2) 이미지 처리
            frame = observation.get("camera")
            if frame is not None:
                if isinstance(frame, dict):
                    cam_key = list(frame.keys())[0]
                    frame = frame[cam_key]

                if isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.1 else frame.astype(np.uint8)
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(window_name, bgr_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            observations.append(observation)
            actions.append(action)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  녹화 중: {frame_count} 프레임 ({frame_count/FPS:.1f}초)")

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(1/FPS - elapsed, 0))

    except KeyboardInterrupt:
        print("\n[Stop] 중단되었습니다.")
    finally:
        cv2.destroyAllWindows()
        # 종료 시 안전하게 토크 해제 시도
        force_torque_off(leader)
        follower.disconnect()
        leader.disconnect()

    # ========================================================================
    # 5. 파일 저장
    # ========================================================================
    if frame_count > 0:
        print(f"\n[Save] {frame_count}개 프레임 저장 중...")
        data = {
            "observations": observations,
            "actions": actions,
            "fps": FPS,
            "num_frames": frame_count,
            "timestamp": timestamp,
            "initial_positions": INITIAL_POSITIONS
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[OK] 저장 완료: {save_path}")
    else:
        print("[Warn] 데이터가 없습니다.")

if __name__ == "__main__":
    main()