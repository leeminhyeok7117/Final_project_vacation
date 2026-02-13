# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import sys
# import time
# import numpy as np
# from pathlib import Path
# import dynamixel_sdk as dxl

# sys.path.insert(0, str(Path(__file__).parent.parent))

# from robots import SimpleRobot 

# # =========================================================
# # 설정 (run.py와 동일)
# # =========================================================
# JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# CONTROL_HZ = 30.0
# GEAR_RATIOS = {1: 2.0, 11: 2.0}

# # ★ 테스트할 가짜 데이터 (AI가 이런 값을 뱉는다고 가정)
# # 0 -> 10 -> 0 -> -10 -> -30 -> -50 (음수로 쭉 내려감)
# TEST_SEQUENCE = [10.0, 0.0, -10.0, -20.0, -30.0, -40.0, -50.0]

# def manual_raw_write(bus, motor_name, raw_value):
#     """run.py와 100% 동일한 쓰기 함수"""
#     if motor_name not in bus.motors: return
#     motor = bus.motors[motor_name]
    
#     # bus 객체 활용
#     packet_handler = dxl.PacketHandler(2.0)
#     port_handler = bus.port_handler
    
#     handler, table, _, _ = bus.get_target_info(motor.id) 
#     addr, size = table["Goal_Position"]
    
#     raw_int = int(raw_value)
    
#     if size == 4:
#         # XL 시리즈 (4byte)
#         packet_val = raw_int & 0xFFFFFFFF
#         packet_handler.write4ByteTxRx(port_handler, motor.id, addr, packet_val)
#     else:
#         # AX 시리즈 (2byte)
#         packet_val = raw_int & 0xFFFF
#         # AX는 기존 handler 사용 (안전을 위해)
#         handler.write2ByteTxRx(port_handler, motor.id, addr, packet_val)

# def main():
#     print("[Info] RealSense 장치 리셋 (생략 가능)...")
    
#     try:
#         # run.py와 동일한 로봇 연결 설정
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
        
#         # ★ 안전장치: 11번 속도 제한만 (모드 변경은 절대 안 함!)
#         # 급발진 방지용으로 속도만 살짝 줄입니다.
#         packet_handler = dxl.PacketHandler(2.0)
#         packet_handler.write4ByteTxRx(bus.port_handler, 11, 112, 100) # 속도 100 (천천히)
#         print("[Safety] 11번 모터 속도 제한(100) 적용됨")
            
#     except Exception as e:
#         print(f"[Error] 로봇 연결 실패: {e}")
#         sys.exit(1)

#     print("\n[Start] 11번 모터 강제 이동 테스트 (3초 뒤 시작)...")
#     time.sleep(3)
    
#     try:
#         # 테스트 시퀀스를 순회
#         for test_val in TEST_SEQUENCE:
#             loop_start = time.time()
            
#             # run.py 처럼 모든 관절을 돌지만, 11번만 우리가 조작함
#             target_norm_list = []
            
#             for name in JOINT_NAMES:
#                 motor_id = None
#                 if name in bus.motors:
#                     motor_id = bus.motors[name].id
                
#                 # 기본값은 0.0으로 가정
#                 pred_val = 0.0
                
#                 # ★★★ [핵심] 11번 모터일 때만 테스트 값(음수) 주입 ★★★
#                 if motor_id == 11:
#                     pred_val = float(test_val) # -10, -30, -50 ...
                    
#                     # ----------------------------------------------------
#                     # run.py에 들어갈 계산 로직 (기어비 + 오프셋)
#                     # ----------------------------------------------------
#                     raw_val = bus.denormalize(name, pred_val * 2.0)
                    
#                 elif motor_id == 1:
#                     # 1번 모터 (있다면)
#                     raw_val = bus.denormalize(name, pred_val * 2.0)
#                 else:
#                     # 나머지 AX 모터들
#                     raw_val = bus.denormalize(name, pred_val)

#                 # 출력용 저장
#                 target_norm_list.append(f"{pred_val:.1f}")
                
#                 # 쓰기
#                 manual_raw_write(bus, name, raw_val)

#             # 상태 출력
#             # 11번 모터의 실제 현재 위치도 같이 찍어서 확인
#             p_pos, _, _ = packet_handler.read4ByteTxRx(bus.port_handler, 11, 132)
#             # 32비트 부호 변환
#             if p_pos > 0x7FFFFFFF: p_pos -= 0x100000000
            
#             print(f"\r[Input] {test_val:>5.1f} | [TargetRaw] {raw_val:>5} | [ActualPos] {p_pos:>5}", end="")
            
#             # 1초씩 대기하며 천천히 이동
#             time.sleep(1.0)

#         print("\n\n[Done] 테스트 종료. 음수 위치로 잘 갔나요?")

#     except KeyboardInterrupt:
#         print("\n[Stop]")
#     finally:
#         robot.disconnect()

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import pickle
import numpy as np
from pathlib import Path
import dynamixel_sdk as dxl

sys.path.insert(0, str(Path(__file__).parent.parent))
from robots import SimpleRobot

# =========================================================
# 설정 (기존 코드 최대 유지)
# =========================================================
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 30.0
GEAR_RATIOS = {1: 2.0, 11: 2.0}

# =========================================================
# PKL 설정
# =========================================================
PKL_PATH = "/home/lmh/minimal_lerobot/examples/data/demo_20260130_142757_19.pkl"  # TODO: 실제 pkl 경로로 변경


def load_action_frames_from_pkl(pkl_path: str, joint_names: list) -> np.ndarray:
    """
    pkl(dict)에서 'actions'를 읽어서 (T, 6) float numpy 배열로 변환.
    actions[t] dict 키가 아래 둘 다 지원:
      - "shoulder_pan"
      - "shoulder_pan.pos"  (현재 pkl이 이 형태)
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict) or "actions" not in obj:
        raise ValueError("pkl이 dict 형태가 아니거나 'actions' 키가 없습니다.")

    actions = obj["actions"]
    if not isinstance(actions, list) or len(actions) == 0:
        raise ValueError("'actions'가 비어있거나 list가 아닙니다.")

    # case A) actions[t] 가 dict 인 경우
    if isinstance(actions[0], dict):
        T = len(actions)
        frames = np.zeros((T, len(joint_names)), dtype=float)

        for t, ad in enumerate(actions):
            keys = list(ad.keys())

            for j, name in enumerate(joint_names):
                # 1) exact key
                if name in ad:
                    frames[t, j] = float(ad[name])
                    continue

                # 2) ".pos" key
                pos_key = f"{name}.pos"
                if pos_key in ad:
                    frames[t, j] = float(ad[pos_key])
                    continue

                # 3) 그래도 없으면 에러
                raise KeyError(f"actions[{t}]에 '{name}' 또는 '{name}.pos' 키가 없습니다. keys={keys}")

        return frames

    # case B) actions[t] 가 list/tuple/ndarray 인 경우
    arr = np.asarray(actions, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"'actions'를 배열로 변환했지만 2차원이 아닙니다. shape={arr.shape}")

    if arr.shape[1] == len(joint_names):
        return arr  # (T,6)
    if arr.shape[0] == len(joint_names) and arr.shape[1] != len(joint_names):
        return arr.T  # (6,T) -> (T,6)

    raise ValueError(f"'actions' shape={arr.shape} 해석 불가. (T,6) 또는 (6,T)여야 합니다.")


def manual_raw_write(bus, motor_name, raw_value):
    """run.py와 100% 동일한 쓰기 함수"""
    if motor_name not in bus.motors:
        return
    motor = bus.motors[motor_name]

    packet_handler = dxl.PacketHandler(2.0)
    port_handler = bus.port_handler

    handler, table, _, _ = bus.get_target_info(motor.id)
    addr, size = table["Goal_Position"]

    raw_int = int(raw_value)

    if size == 4:
        packet_val = raw_int & 0xFFFFFFFF
        packet_handler.write4ByteTxRx(port_handler, motor.id, addr, packet_val)
    else:
        packet_val = raw_int & 0xFFFF
        handler.write2ByteTxRx(port_handler, motor.id, addr, packet_val)


def main():
    # =========================================================
    # 1) pkl 로드: actions -> (T,6)
    # =========================================================
    try:
        frames = load_action_frames_from_pkl(PKL_PATH, JOINT_NAMES)
        print(f"[Info] PKL 로드 완료: {PKL_PATH} / actions shape={frames.shape}")
        print(f"[Info] actions 예시(첫 프레임): {frames[0]}")
    except Exception as e:
        print(f"[Error] pkl 로드 실패: {e}")
        sys.exit(1)

    print("[Info] RealSense 장치 리셋 (생략 가능)...")

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

        # 안전장치: 11번 속도 제한만 (원 코드 유지)
        packet_handler = dxl.PacketHandler(2.0)
        packet_handler.write4ByteTxRx(bus.port_handler, 11, 112, 100)
        print("[Safety] 11번 모터 속도 제한(100) 적용됨")

    except Exception as e:
        print(f"[Error] 로봇 연결 실패: {e}")
        sys.exit(1)

    print("\n[Start] PKL actions 기반 전체 모터 구동 (3초 뒤 시작)...")
    time.sleep(3)

    try:
        packet_handler = dxl.PacketHandler(2.0)

        # pkl fps가 있으면 그걸 사용(없으면 CONTROL_HZ)
        with open(PKL_PATH, "rb") as f:
            meta = pickle.load(f)
        hz = float(meta.get("fps", CONTROL_HZ))
        dt_target = 1.0 / hz

        for t in range(frames.shape[0]):
            loop_start = time.time()

            target_norm_list = []
            raw_by_name = {}

            # 프레임의 6개 값을 JOINT_NAMES 순서대로 사용
            for j_idx, name in enumerate(JOINT_NAMES):
                if name not in bus.motors:
                    target_norm_list.append("NA")
                    continue

                motor_id = bus.motors[name].id
                pred_val = float(frames[t, j_idx])

                # 기존 코드 로직 유지: 특정 모터는 *2
                ratio = GEAR_RATIOS.get(motor_id, 1.0)
                raw_val = bus.denormalize(name, pred_val * ratio)

                target_norm_list.append(f"{pred_val:.3f}")
                raw_by_name[name] = int(raw_val)

                manual_raw_write(bus, name, raw_val)

            # 11번 현재 위치 출력(기존 스타일 유지)
            p_pos, _, _ = packet_handler.read4ByteTxRx(bus.port_handler, 11, 132)
            if p_pos > 0x7FFFFFFF:
                p_pos -= 0x100000000

            # 11번 목표 raw 찾기
            raw11 = None
            for n in JOINT_NAMES:
                if n in bus.motors and bus.motors[n].id == 11:
                    raw11 = raw_by_name.get(n, None)
                    break

            print(
                f"\r[t={t:04d}/{frames.shape[0]-1:04d}] "
                f"[Inputs] {target_norm_list} | [TargetRaw11] {raw11} | [ActualPos11] {p_pos}",
                end=""
            )

            # fps 주기 맞추기
            dt = time.time() - loop_start
            time.sleep(max(0.0, dt_target - dt))

        print("\n\n[Done] pkl actions 재생 완료")

    except KeyboardInterrupt:
        print("\n[Stop]")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()