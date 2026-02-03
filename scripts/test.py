#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
from pathlib import Path
import dynamixel_sdk as dxl

sys.path.insert(0, str(Path(__file__).parent.parent))

from robots import SimpleRobot 

# =========================================================
# 설정 (run.py와 동일)
# =========================================================
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 30.0
GEAR_RATIOS = {1: 2.0, 11: 2.0}

# ★ 테스트할 가짜 데이터 (AI가 이런 값을 뱉는다고 가정)
# 0 -> 10 -> 0 -> -10 -> -30 -> -50 (음수로 쭉 내려감)
TEST_SEQUENCE = [10.0, 0.0, -10.0, -20.0, -30.0, -40.0, -50.0]

def manual_raw_write(bus, motor_name, raw_value):
    """run.py와 100% 동일한 쓰기 함수"""
    if motor_name not in bus.motors: return
    motor = bus.motors[motor_name]
    
    # bus 객체 활용
    packet_handler = dxl.PacketHandler(2.0)
    port_handler = bus.port_handler
    
    handler, table, _, _ = bus.get_target_info(motor.id) 
    addr, size = table["Goal_Position"]
    
    raw_int = int(raw_value)
    
    if size == 4:
        # XL 시리즈 (4byte)
        packet_val = raw_int & 0xFFFFFFFF
        packet_handler.write4ByteTxRx(port_handler, motor.id, addr, packet_val)
    else:
        # AX 시리즈 (2byte)
        packet_val = raw_int & 0xFFFF
        # AX는 기존 handler 사용 (안전을 위해)
        handler.write2ByteTxRx(port_handler, motor.id, addr, packet_val)

def main():
    print("[Info] RealSense 장치 리셋 (생략 가능)...")
    
    try:
        # run.py와 동일한 로봇 연결 설정
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
        
        # ★ 안전장치: 11번 속도 제한만 (모드 변경은 절대 안 함!)
        # 급발진 방지용으로 속도만 살짝 줄입니다.
        packet_handler = dxl.PacketHandler(2.0)
        packet_handler.write4ByteTxRx(bus.port_handler, 11, 112, 100) # 속도 100 (천천히)
        print("[Safety] 11번 모터 속도 제한(100) 적용됨")
            
    except Exception as e:
        print(f"[Error] 로봇 연결 실패: {e}")
        sys.exit(1)

    print("\n[Start] 11번 모터 강제 이동 테스트 (3초 뒤 시작)...")
    time.sleep(3)
    
    try:
        # 테스트 시퀀스를 순회
        for test_val in TEST_SEQUENCE:
            loop_start = time.time()
            
            # run.py 처럼 모든 관절을 돌지만, 11번만 우리가 조작함
            target_norm_list = []
            
            for name in JOINT_NAMES:
                motor_id = None
                if name in bus.motors:
                    motor_id = bus.motors[name].id
                
                # 기본값은 0.0으로 가정
                pred_val = 0.0
                
                # ★★★ [핵심] 11번 모터일 때만 테스트 값(음수) 주입 ★★★
                if motor_id == 11:
                    pred_val = float(test_val) # -10, -30, -50 ...
                    
                    # ----------------------------------------------------
                    # run.py에 들어갈 계산 로직 (기어비 + 오프셋)
                    # ----------------------------------------------------
                    raw_val = bus.denormalize(name, pred_val * 2.0)
                    
                elif motor_id == 1:
                    # 1번 모터 (있다면)
                    raw_val = bus.denormalize(name, pred_val * 2.0)
                else:
                    # 나머지 AX 모터들
                    raw_val = bus.denormalize(name, pred_val)

                # 출력용 저장
                target_norm_list.append(f"{pred_val:.1f}")
                
                # 쓰기
                manual_raw_write(bus, name, raw_val)

            # 상태 출력
            # 11번 모터의 실제 현재 위치도 같이 찍어서 확인
            p_pos, _, _ = packet_handler.read4ByteTxRx(bus.port_handler, 11, 132)
            # 32비트 부호 변환
            if p_pos > 0x7FFFFFFF: p_pos -= 0x100000000
            
            print(f"\r[Input] {test_val:>5.1f} | [TargetRaw] {raw_val:>5} | [ActualPos] {p_pos:>5}", end="")
            
            # 1초씩 대기하며 천천히 이동
            time.sleep(1.0)

        print("\n\n[Done] 테스트 종료. 음수 위치로 잘 갔나요?")

    except KeyboardInterrupt:
        print("\n[Stop]")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()