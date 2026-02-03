#!/usr/bin/env python
"""
텔레오퍼레이션 스크립트 - URDF 제거
Leader 암을 조작하여 Follower 로봇 제어
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path

# 패키지 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from robots import SimpleRobot


def main():
    """텔레오퍼레이션 메인"""

    # ========================================================================
    # 인자 파싱
    # ========================================================================
    parser = argparse.ArgumentParser(description="Leader-Follower 텔레오퍼레이션")
    parser.add_argument("--leader-port", default="/dev/ttyUSB0", help="Leader 포트")
    parser.add_argument("--follower-port", default="/dev/ttyUSB1", help="Follower 포트")
    parser.add_argument("--motor-model", default="ax-12a", help="모터 모델")
    parser.add_argument("--fps", type=int, default=30, help="제어 주파수")
    args = parser.parse_args()

    # ========================================================================
    # 설정
    # ========================================================================
    LEADER_PORT = args.leader_port
    FOLLOWER_PORT = args.follower_port
    MOTOR_MODEL = args.motor_model
    FPS = args.fps

    print("\n" + "="*70)
    print("  텔레오퍼레이션 시스템")
    print("="*70)
    print(f"  Leader:   {LEADER_PORT}")
    print(f"  Follower: {FOLLOWER_PORT}")
    print(f"  Model:    {MOTOR_MODEL}")
    print(f"  FPS:      {FPS}")
    print("="*70 + "\n")

    # ========================================================================
    # 로봇 초기화
    # ========================================================================
    print("[Init] Leader 암 초기화...")
    leader = SimpleRobot(
        port=LEADER_PORT,
        motor_ids=[0, 1, 2, 3, 4, 5],
        motor_model=MOTOR_MODEL,
        robot_id="leader",
        is_leader=True,  # 토크 OFF, 손으로 움직임
    )

    print("[Init] Follower 로봇 초기화...")
    follower = SimpleRobot(
        port=FOLLOWER_PORT,
        motor_ids=[10, 11, 12, 13, 14, 15],
        motor_model=MOTOR_MODEL,
        robot_id="follower",
        is_leader=False,  # 토크 ON, 명령으로 움직임
    )

    # ========================================================================
    # 연결
    # ========================================================================
    try:
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)    

    except Exception as e:
        print(f"\n[ERROR] 연결 실패: {e}")
        return

    # ========================================================================
    # 텔레오퍼레이션 루프
    # ========================================================================
    print("\n" + "="*70)
    print("  텔레오퍼레이션 시작!")
    print("="*70)
    print("[Info] Leader를 움직이면 Follower가 따라합니다.")
    print("[Info] Ctrl+C로 종료")
    print("="*70 + "\n")
    loop_count = 0
    try:
        while True:
            start = time.perf_counter()

            # 1. Leader 위치 읽기
            leader_obs = leader.get_observation()

            # 2. Action 생성 (Follower에 전송할 .pos 값들)
            action = {
                k: v
                for k, v in leader_obs.items() 
                if k.endswith(".pos")
            }

            # 3. Follower 제어 명령 전송
            follower.send_action(action)

            # 4. 실시간 상태 출력 (화면 고정 모드)
            # 매 루프마다 출력하면 깜빡임이 심하므로 FPS의 절반 주기로 업데이트
            if loop_count % 5 == 0:
                # 리눅스 터미널 지우기
                os.system('clear')
                
                # 출력용 데이터 정리 (소수점 2자리)
                display_leader = {k: round(v, 2) for k, v in action.items()}
                
                print("="*70)
                print(f" [Teleoperation Status] FPS: {FPS} | Running...")
                print("="*70)
                print("\n[Leader Normalized Positions]")
                # 정렬된 JSON 형태로 출력
                print(json.dumps(display_leader, indent=4, sort_keys=False))
                print("\n" + "="*70)
                print(" [Ctrl+C] 종료")

            loop_count += 1

            # FPS 유지
            elapsed = time.perf_counter() - start
            time.sleep(max(1/FPS - elapsed, 0))

    except KeyboardInterrupt:
        print("\n\n[Stop] 종료됨")

    finally:
        follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
