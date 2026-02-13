import sys
import time
import cv2
import numpy as np
from pathlib import Path

# (경로 설정 부분은 기존과 동일)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from robots import SimpleRobot 

def main():
    print("[INFO] RealSense 카메라 연결 테스트...")
    
    # ★ 중요: 학습 때와 똑같이 RealSense로 설정해야 합니다.
    try:
        robot = SimpleRobot(
            port="/dev/ttyUSB0", 
            motor_ids=[10, 11, 12, 13, 14, 15], 
            robot_id="follower", 
            is_leader=False,       
            camera_index=0,        
            camera_type="realsense"  # <--- 학습 환경과 동일하게 설정
        )
        robot.connect(calibrate=False)
    except Exception as e:
        print(f"[Error] 연결 실패: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print(" [RealSense 색상 데이터 검증] ")
    print(" 1. 카메라 앞에 '파란색(Blue)' 물건을 두세요.")
    print(" 2. 'Raw View' 창을 보세요.")
    print("    - 물건이 '빨간색'으로 보인다? -> 정상 (RGB 데이터임)")
    print("    - 물건이 '파란색'으로 보인다? -> 주의 (BGR 데이터임)")
    print("="*60 + "\n")

    try:
        while True:
            obs = robot.get_observation()
            if "camera" not in obs or obs["camera"] is None:
                continue
            
            # 로봇(RealSense)이 주는 원본 데이터
            img = obs["camera"]

            # ---------------------------------------------------------
            # [검증 1] 모델에 들어가는 데이터 (Raw)
            # cv2.imshow는 BGR을 기대하므로, RGB 데이터가 들어오면 색이 반전되어 보임.
            # 즉, 여기서 색이 이상해야(파란색->빨간색) 모델 입장에서는 정답(RGB)인 것임.
            cv2.imshow("1. Raw View (Expect Inverted Colors)", img)
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # [검증 2] 사람 눈에 편하게 보정한 화면
            # 사람이 보기에 정상이라면, 실제 데이터는 RGB라는 뜻.
            # (RGB -> BGR로 바꿔서 화면에 뿌림)
            img_bgr_for_human = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("2. Human View (Corrected)", img_bgr_for_human)
            # ---------------------------------------------------------

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    finally:
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()