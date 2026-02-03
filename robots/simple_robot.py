"""
간단한 로봇 구현 - URDF 제거, 모방학습에 집중
AX-12A 또는 XL/XM 시리즈 Dynamixel 모터 사용
"""

import logging
from pathlib import Path
from typing import Any

from motors import Motor, MotorNormMode, DynamixelMotorsBus, MotorCalibration

# cameras 패키지에서:
#   - OpenCVCamera: OpenCV 카메라 래퍼
#   - RealSenseCamera: RealSense D435 카메라
from cameras import OpenCVCamera, RealSenseCamera

# data_types 패키지에서:
#   - RobotObservation: LeRobot 호환 관측 데이터
#   - RobotAction: LeRobot 호환 액션 데이터
#   - dict_to_observation: dict → RobotObservation 변환
from data_types import RobotObservation, RobotAction, dict_to_observation

logger = logging.getLogger(__name__)


class SimpleRobot:
    """
    간단한 6-DOF 로봇 구현

    모방학습(Behavior Cloning)을 위한 최소 기능:
    - Leader-Follower 텔레오퍼레이션
    - 데이터 수집 (관측 + 액션)
    - Policy 학습 및 평가

    사용 예:
        # Leader 로봇 (손으로 움직임)
        leader = SimpleRobot(
            port="/dev/ttyUSB0",
            robot_id="leader",
            is_leader=True,
        )

        # Follower 로봇 (명령으로 움직임)
        follower = SimpleRobot(
            port="/dev/ttyUSB1",
            robot_id="follower",
            is_leader=False,
            camera_index=0,  # 카메라 추가
        )

        leader.connect()
        follower.connect()

        # 텔레오퍼레이션
        leader_obs = leader.get_observation()
        action = {k: v for k, v in leader_obs.items() if k.endswith(".pos")}
        follower.send_action(action)
    """

    def __init__(
        self,
        port: str,
        motor_ids: list[int],
        motor_model: str = "ax-12a",
        robot_id: str = "robot",
        calib_dir: Path | None = None,
        camera_index: int | None = None,
        camera_type: str = "opencv",
        is_leader: bool = False,
    ):
        """
        Args:
            port: USB 포트 (예: "/dev/ttyUSB0")
            motor_model: 모터 모델 ("ax-12a", "xl430-w250", 등)
            robot_id: 로봇 ID (캘리브레이션 파일명)
            calib_dir: 캘리브레이션 디렉토리
            camera_index: 카메라 인덱스 (None이면 카메라 없음)
            camera_type: 카메라 타입 ("opencv" 또는 "realsense")
            is_leader: Leader 모드 (토크 비활성화)
        """
        self.robot_id = robot_id
        self.motor_model = motor_model
        self.is_leader = is_leader
        self.gear_ratios = {1: 2.0, 11: 2.0}
        # self.gear_ratios = {}
        # ====================================================================
        # 캘리브레이션 경로 설정
        # ====================================================================
        if calib_dir is None:
            calib_dir = Path.home() / ".minimal_lerobot" / "calibration"
        self.calib_file = calib_dir / f"{robot_id}.json"

        # ====================================================================
        # 모터 정의 - 여기서 모터 구성을 정의합니다
        # ====================================================================

        norm_mode = MotorNormMode.RANGE_M100_100  # -100~100 정규화

        motors = {
            "shoulder_pan":  Motor(motor_ids[0], "ax-12a", norm_mode),
            "shoulder_lift": Motor(motor_ids[1], "xl430", norm_mode),
            "elbow_flex":    Motor(motor_ids[2], "ax-12a", norm_mode),
            "wrist_flex":    Motor(motor_ids[3], "ax-12a", norm_mode),
            "wrist_roll":    Motor(motor_ids[4], "ax-12a", norm_mode),
            "gripper":       Motor(motor_ids[5], "ax-12a", MotorNormMode.RANGE_0_100),  # 0~100
        }

        # ====================================================================
        # 커스텀 로봇을 만들려면 위의 motors 딕셔너리를 수정하세요
        # ====================================================================
        # 예제 1: 4-DOF 로봇
        # motors = {
        #     "joint1": Motor(1, motor_model, norm_mode),
        #     "joint2": Motor(2, motor_model, norm_mode),
        #     "joint3": Motor(3, motor_model, norm_mode),
        #     "joint4": Motor(4, motor_model, norm_mode),
        # }

        # 예제 2: 다른 모터 ID 사용
        # motors = {
        #     "shoulder_pan":  Motor(10, motor_model, norm_mode),  # ID=10
        #     "shoulder_lift": Motor(20, motor_model, norm_mode),  # ID=20
        #     "elbow_flex":    Motor(30, motor_model, norm_mode),  # ID=30
        # }

        # 예제 3: 각도(도) 정규화 사용
        # motors = {
        #     "joint1": Motor(1, motor_model, MotorNormMode.DEGREES),  # 각도로 표현
        # }
        # ====================================================================

        # ====================================================================
        # DynamixelMotorsBus 초기화
        # ====================================================================
        # DynamixelMotorsBus는 motors 패키지에서 import한 클래스입니다.
        # 역할:
        #   - DynamixelSDK 래퍼
        #   - sync_read("Present_Position"): 모든 모터 위치 읽기
        #   - sync_write("Goal_Position", {...}): 모든 모터 위치 쓰기
        #   - enable_torque(): 토크 활성화
        #   - disable_torque(): 토크 비활성화

        self.bus = DynamixelMotorsBus(
            port=port,
            motors=motors,
            calibration=None,
        )

        # 캘리브레이션 로드 (있으면)
        if self.calib_file.exists():
            self.bus.load_calibration(self.calib_file)

        # ====================================================================
        # 카메라 초기화 (선택사항)
        # ====================================================================
        # 카메라 타입:
        #   - "opencv": OpenCVCamera (일반 USB 카메라)
        #   - "realsense": RealSenseCamera (D435, RGB + Depth)
        #
        # OpenCVCamera:
        #   - read(): RGB 이미지 (numpy array)
        #
        # RealSenseCamera:
        #   - read(): RGB 이미지
        #   - read_rgbd(): RGB + Depth 이미지
        #   - get_intrinsics(): 카메라 내부 파라미터

        self.camera = None
        self.camera_type = camera_type

        if camera_index is not None:
            if camera_type == "opencv":
                self.camera = OpenCVCamera(index=camera_index)
            elif camera_type == "realsense":
                self.camera = RealSenseCamera(
                    width=640,
                    height=480,
                    fps=30,
                    enable_depth=True,
                )
            else:
                raise ValueError(
                    f"Unknown camera_type: {camera_type}. "
                    f"Use 'opencv' or 'realsense'"
                )

    @property
    def is_connected(self) -> bool:
        """연결 상태"""
        camera_ok = self.camera.is_connected if self.camera else True
        return self.bus.is_connected and camera_ok

    def connect(self, calibrate: bool = True):
        """
        로봇 연결

        Args:
            calibrate: 캘리브레이션 파일이 없으면 수동 캘리브레이션 실행
        """
        # 모터 연결
        self.bus.connect()

        # 캘리브레이션 확인
        if not self.bus.is_calibrated and calibrate:
            print("\n[WARNING] 캘리브레이션 데이터가 없습니다!")
            self.calibrate()

        # 카메라 연결
        if self.camera:
            self.camera.connect()

        # ====================================================================
        # 토크 설정: Leader는 비활성화, Follower는 활성화
        # ====================================================================
        # Leader (is_leader=True):
        #   - 토크 OFF → 손으로 움직일 수 있음
        #   - get_observation()으로 위치 읽기만 가능
        #
        # Follower (is_leader=False):
        #   - 토크 ON → 명령으로 움직임
        #   - send_action()으로 위치 쓰기 가능

        if self.is_leader:
            self.bus.disable_torque()
            print(f"[Leader] 토크 비활성화 (손으로 움직임)")
        else:
            self.bus.enable_torque()
            print(f"[Follower] 토크 활성화 (명령으로 움직임)")

        print(f"[OK] {self.robot_id} 연결 완료!\n")

    def disconnect(self):
        """연결 해제"""
        self.bus.disconnect(disable_torque=True)

        if self.camera:
            self.camera.disconnect()

        print(f"[Disconnect] {self.robot_id} 연결 해제됨\n")

    def calibrate(self):
        """
        간소화된 캘리브레이션

        사용자가 각 모터의 범위를 입력
        """
        print("\n" + "="*60)
        print("  수동 캘리브레이션")
        print("="*60)
        print("각 모터의 최소/최대 위치를 입력하세요.")
        print("(토크를 끄고 직접 움직여서 확인)")
        print("="*60 + "\n")

        self.bus.disable_torque()

        calibration = {}

        for motor_name, motor in self.bus.motors.items():
            print(f"\n[Calibrate] {motor_name} (ID: {motor.id})")

            # 간단한 기본값 사용
            if "gripper" in motor_name:
                range_min = 0
                range_max = 1023 if "ax" in self.motor_model.lower() else 4095
            else:
                if "ax" in self.motor_model.lower():
                    range_min = 200
                    range_max = 800
                else:
                    range_min = 1000
                    range_max = 3000

            print(f"  기본값: {range_min} ~ {range_max}")
            user_input = input("  이 값 사용? (엔터=예, 'n'=직접 입력): ")

            if user_input.lower() == 'n':
                input(f"  [방법] 로봇을 '최소' 위치로 옮기고 [엔터]를 누르세요...")
                range_min = self.bus.get_raw_position(motor.id) # 로봇이 직접 자기 위치를 읽음
                print(f"  => 기록된 값: {range_min}")

                input(f"  [방법] 로봇을 '최대' 위치로 옮기고 [엔터]를 누르세요...")
                range_max = self.bus.get_raw_position(motor.id) # 로봇이 직접 자기 위치를 읽음
                print(f"  => 기록된 값: {range_max}")

            calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=512 if "ax" in self.motor_model.lower() else 2048,
                range_min=range_min,
                range_max=range_max,
            )

        # 저장
        self.bus.calibration = calibration
        self.bus.save_calibration(self.calib_file)

        print(f"\n[OK] 캘리브레이션 완료!")
        print(f"[Save] 저장: {self.calib_file}\n")

    # ========================================================================
    # 핵심 메서드: get_observation, send_action
    # ========================================================================

    def get_observation(self, as_dataclass: bool = False) -> dict[str, Any] | RobotObservation:
        """
        현재 상태 읽기

        Args:
            as_dataclass: True면 RobotObservation 반환 (LeRobot 호환)

        Returns:
            dict 또는 RobotObservation

        사용 예:
            # dict 형식 (기본)
            obs = robot.get_observation()
            # {"shoulder_pan.pos": 0.5, "shoulder_lift.pos": -0.3, ...}

            # RobotObservation 형식 (LeRobot 호환)
            obs = robot.get_observation(as_dataclass=True)
            # RobotObservation(state={...}, images={...})
        """
        obs = {}

        # ====================================================================
        # 모터 위치 읽기
        # ====================================================================
        # DynamixelMotorsBus.sync_read()를 사용합니다.
        #
        # sync_read("Present_Position")는:
        #   1. GroupSyncRead로 모든 모터 위치 읽기
        #   2. raw 값을 정규화된 값으로 변환 (-100~100 또는 0~100)
        #   3. 딕셔너리 반환: {"motor_name": normalized_value}
        positions = self.bus.sync_read("Present_Position")
        for motor_name, pos in positions.items():
            # 모터 ID 확인
            motor = self.bus.motors[motor_name]
            
            # 기어비가 설정된 모터라면, 값을 나누어 '관절 각도(출력축)' 기준으로 변환
            if motor.id in self.gear_ratios:
                ratio = self.gear_ratios[motor.id]
                # 예: 모터가 200만큼 돌았으면, 실제 관절은 100만큼 돈 것
                pos = pos / ratio
            
            obs[f"{motor_name}.pos"] = pos

        # ====================================================================
        # 카메라 이미지 읽기
        # ====================================================================
        # OpenCVCamera.read()를 사용합니다.
        #
        # read()는:
        #   1. cv2.VideoCapture.read()로 이미지 캡처
        #   2. BGR → RGB 변환
        #   3. numpy array 반환: (H, W, 3), dtype=uint8

        if self.camera and self.camera.is_connected:
            obs["camera"] = self.camera.read()

        # LeRobot 호환 형식 변환
        if as_dataclass:
            return dict_to_observation(obs)
        return obs

    def send_action(self, action: dict[str, Any] | RobotAction) -> dict[str, Any]:
        """
        동작 명령 전송

        Args:
            action: dict 또는 RobotAction

        Returns:
            실제 전송된 액션 (dict)

        사용 예:
            # dict 형식
            robot.send_action({"shoulder_pan.pos": 10.0, "shoulder_lift.pos": -5.0})

            # RobotAction 형식 (LeRobot 호환)
            action = RobotAction(positions={"shoulder_pan.pos": 10.0})
            robot.send_action(action)
        """
        if self.is_leader:
            logger.warning("Leader 모드에서는 send_action을 사용할 수 없습니다")
            return action if isinstance(action, dict) else action.to_dict()

        # RobotAction을 dict로 변환
        if isinstance(action, RobotAction):
            action = action.to_dict()

        # ====================================================================
        # ".pos" 접미사 제거
        # ====================================================================
        # 입력: {"shoulder_pan.pos": 10.0, "shoulder_lift.pos": -5.0}
        # 출력: {"shoulder_pan": 10.0, "shoulder_lift": -5.0}

        positions = {}
        for key, value in action.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                
                # 모터 ID 확인
                if motor_name in self.bus.motors:
                    motor = self.bus.motors[motor_name]
                    
                    # 기어비가 설정된 모터라면, 값을 곱해서 '모터 회전량'으로 변환
                    if motor.id in self.gear_ratios:
                        ratio = self.gear_ratios[motor.id]
                        # 예: 관절을 100만큼 돌리려면, 모터는 200만큼 돌려야 함
                        value = value * ratio
                    
                    positions[motor_name] = value

        # ====================================================================
        # DynamixelMotorsBus로 전송
        # ====================================================================
        # DynamixelMotorsBus.sync_write()를 사용합니다.
        #
        # sync_write("Goal_Position", positions)는:
        #   1. 정규화된 값을 raw 값으로 변환
        #   2. GroupSyncWrite로 모든 모터에 동시 전송
        #   3. 모터가 실제로 움직입니다!

        self.bus.sync_write("Goal_Position", positions)

        return action

    @property
    def action_features(self) -> list[str]:
        """액션 feature 이름들 (LeRobot 호환)"""
        return [f"{name}.pos" for name in self.bus.motors.keys()]

    @property
    def observation_features(self) -> list[str]:
        """관측 feature 이름들 (LeRobot 호환)"""
        features = [f"{name}.pos" for name in self.bus.motors.keys()]
        if self.camera:
            features.append("camera")
        return features

    # ========================================================================
    # 추가 메서드: LeRobot 호환 및 상세 정보 출력
    # ========================================================================

    def print_observation(self, obs: dict[str, Any] | RobotObservation):
        """
        관측 데이터를 LeRobot 스타일로 출력

        Args:
            obs: dict 또는 RobotObservation

        출력 예시:
            ------------------
            NAME             | VALUE
            ------------------
            shoulder_pan.pos |   10.50
            shoulder_lift.pos|  -20.30
            ...
        """
        import numpy as np

        # RobotObservation을 dict로 변환
        if isinstance(obs, RobotObservation):
            obs = obs.to_dict()

        # 최대 이름 길이 계산 (이미지 제외)
        motor_keys = [k for k, v in obs.items() if isinstance(v, (int, float))]
        if not motor_keys:
            return

        display_len = max(len(key) for key in motor_keys)

        print("\n" + "-" * (display_len + 12))
        print(f"{'NAME':<{display_len}} | {'VALUE':>7}")
        print("-" * (display_len + 12))

        # 모터 상태만 출력 (이미지는 제외)
        for key in sorted(motor_keys):
            value = obs[key]
            print(f"{key:<{display_len}} | {value:>7.2f}")

        # 이미지 정보
        image_keys = [k for k, v in obs.items() if isinstance(v, np.ndarray)]
        if image_keys:
            print(f"\nImages: {image_keys}")

    def print_action(self, action: dict[str, Any] | RobotAction):
        """
        액션 데이터를 LeRobot 스타일로 출력

        Args:
            action: dict 또는 RobotAction

        출력 예시:
            ------------------
            NAME             |    NORM
            ------------------
            shoulder_pan.pos |   10.50
            ...
        """
        # RobotAction을 dict로 변환
        if isinstance(action, RobotAction):
            action = action.to_dict()

        # 최대 이름 길이 계산
        action_keys = [k for k, v in action.items() if isinstance(v, (int, float))]
        if not action_keys:
            return

        display_len = max(len(key) for key in action_keys)

        print("\n" + "-" * (display_len + 12))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        print("-" * (display_len + 12))

        for key in sorted(action_keys):
            value = action[key]
            print(f"{key:<{display_len}} | {value:>7.2f}")

    def get_state_dict(self) -> dict[str, Any]:
        """
        로봇의 전체 상태를 dict로 반환

        Returns:
            {
                "robot_id": str,
                "is_leader": bool,
                "is_connected": bool,
                "motor_count": int,
                "has_camera": bool,
                "action_features": list[str],
                "observation_features": list[str],
            }
        """
        return {
            "robot_id": self.robot_id,
            "is_leader": self.is_leader,
            "is_connected": self.is_connected,
            "motor_count": len(self.bus.motors),
            "motor_names": list(self.bus.motors.keys()),
            "has_camera": self.camera is not None,
            "action_features": self.action_features,
            "observation_features": self.observation_features,
        }

    def print_robot_info(self):
        """
        로봇 정보를 출력

        출력 예시:
            ================================
            Robot: follower
            ================================
            Mode:    Follower (토크 ON)
            Motors:  6 (shoulder_pan, ...)
            Camera:  Yes
            Status:  Connected
            ================================
        """
        print("\n" + "="*50)
        print(f"Robot: {self.robot_id}")
        print("="*50)
        print(f"Mode:    {'Leader (토크 OFF)' if self.is_leader else 'Follower (토크 ON)'}")
        motor_names = list(self.bus.motors.keys())
        motor_preview = ', '.join(motor_names[:3])
        if len(motor_names) > 3:
            motor_preview += ", ..."
        print(f"Motors:  {len(self.bus.motors)} ({motor_preview})")
        print(f"Camera:  {'Yes' if self.camera else 'No'}")
        print(f"Status:  {'Connected' if self.is_connected else 'Disconnected'}")
        print("="*50 + "\n")
