# 프로젝트 구조

```
minimal_lerobot/
├── GUIDE.md                    # 완벽한 사용 가이드 (데이터 수집 → ACT 학습)
├── README.md                   # 프로젝트 개요
├── requirements.txt            # 의존성
│
├── cameras/                    # 카메라 모듈
│   ├── opencv_camera.py        # OpenCV 카메라 (일반 USB)
│   └── realsense_camera.py     # RealSense D435 (RGB + Depth)
│
├── motors/                     # 모터 제어
│   ├── motor_core.py           # Motor, MotorNormMode, MotorCalibration
│   └── dynamixel_bus.py        # DynamixelMotorsBus (SDK 래퍼)
│
├── robots/                     # 로봇 인터페이스
│   └── simple_robot.py         # SimpleRobot (핵심 클래스)
│
├── policy/                     # Policy 모델
│   └── simple_policy.py        # SimplePolicy, DatasetFromPickle
│
├── scripts/                    # 실행 스크립트
│   ├── teleoperate.py          # Leader-Follower 텔레오퍼레이션
│   ├── record_data.py          # 데이터 녹화
│   ├── train_policy.py         # Policy 학습 (BC)
│   └── evaluate_policy.py      # Policy 평가
│
└── data_types.py               # LeRobot 호환 데이터 타입
```

## 핵심 파일 설명

### 1. SimpleRobot (robots/simple_robot.py)
로봇 하드웨어 인터페이스

**기능:**
- Dynamixel 모터 제어
- 카메라 연동 (OpenCV / RealSense)
- Leader-Follower 모드
- 관측 읽기 (get_observation)
- 액션 전송 (send_action)

**사용:**
```python
from robots import SimpleRobot

robot = SimpleRobot(
    port="/dev/ttyUSB0",
    motor_model="ax-12a",
    camera_type="realsense",
    camera_index=0,
)
robot.connect()
obs = robot.get_observation()
robot.send_action({"shoulder_pan.pos": 10.0})
```

### 2. RealSenseCamera (cameras/realsense_camera.py)
RealSense D435 카메라 제어

**기능:**
- RGB + Depth 동시 캡처
- Depth를 Color에 정렬
- Camera intrinsics

**사용:**
```python
from cameras import RealSenseCamera

camera = RealSenseCamera(width=640, height=480, fps=30)
camera.connect()

rgb = camera.read()
rgb, depth = camera.read_rgbd()
intrinsics = camera.get_intrinsics()
```

### 3. 데이터 수집 (scripts/record_data.py)

**실행:**
```bash
python scripts/record_data.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --motor-model ax-12a \
    --camera-type realsense \
    --fps 30
```

**출력:** ~/robot_data/demo_YYYYMMDD_HHMMSS.pkl

### 4. Policy 학습 (scripts/train_policy.py)

**실행:**
```bash
python scripts/train_policy.py
```

**모델 저장:** ~/robot_models/best_policy.pth

### 5. Policy 평가 (scripts/evaluate_policy.py)

**실행:**
```bash
python scripts/evaluate_policy.py \
    --model-path ~/robot_models/best_policy.pth \
    --port /dev/ttyUSB0 \
    --camera-type realsense
```

## 워크플로우

```
1. Teleoperation
   python scripts/teleoperate.py

2. Data Collection
   python scripts/record_data.py

3. Training
   python scripts/train_policy.py

4. Evaluation
   python scripts/evaluate_policy.py
```

## 주요 변경 사항

### 제거된 것:
- URDF 파일 및 파서
- old 파일들 (simple_robot_old.py, teleoperate_old.py 등)
- examples/
- utils/
- simulation/
- teleoperators/
- policies/
- 모든 이모지 (텍스트 태그로 교체)

### 추가된 것:
- RealSense D435 지원 (cameras/realsense_camera.py)
- 통합 가이드 (GUIDE.md)
- 코드 상세 설명 (주석)
- camera_type 파라미터 (opencv/realsense 선택)
