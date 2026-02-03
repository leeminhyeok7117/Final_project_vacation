# Minimal LeRobot

DynamixelSDK 기반 간단한 로봇 제어 및 모방학습 시스템

## 주요 기능

✅ **DynamixelSDK 직접 통합**
- Protocol 1.0 (AX-12A) 및 2.0 (XL/XM 시리즈) 지원
- GroupSyncRead/GroupSyncWrite로 효율적인 통신
- 자동 정규화/비정규화

✅ **Leader-Follower 텔레오퍼레이션**
- Leader 로봇을 손으로 움직이면 Follower가 실시간으로 따라함
- 토크 자동 제어 (Leader: OFF, Follower: ON)

✅ **URDF 파일 지원**
- URDF에서 로봇 모델 자동 로딩
- 조인트 이름과 모터 ID 매핑

✅ **데이터 수집 및 학습**
- 텔레오퍼레이션 데이터 자동 녹화
- Behavior Cloning 기반 Policy 학습
- 학습된 Policy로 로봇 제어

## 설치

```bash
# 1. DynamixelSDK 설치
pip install dynamixel-sdk

# 2. 기타 의존성
pip install torch torchvision opencv-python numpy tqdm
```

## 프로젝트 구조

```
minimal_lerobot/
├── motors/              # 모터 제어
│   ├── motor_core.py       # 모터 정의
│   └── dynamixel_bus.py    # DynamixelSDK 래퍼
├── robots/              # 로봇 구현
│   └── simple_robot.py     # 6-DOF 로봇
├── cameras/             # 카메라
│   └── opencv_camera.py    # OpenCV 카메라
├── policies/            # Policy 모델
│   └── bc_policy.py        # Behavior Cloning
├── utils/               # 유틸리티
│   └── urdf_parser.py      # URDF 파서
├── scripts/             # 실행 스크립트
│   ├── teleoperate.py      # 텔레오퍼레이션
│   ├── record_data.py      # 데이터 수집
│   ├── train_policy.py     # Policy 학습
│   └── evaluate_policy.py  # Policy 평가
└── examples/            # 예제 파일
    └── simple_6dof_robot.urdf  # URDF 예제
```

## 사용 방법

### 1️⃣ 텔레오퍼레이션 (Leader-Follower)

Leader 로봇을 손으로 움직이면 Follower가 따라합니다.

```bash
# 기본 사용
python scripts/teleoperate.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --motor-model ax-12a

# URDF 파일 사용
python scripts/teleoperate.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --motor-model xl430-w250 \
    --urdf examples/simple_6dof_robot.urdf
```

**매개변수:**
- `--leader-port`: Leader 로봇 시리얼 포트
- `--follower-port`: Follower 로봇 시리얼 포트
- `--motor-model`: 모터 모델 (`ax-12a`, `xl430-w250`, `xm430-w350` 등)
- `--fps`: 제어 주파수 (기본: 30 Hz)
- `--urdf`: URDF 파일 경로 (선택)

### 2️⃣ 데이터 수집

텔레오퍼레이션하면서 동시에 데이터를 녹화합니다.

```bash
# 데이터 녹화
python scripts/record_data.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --camera 0 \
    --output-dir ~/robot_data

# URDF 사용
python scripts/record_data.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --camera 0 \
    --urdf examples/simple_6dof_robot.urdf \
    --output-dir ~/robot_data
```

**매개변수:**
- `--camera`: 카메라 인덱스 (0, 1, 2...)
- `--output-dir`: 데이터 저장 디렉토리
- 기타는 텔레오퍼레이션과 동일

**녹화 방법:**
1. 엔터를 눌러 녹화 시작
2. Leader를 움직이면서 작업 시연
3. Ctrl+C로 녹화 종료
4. 데이터가 `~/robot_data/demo_YYYYMMDD_HHMMSS.pkl`에 저장됨

**여러 데모 녹화:**
```bash
# 데모 1
python scripts/record_data.py ...
# 데모 2
python scripts/record_data.py ...
# 데모 3
python scripts/record_data.py ...
```

### 3️⃣ Policy 학습

수집한 데이터로 Behavior Cloning Policy를 학습합니다.

```bash
python scripts/train_policy.py
```

학습된 모델이 `~/robot_models/best_policy.pth`에 저장됩니다.

### 4️⃣ Policy 평가

학습된 Policy로 로봇을 제어합니다.

```bash
# 기본 사용
python scripts/evaluate_policy.py \
    --robot-port /dev/ttyUSB0 \
    --model-path ~/robot_models/best_policy.pth \
    --camera 0

# URDF 사용
python scripts/evaluate_policy.py \
    --robot-port /dev/ttyUSB0 \
    --model-path ~/robot_models/best_policy.pth \
    --camera 0 \
    --urdf examples/simple_6dof_robot.urdf
```

## URDF 파일 사용하기

### URDF 파일 작성

`examples/simple_6dof_robot.urdf` 참고

```xml
<robot name="my_robot">
  <joint name="shoulder_pan" type="revolute">
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>
  <!-- 추가 조인트... -->
</robot>
```

### 조인트 이름과 모터 ID 매핑

URDF를 사용할 때는 조인트 이름을 Dynamixel 모터 ID에 매핑해야 합니다.

**기본 매핑** (자동 적용):
```python
{
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
```

**커스텀 매핑**이 필요하면 코드에서 직접 수정:

```python
from minimal_lerobot.robots import SimpleRobot
from minimal_lerobot.utils import create_motors_from_urdf

# 커스텀 매핑
motor_id_mapping = {
    "joint1": 1,
    "joint2": 2,
    "joint3": 3,
    # ...
}

robot = SimpleRobot(
    port="/dev/ttyUSB0",
    motor_model="ax-12a",
    urdf_path="my_robot.urdf",
    motor_id_mapping=motor_id_mapping,
)
```

## Python API 사용

### 기본 사용

```python
from minimal_lerobot.robots import SimpleRobot

# 로봇 생성
robot = SimpleRobot(
    port="/dev/ttyUSB0",
    motor_model="ax-12a",
    robot_id="my_robot",
    is_leader=False,  # Follower 모드
    camera_index=0,   # 카메라 사용
)

# 연결
robot.connect()

# 관측 읽기
obs = robot.get_observation()
# {
#     "shoulder_pan.pos": 0.5,
#     "shoulder_lift.pos": -0.3,
#     ...
#     "camera": np.ndarray (480, 640, 3)
# }

# 액션 전송
robot.send_action({
    "shoulder_pan.pos": 0.8,
    "shoulder_lift.pos": -0.2,
    # ...
})

# 연결 해제
robot.disconnect()
```

### URDF 사용

```python
from minimal_lerobot.robots import SimpleRobot

robot = SimpleRobot(
    port="/dev/ttyUSB0",
    motor_model="xl430-w250",
    robot_id="my_robot",
    urdf_path="examples/simple_6dof_robot.urdf",
    motor_id_mapping={
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "wrist_roll": 5,
        "gripper": 6,
    },
)

robot.connect()
# ...
```

### Policy 사용

```python
import torch
from minimal_lerobot.policies import BCPolicy
from minimal_lerobot.robots import SimpleRobot

# Policy 로드
policy = BCPolicy(obs_dim=6, action_dim=6)
policy.load("~/robot_models/best_policy.pth")
policy.eval()

# 로봇 생성
robot = SimpleRobot(...)
robot.connect()

# 추론 루프
while True:
    # 관측
    obs = robot.get_observation()

    # 이미지 전처리 (224x224로 리사이즈)
    image = torch.from_numpy(obs["camera"]).permute(2, 0, 1).float() / 255.0
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=(224, 224), mode="bilinear"
    )

    # 상태 (조인트 위치)
    state = torch.tensor([obs[f"{k}.pos"] for k in motor_names])

    # 예측
    action = policy.predict({
        "image": image,
        "state": state.unsqueeze(0),
    })

    # 액션 전송
    robot.send_action({
        f"{name}.pos": action[0, i].item()
        for i, name in enumerate(motor_names)
    })
```

## 캘리브레이션

처음 연결 시 캘리브레이션이 필요합니다.

```python
robot = SimpleRobot(...)
robot.connect(calibrate=True)  # 캘리브레이션 실행
```

또는 수동으로:

```python
robot.connect()
robot.calibrate()  # 대화형 캘리브레이션
```

캘리브레이션 데이터는 `~/.minimal_lerobot/calibration/{robot_id}.json`에 저장됩니다.

## 지원 모터

### Protocol 1.0
- AX-12A
- AX-18A

### Protocol 2.0
- XL430-W250
- XL330-M288
- XM430-W350
- XM540-W270

다른 모터도 추가 가능합니다. `motors/dynamixel_bus.py`의 `CTRL_TABLE` 참고.

## 트러블슈팅

### 포트 찾기

```bash
# Linux
ls /dev/ttyUSB*

# macOS
ls /dev/tty.usb*

# Windows
# 장치 관리자에서 확인 (COM1, COM2 등)
```

### 권한 오류 (Linux)

```bash
sudo usermod -a -G dialout $USER
# 로그아웃 후 다시 로그인
```

### 모터가 응답하지 않음

1. 전원 확인
2. 통신 속도 확인 (기본: 1000000 bps)
3. 모터 ID 확인
4. 케이블 연결 확인

### 캘리브레이션 재설정

```bash
rm ~/.minimal_lerobot/calibration/{robot_id}.json
```

## 고급 사용

### 커스텀 로봇

```python
from minimal_lerobot.motors import Motor, MotorNormMode, DynamixelMotorsBus

# 커스텀 모터 구성
motors = {
    "joint1": Motor(1, "ax-12a", MotorNormMode.RANGE_M100_100),
    "joint2": Motor(2, "ax-12a", MotorNormMode.RANGE_M100_100),
    "joint3": Motor(3, "xl430-w250", MotorNormMode.DEGREES),
    # ...
}

bus = DynamixelMotorsBus(
    port="/dev/ttyUSB0",
    motors=motors,
)

bus.connect()
positions = bus.sync_read("Present_Position")
bus.sync_write("Goal_Position", {"joint1": 50.0, "joint2": -30.0})
```

### 커스텀 Policy

```python
import torch.nn as nn
from minimal_lerobot.policies import BCPolicy

class MyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 커스텀 아키텍처

    def forward(self, obs):
        # 커스텀 forward
        pass
```

## 예제

더 많은 예제는 `examples/` 디렉토리 참고:

- `simple_6dof_robot.urdf`: 6-DOF 로봇 URDF 예제

## 라이선스

MIT License

## 기여

Pull Request 환영합니다!

## 문의

Issues에 문의 남겨주세요.
