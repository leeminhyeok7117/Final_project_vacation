# LeRobot 모방학습 완벽 가이드

## 목차
1. [시스템 개요](#시스템-개요)
2. [데이터 수집](#데이터-수집)
3. [ACT 학습](#act-학습)
4. [Policy 평가](#policy-평가)
5. [코드 구조 설명](#코드-구조-설명)

---

## 시스템 개요

### 모방학습 워크플로우
```
1. Teleoperation (사람이 직접 로봇 조작)
   → Leader 암을 손으로 움직임
   → Follower 로봇이 따라함

2. Data Collection (데이터 수집)
   → 관측(observation): Follower의 모터 위치 + 카메라 이미지
   → 액션(action): Leader의 모터 위치
   → pickle 파일로 저장

3. Training (학습)
   → ACT Policy 학습
   → 입력: observation
   → 출력: action

4. Evaluation (평가)
   → 학습된 Policy로 로봇 제어
   → 사람 없이 자동으로 동작
```

### 필요한 하드웨어
- Leader 암 (Dynamixel 모터)
- Follower 로봇 (Dynamixel 모터)
- RealSense D435 카메라

---

## 데이터 수집

### 1단계: Teleoperation 테스트

먼저 텔레오퍼레이션이 잘 작동하는지 확인합니다.

```bash
python scripts/teleoperate.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --motor-model ax-12a \
    --fps 30
```

**동작 확인:**
- Leader 암을 손으로 움직이면 Follower가 똑같이 따라해야 함
- 문제가 있다면 캘리브레이션 확인

### 2단계: 데이터 녹화

```bash
python scripts/record_data.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --motor-model ax-12a \
    --camera realsense \
    --fps 30 \
    --output-dir ~/robot_data
```

**데이터 녹화 프로세스:**
1. 스크립트 실행
2. "엔터를 누르면 녹화 시작..." 메시지 확인
3. 준비되면 엔터 입력
4. Leader 암을 조작하여 원하는 동작 시연
5. Ctrl+C로 녹화 종료
6. 데이터 자동 저장 (`~/robot_data/demo_YYYYMMDD_HHMMSS.pkl`)

**데이터 구조:**
```python
{
    "observations": [
        {
            "shoulder_pan.pos": 0.5,
            "shoulder_lift.pos": -0.3,
            # ... 모터 위치 (6개)
            "camera": np.array([480, 640, 3], dtype=uint8)  # RGB 이미지
        },
        # ... 프레임마다 반복
    ],
    "actions": [
        {
            "shoulder_pan.pos": 0.5,
            "shoulder_lift.pos": -0.3,
            # ... Leader의 모터 위치
        },
        # ... 프레임마다 반복
    ],
    "fps": 30,
    "motor_model": "ax-12a",
    "num_frames": 150
}
```

**녹화 팁:**
- 하나의 task당 10-20회 시연 권장
- 각 시연은 5-10초 정도
- 다양한 시작 위치에서 시연
- 성공적인 시연만 저장 (실패한 시연은 삭제)

### 3단계: 데이터 확인

```bash
python scripts/visualize_data.py --data-dir ~/robot_data
```

**확인 사항:**
- 총 데이터 파일 개수
- 총 프레임 수
- 모터 위치 범위
- 카메라 이미지 품질

---

## ACT 학습

### ACT (Action Chunking with Transformers)

ACT는 Google DeepMind의 RT-1을 기반으로 한 모방학습 알고리즘입니다.

**핵심 개념:**
- Vision Encoder: 이미지를 feature vector로 변환 (ResNet 또는 ViT)
- State Encoder: 모터 위치를 embedding으로 변환
- Transformer: feature를 시퀀스로 처리
- Action Decoder: action chunk를 예측 (한 번에 여러 timestep 예측)

### 1단계: ACT Policy 학습

```bash
python scripts/train_act.py \
    --data-dir ~/robot_data \
    --model-dir ~/robot_models \
    --batch-size 8 \
    --num-epochs 500 \
    --learning-rate 1e-4 \
    --chunk-size 100 \
    --gpu 0
```

**하이퍼파라미터 설명:**

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `batch-size` | 8 | 배치 크기 (GPU 메모리에 따라 조절) |
| `num-epochs` | 500 | 학습 에폭 수 |
| `learning-rate` | 1e-4 | 학습률 (Adam optimizer) |
| `chunk-size` | 100 | action chunk 크기 (예측할 미래 timestep) |
| `hidden-dim` | 512 | Transformer hidden dimension |
| `num-layers` | 4 | Transformer layer 개수 |
| `num-heads` | 8 | Multi-head attention head 개수 |

### 2단계: 학습 모니터링

학습 중 출력 예시:
```
Epoch   1/500 | Train Loss: 2.3456 | Val Loss: 2.4567
Epoch   2/500 | Train Loss: 2.1234 | Val Loss: 2.2345
Epoch   3/500 | Train Loss: 1.9876 | Val Loss: 2.0987
...
Epoch 100/500 | Train Loss: 0.1234 | Val Loss: 0.1456
   Best model saved (Val Loss: 0.1456)
```

**학습 팁:**
- Val Loss가 더 이상 감소하지 않으면 조기 종료
- 과적합(overfitting) 징후: Train Loss는 감소하지만 Val Loss는 증가
- 해결법: 더 많은 데이터 수집, dropout 추가, weight decay 증가

### 3단계: 학습 완료

학습이 완료되면 다음 파일들이 생성됩니다:
```
~/robot_models/
├── best_act_policy.pth        # 최고 성능 모델
├── final_act_policy.pth       # 마지막 epoch 모델
├── config.json                # 모델 설정
└── training_stats.json        # 학습 통계
```

---

## Policy 평가

### 1단계: Policy로 로봇 제어

```bash
python scripts/evaluate_policy.py \
    --model-path ~/robot_models/best_act_policy.pth \
    --port /dev/ttyUSB0 \
    --motor-model ax-12a \
    --camera realsense \
    --fps 30
```

**평가 프로세스:**
1. 로봇을 초기 위치로 이동
2. Policy 로드
3. 관측(모터 위치 + 카메라) → Policy → 액션
4. 액션을 로봇에 전송
5. 반복

### 2단계: 성공률 측정

```bash
python scripts/evaluate_policy.py \
    --model-path ~/robot_models/best_act_policy.pth \
    --port /dev/ttyUSB0 \
    --motor-model ax-12a \
    --camera realsense \
    --num-trials 50 \
    --record-video
```

**성공률 기준:**
- Task 정의: 예) 물체를 A에서 B로 이동
- 성공 판정: 물체가 B에 도달하면 성공
- 성공률: 50회 시도 중 성공 횟수

---

## 코드 구조 설명

### 1. SimpleRobot 클래스 (`robots/simple_robot.py`)

**역할:** 로봇 하드웨어 인터페이스

```python
from robots import SimpleRobot

# 로봇 생성
robot = SimpleRobot(
    port="/dev/ttyUSB0",
    motor_model="ax-12a",
    robot_id="follower",
    is_leader=False,
    camera_type="realsense",
)

# 연결
robot.connect()

# 관측 읽기
obs = robot.get_observation()
# obs = {
#     "shoulder_pan.pos": 0.5,
#     "shoulder_lift.pos": -0.3,
#     ...
#     "camera": np.array([480, 640, 3])  # RealSense RGB
# }

# 액션 전송
robot.send_action({
    "shoulder_pan.pos": 10.0,
    "shoulder_lift.pos": -5.0,
    ...
})

# 연결 해제
robot.disconnect()
```

**내부 동작:**

1. **get_observation()** (Line 287-338)
   ```python
   def get_observation(self, as_dataclass=False):
       obs = {}

       # DynamixelMotorsBus.sync_read()
       # - GroupSyncRead로 모든 모터 위치 읽기
       # - raw 값을 정규화 (-100~100)
       positions = self.bus.sync_read("Present_Position")
       for motor_name, pos in positions.items():
           obs[f"{motor_name}.pos"] = pos

       # RealSense pipeline.wait_for_frames()
       # - depth + color frame 읽기
       # - align depth to color
       # - numpy array로 변환
       if self.camera:
           obs["camera"] = self.camera.read()

       return obs
   ```

2. **send_action()** (Line 340-390)
   ```python
   def send_action(self, action):
       # ".pos" 제거
       positions = {}
       for key, value in action.items():
           if key.endswith(".pos"):
               motor_name = key.removesuffix(".pos")
               positions[motor_name] = value

       # DynamixelMotorsBus.sync_write()
       # - 정규화된 값을 raw 값으로 변환
       # - GroupSyncWrite로 모든 모터에 전송
       self.bus.sync_write("Goal_Position", positions)
   ```

### 2. RealSenseCamera 클래스 (`cameras/realsense_camera.py`)

**역할:** RealSense D435 카메라 인터페이스

```python
import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # RGB stream
        self.config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, fps
        )

        # Depth stream (optional)
        self.config.enable_stream(
            rs.stream.depth, width, height, rs.format.z16, fps
        )

        # Align depth to color
        self.align = rs.align(rs.stream.color)

    def connect(self):
        self.pipeline.start(self.config)

    def read(self):
        # Wait for frames
        frames = self.pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = self.align.process(frames)

        # Get color frame
        color_frame = aligned_frames.get_color_frame()

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # BGR to RGB
        rgb_image = color_image[:, :, ::-1]

        return rgb_image

    def disconnect(self):
        self.pipeline.stop()
```

### 3. DatasetFromPickle 클래스 (`policy/simple_policy.py`)

**역할:** pickle 파일에서 데이터 로드

```python
import pickle
import torch
from torch.utils.data import Dataset

class DatasetFromPickle(Dataset):
    def __init__(self, pkl_files):
        self.observations = []
        self.actions = []

        # 모든 pickle 파일 로드
        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                self.observations.extend(data["observations"])
                self.actions.extend(data["actions"])

        print(f"Loaded {len(self.observations)} samples from {len(pkl_files)} files")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        action = self.actions[idx]

        # 모터 위치 추출
        state = []
        for key in sorted(obs.keys()):
            if key.endswith(".pos"):
                state.append(obs[key])
        state = torch.tensor(state, dtype=torch.float32)

        # 액션 추출
        action_values = []
        for key in sorted(action.keys()):
            if key.endswith(".pos"):
                action_values.append(action[key])
        action_tensor = torch.tensor(action_values, dtype=torch.float32)

        return state, action_tensor
```

### 4. ACTPolicy 클래스 (`policy/act_policy.py`)

**역할:** ACT 알고리즘 구현

```python
import torch
import torch.nn as nn

class ACTPolicy(nn.Module):
    def __init__(
        self,
        state_dim=6,
        action_dim=6,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        chunk_size=100,
    ):
        super().__init__()
        self.chunk_size = chunk_size

        # Vision encoder (ResNet-18)
        self.vision_encoder = torchvision.models.resnet18(pretrained=True)
        self.vision_encoder.fc = nn.Linear(512, hidden_dim)

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * chunk_size),
        )

    def forward(self, state, image):
        # Image encoding
        vision_feat = self.vision_encoder(image)  # [B, hidden_dim]

        # State encoding
        state_feat = self.state_encoder(state)  # [B, hidden_dim]

        # Concatenate
        feat = vision_feat + state_feat  # [B, hidden_dim]

        # Transformer
        feat = feat.unsqueeze(0)  # [1, B, hidden_dim]
        feat = self.transformer(feat)  # [1, B, hidden_dim]
        feat = feat.squeeze(0)  # [B, hidden_dim]

        # Action prediction
        action_chunk = self.action_decoder(feat)  # [B, action_dim * chunk_size]
        action_chunk = action_chunk.view(-1, self.chunk_size, action_dim)

        return action_chunk
```

**Action Chunking:**
- 한 번의 forward pass로 미래 `chunk_size` timestep의 action 예측
- 예: chunk_size=100이면 100개의 action을 한 번에 예측
- 실행 시: 예측된 action을 순차적으로 실행

### 5. 학습 루프 (`scripts/train_act.py`)

```python
# 데이터 로드
dataset = ACTDataset(pkl_files)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 모델 초기화
model = ACTPolicy(
    state_dim=6,
    action_dim=6,
    hidden_dim=512,
    chunk_size=100,
)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 학습 루프
for epoch in range(num_epochs):
    for batch in train_loader:
        states, images, actions = batch

        # Forward
        pred_actions = model(states, images)  # [B, chunk_size, action_dim]

        # Loss (첫 번째 action만 사용)
        loss = criterion(pred_actions[:, 0, :], actions)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## FAQ

**Q: 데이터를 얼마나 수집해야 하나요?**
A: 간단한 task는 10-20개 시연, 복잡한 task는 50-100개 시연 권장

**Q: GPU가 없어도 학습 가능한가요?**
A: 가능하지만 매우 느립니다. CUDA GPU 권장 (RTX 3060 이상)

**Q: ACT 대신 다른 알고리즘을 사용할 수 있나요?**
A: 네, Behavior Cloning (BC), Diffusion Policy 등 다양한 알고리즘 지원

**Q: RealSense 대신 다른 카메라를 사용할 수 있나요?**
A: 네, OpenCV 호환 카메라 모두 사용 가능

**Q: 모터가 6개가 아니면 어떻게 하나요?**
A: `robots/simple_robot.py`의 109-116줄에서 모터 정의를 수정하세요.

---

## 참고 자료

- LeRobot 공식 문서: https://github.com/huggingface/lerobot
- ACT 논문: https://arxiv.org/abs/2304.13705
- RealSense SDK: https://github.com/IntelRealSense/librealsense
- Dynamixel SDK: https://github.com/ROBOTIS-GIT/DynamixelSDK
