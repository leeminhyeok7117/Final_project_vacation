# Examples

이 디렉토리에는 학습용 예제 데이터와 학습된 모델이 저장됩니다.

## 디렉토리 구조

```
examples/
├── data/                   # 학습 데이터
│   ├── demo_001.pkl       # 합성 데이터 1 (5초, 150 프레임)
│   ├── demo_002.pkl       # 합성 데이터 2 (4초, 120 프레임)
│   └── demo_003.pkl       # 합성 데이터 3 (6초, 180 프레임)
├── models/                 # 학습된 모델
│   ├── best_policy.pth    # 검증 손실이 가장 낮은 모델
│   └── final_policy.pth   # 마지막 epoch의 모델
├── create_synthetic_data.py  # 합성 데이터 생성 스크립트
└── README.md              # 이 파일
```

## 합성 데이터

현재 포함된 데이터는 테스트용 합성 데이터입니다:

- **총 3개 데모**
- **총 450 프레임** (15초 분량, 30fps)
- **각 프레임**: 6개 모터 위치 + 480x640 RGB 이미지
- **데이터 크기**: ~396 MB

### 데이터 구조

```python
{
    "observations": [
        {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 25.0,
            "elbow_flex.pos": 43.3,
            "wrist_flex.pos": 50.0,
            "wrist_roll.pos": 43.3,
            "gripper.pos": 25.0,
            "camera": np.array([480, 640, 3], dtype=uint8)
        },
        # ... 프레임마다 반복
    ],
    "actions": [
        {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 25.0,
            # ...
        },
        # ... 프레임마다 반복
    ],
    "fps": 30,
    "motor_model": "ax-12a",
    "num_frames": 150
}
```

## 사용 방법

### 1. 합성 데이터 생성 (선택)

이미 합성 데이터가 포함되어 있지만, 새로 생성하려면:

```bash
cd /home/temp_id/minimal_lerobot
python examples/create_synthetic_data.py
```

### 2. 학습

합성 데이터로 학습:

```bash
cd /home/temp_id/minimal_lerobot
python scripts/train_policy.py
```

커스텀 설정:

```bash
python scripts/train_policy.py \
    --batch-size 16 \
    --num-epochs 50 \
    --learning-rate 0.0001
```

### 3. 실제 데이터 수집

실제 로봇으로 데이터를 수집하려면:

```bash
python scripts/record_data.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --motor-model ax-12a
```

데이터가 `examples/data/demo_YYYYMMDD_HHMMSS.pkl`에 저장됩니다.

### 4. 모델 평가

시뮬레이션에서 평가:

```bash
python scripts/evaluate_policy.py
```

실제 로봇에서 평가:

```bash
python scripts/evaluate_policy.py robot
```

## 학습 결과

현재 포함된 모델 (합성 데이터로 학습됨):

- **Best Model**: `examples/models/best_policy.pth`
- **Val Loss**: ~0.51
- **파라미터 수**: 69,126개
- **디바이스**: CUDA (GPU)

## 실제 로봇 데이터로 전환

합성 데이터로 파이프라인을 검증한 후, 실제 로봇 데이터로 전환하세요:

1. **합성 데이터 백업** (선택):
   ```bash
   mkdir -p examples/synthetic_data_backup
   mv examples/data/*.pkl examples/synthetic_data_backup/
   ```

2. **실제 데이터 수집**:
   ```bash
   python scripts/record_data.py
   ```

3. **학습**:
   ```bash
   python scripts/train_policy.py --num-epochs 100
   ```

## 주의사항

- 합성 데이터는 **테스트 및 검증 전용**입니다
- 실제 로봇 제어에는 **실제 데이터로 학습된 모델**을 사용하세요
- 데이터 파일이 크므로 Git에 커밋하지 마세요 (`.gitignore`에 추가됨)

## 데이터 삭제

필요시 데이터/모델 삭제:

```bash
# 데이터만 삭제
rm examples/data/*.pkl

# 모델만 삭제
rm examples/models/*.pth

# 전체 삭제
rm -rf examples/data examples/models
```
