#!/usr/bin/env python
"""
Policy 학습 스크립트
Behavior Cloning (BC) 방식
"""

import sys
import glob
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from policy import SimplePolicy
from policy.simple_policy import DatasetFromPickle


def train():
    """Policy 학습 메인"""

    # ========================================================================
    # 인자 파싱
    # ========================================================================
    parser = argparse.ArgumentParser(description="Policy 학습")
    parser.add_argument("--data-dir", default=None, help="데이터 디렉토리 (기본: examples/data)")
    parser.add_argument("--model-dir", default=None, help="모델 저장 디렉토리 (기본: examples/models)")
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
    parser.add_argument("--num-epochs", type=int, default=100, help="에폭 수")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="학습률")
    args = parser.parse_args()

    # ========================================================================
    # 설정
    # ========================================================================
    # 프로젝트 루트 경로
    PROJECT_ROOT = Path(__file__).parent.parent

    # 데이터 디렉토리 (기본: minimal_lerobot/examples/data)
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    else:
        DATA_DIR = PROJECT_ROOT / "examples" / "data"

    # 모델 저장 디렉토리 (기본: minimal_lerobot/examples/models)
    if args.model_dir:
        MODEL_DIR = Path(args.model_dir)
    else:
        MODEL_DIR = PROJECT_ROOT / "examples" / "models"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 하이퍼파라미터
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    STATE_DIM = 6  # 관절 개수
    ACTION_DIM = 6

    print("\n" + "="*70)
    print("  Policy 학습 (Behavior Cloning)")
    print("="*70)
    print(f"  데이터: {DATA_DIR.absolute()}")
    print(f"  모델: {MODEL_DIR.absolute()}")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  에폭: {NUM_EPOCHS}")
    print(f"  학습률: {LEARNING_RATE}")
    print("="*70 + "\n")

    # ========================================================================
    # 데이터 로드
    # ========================================================================
    print("[Load] 데이터 로드 중...")

    # 모든 pickle 파일 찾기
    pkl_files = glob.glob(str(DATA_DIR / "demo_*.pkl"))

    if not pkl_files:
        print(f"[ERROR] 데이터 파일 없음: {DATA_DIR}")
        print(f"        먼저 데이터를 녹화하세요: python scripts/record_data.py")
        return

    print(f"   발견된 파일: {len(pkl_files)}개")

    # 데이터셋 생성
    dataset = DatasetFromPickle(pkl_files)

    # Train/Val 분할 (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"   학습 샘플: {len(train_dataset)}")
    print(f"   검증 샘플: {len(val_dataset)}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================================
    # 모델 초기화
    # ========================================================================
    print("\n[Model] 모델 초기화...")

    model = SimplePolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=256,
        num_layers=3,
    )

    # GPU 사용 가능하면 GPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"   디바이스: {device}")
    print(f"   파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ========================================================================
    # 학습 루프
    # ========================================================================
    print("\n" + "="*70)
    print("  학습 시작")
    print("="*70 + "\n")

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # ====================================================================
        # 학습
        # ====================================================================
        model.train()
        train_loss = 0.0

        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            # Forward
            pred_actions = model(states)
            loss = criterion(pred_actions, actions)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ====================================================================
        # 검증
        # ====================================================================
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device)
                actions = actions.to(device)

                pred_actions = model(states)
                loss = criterion(pred_actions, actions)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # ====================================================================
        # 로그
        # ====================================================================
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        # 베스트 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = MODEL_DIR / "best_policy.pth"
            model.save(str(model_path))
            print(f"   [Save] 베스트 모델 저장됨 (Val Loss: {val_loss:.4f})")

    # ========================================================================
    # 최종 모델 저장
    # ========================================================================
    final_model_path = MODEL_DIR / "final_policy.pth"
    model.save(str(final_model_path))

    print("\n" + "="*70)
    print("  학습 완료!")
    print("="*70)
    print(f"   베스트 모델: {MODEL_DIR}/best_policy.pth")
    print(f"   최종 모델: {MODEL_DIR}/final_policy.pth")
    print(f"   베스트 검증 손실: {best_val_loss:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    train()
