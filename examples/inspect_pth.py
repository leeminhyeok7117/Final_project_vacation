import torch
import os
import sys

# =========================================================
# 분석할 파일 경로 (사용자 환경에 맞춤)
# =========================================================
CHECKPOINT_PATH = "examples/models/latest_policy.pth"

def inspect_checkpoint(path):
    print(f"[{path}] 분석 시작...\n")
    
    if not os.path.exists(path):
        print(f"[Error] 파일을 찾을 수 없습니다: {path}")
        return

    try:
        # CPU로 안전하게 로드
        ckpt = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"[Error] 로드 실패: {e}")
        return

    # 1. 최상위 키 확인
    print(f"📌 최상위 키(Keys): {list(ckpt.keys())}")
    
    # state_dict 추출
    if 'model' in ckpt:
        state_dict = ckpt['model']
        print("   -> 'model' 키 안에서 가중치를 발견했습니다.")
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print("   -> 'state_dict' 키 안에서 가중치를 발견했습니다.")
    else:
        state_dict = ckpt
        print("   -> 파일 자체가 가중치 딕셔너리(state_dict)입니다.")

    print("-" * 50)
    
    # 2. 핵심 구조 분석 (탐정 모드)
    print("🔍 [구조 정밀 분석 Results]")
    
    keys = list(state_dict.keys())
    
    # (1) 백본 확인 (CNN vs ResNet)
    has_backbone = any("vision.backbone" in k for k in keys)
    has_simple_net = any("vision.net" in k for k in keys)
    
    if has_backbone:
        print("✅ Vision Type: [ResNet18] (vision.backbone 발견됨)")
    elif has_simple_net:
        print("✅ Vision Type: [SimpleCNN] (vision.net 발견됨)")
    else:
        print("❓ Vision Type: 알 수 없음 (vision 키를 찾을 수 없음)")

    # (2) 트랜스포머 크기 확인 (Feedforward Dimension)
    # 보통 encoder.layers.0.linear1.weight 모양을 보면 알 수 있음
    # shape: (dim_feedforward, hidden_dim) -> 예: (1024, 256) 또는 (512, 256)
    ff_layer_key = "encoder.layers.0.linear1.weight"
    if ff_layer_key in state_dict:
        shape = state_dict[ff_layer_key].shape
        dim_ff = shape[0]
        hidden_dim = shape[1]
        print(f"✅ Feedforward Dim: [{dim_ff}] (기본값 512 vs 1024 확인용)")
        print(f"✅ Hidden Dim:      [{hidden_dim}]")
    else:
        print("⚠️ Transformer Layer 정보를 찾을 수 없습니다.")

    print("-" * 50)

    # 3. 레이어 요약 출력 (처음 20개만)
    print("📋 [저장된 레이어 목록 (상위 20개)]")
    for i, (k, v) in enumerate(state_dict.items()):
        if i >= 20: 
            print("... (생략) ...")
            break
        print(f" - {k:<50} | Shape: {tuple(v.shape)}")

if __name__ == "__main__":
    inspect_checkpoint(CHECKPOINT_PATH)