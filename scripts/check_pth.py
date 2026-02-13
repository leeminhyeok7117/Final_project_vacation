import torch
import numpy as np

# ==========================================
# 확인하고 싶은 모델 경로를 입력하세요
# ==========================================
CKPT_PATH = "./examples/models/checkpoint_epoch_200.pth"  # 경로 수정 필요

def check_checkpoint():
    print(f"[INFO] 모델 파일 분석 중: {CKPT_PATH}")
    try:
        # CPU로 로드
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
    except FileNotFoundError:
        print("[Error] 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 1. 저장된 키 확인
    print(f"\n[1] 파일 내부 키 목록: {list(ckpt.keys())}")

    # 2. Config 확인 (이미지 정규화 정보 찾기)
    if 'config' in ckpt:
        print("\n[2] 학습 설정 (Config) 요약:")
        cfg = ckpt['config']
        # 딕셔너리인지 객체인지 확인 후 처리
        if isinstance(cfg, dict):
            print(f"   - dataset_stats 사용 여부: {cfg.get('use_dataset_stats', '정보 없음')}")
            print(f"   - normalization_mapping: {cfg.get('normalization_mapping', '정보 없음')}")
            print(f"   - image features: {cfg.get('visual_features', '정보 없음')}")
        else:
            print(f"   - Config 내용: {cfg}")
    else:
        print("\n[2] 'config' 키가 없습니다. (매우 구버전이거나 다른 포맷)")

    # 3. Stats 확인 (관절 값 정규화 정보)
    if 'stats' in ckpt:
        print("\n[3] 통계치 (Stats) 확인:")
        stats = ckpt['stats']
        print(f"   - 포함된 통계 키: {list(stats.keys())}")
        
        # qpos_mean / qpos_std 확인
        if 'qpos_mean' in stats:
            print(f"   - qpos_mean (관절 평균): {stats['qpos_mean']}")
        if 'qpos_std' in stats:
            print(f"   - qpos_std  (관절 표준편차): {stats['qpos_std']}")
        if 'min' in stats:
             print(f"   - min (최솟값): {stats['min']}")
        if 'max' in stats:
             print(f"   - max (최댓값): {stats['max']}")
    else:
        print("\n[3] 'stats' 키가 없습니다. (모델이 정규화 정보를 모름!)")

    # 4. 이미지 정규화 추론
    print("\n[4] 결론 및 진단:")
    if 'config' in ckpt and isinstance(ckpt['config'], dict):
        if ckpt['config'].get('pretrained_backbone_weights') and 'ImageNet' in str(ckpt['config']):
             print("   -> 이 모델은 'ImageNet' 통계(mean=[0.485...])를 사용했을 가능성이 높습니다.")
             print("   -> 현재 코드의 하드코딩된 값과 일치할 것입니다.")
        else:
             print("   -> 학습 데이터셋 자체의 픽셀 평균/분산을 썼을 수도 있습니다.")
             print("   -> 만약 그렇다면 코드의 transforms.Normalize 값을 수정해야 합니다.")

if __name__ == "__main__":
    check_checkpoint()