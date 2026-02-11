import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys
from pathlib import Path
from torchvision import transforms

# =========================================================
# 1. 설정값 (본인의 환경에 맞게 수정하세요)
# =========================================================
MODEL_FILE_NAME = "checkpoint_epoch_200.pth"
# 파일이 위치한 실제 경로로 수정
MODEL_PATH = Path("./examples/models") / MODEL_FILE_NAME
TEST_IMAGE_PATH = "test_input.jpg"  # 테스트하고 싶은 사진 경로

# 모델 구조 설정 (학습 시 설정과 동일해야 함)
STATE_DIM = 6
ACTION_DIM = 6
CHUNK_SIZE = 8
D_MODEL = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 2. 이미지 복원 함수 (텐서 -> 시각화 이미지)
# =========================================================
def denormalize_for_vis(tensor):
    """모델에 입력되는 정규화된 텐서를 인간이 볼 수 있는 BGR 이미지로 변환"""
    img = tensor.detach().cpu().numpy().squeeze()
    img = img.transpose(1, 2, 0) # (H, W, C)

    # ImageNet 정규화 역산
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std) + mean
    
    img = np.clip(img, 0, 1) * 255
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

# =========================================================
# 3. 실행부
# =========================================================
def main():
    if not MODEL_PATH.exists():
        print(f"[Error] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return

    # 1. 이미지 로드
    raw_img = cv2.imread(TEST_IMAGE_PATH)
    if raw_img is None:
        print(f"[Error] 이미지를 불러올 수 없습니다: {TEST_IMAGE_PATH}")
        # 테스트용 더미 이미지 생성 (파일이 없을 경우 대비)
        raw_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(raw_img, "No Image Found", (150, 240), 1, 2, (255, 255, 255), 2)

    # 2. 모델 입력용 전처리 수행
    # OpenCV(BGR) -> RGB 변환
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # 리사이즈 (모델 입력 크기인 128x128로)
    img_t = F.interpolate(img_t.unsqueeze(0), size=(128, 128), mode="bilinear").squeeze(0)
    
    # 정규화 적용
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_t_final = normalize(img_t)

    # 3. 모델이 실제로 보는 모습 복원
    model_view = denormalize_for_vis(img_t_final)
    # 비교를 위해 원본도 128x128로 리사이즈
    resized_raw = cv2.resize(raw_img, (128, 128))

    # 4. 화면 표시
    # 왼쪽: 카메라 원본(리사이즈), 오른쪽: 모델이 정규화까지 마친 최종 입력
    combined = np.hstack((resized_raw, model_view))
    
    print("\n[INFO] 시각화 완료")
    print(" - 왼쪽: 카메라 원본 (128x128 Resize)")
    print(" - 오른쪽: 모델 입력 상태 (Normalization 완료)")
    print(" 아무 키나 누르면 종료됩니다.")

    cv2.imshow("Comparison: Raw vs Model Input", cv2.resize(combined, (512, 256)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()