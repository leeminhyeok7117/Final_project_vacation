"""간소화된 OpenCV 카메라"""
import cv2
import numpy as np


class OpenCVCamera:
    """OpenCV 기반 카메라"""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Args:
            index: 카메라 인덱스 (0, 1, 2, ...)
            width: 이미지 너비
            height: 이미지 높이
            fps: 프레임률
        """
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_connected = False

    def connect(self):
        """카메라 연결"""
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 열기 실패: {self.index}")

        self.is_connected = True

    def disconnect(self):
        """카메라 연결 해제"""
        if self.cap:
            self.cap.release()
        self.is_connected = False

    def read(self) -> np.ndarray:
        """
        이미지 읽기

        Returns:
            numpy array (height, width, 3) - RGB 이미지
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("카메라 프레임 읽기 실패")

        # BGR -> RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def async_read(self) -> np.ndarray:
        """비동기 읽기 (간단한 버전은 동기와 동일)"""
        return self.read()
