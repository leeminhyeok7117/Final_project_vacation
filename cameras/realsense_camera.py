"""
RealSense D435 카메라 구현
pyrealsense2를 사용하여 RGB + Depth 캡처
"""

import logging
import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    logging.warning("pyrealsense2가 설치되지 않았습니다. RealSense 카메라를 사용할 수 없습니다.")

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """
    RealSense D435 카메라

    특징:
    - RGB + Depth 동시 캡처
    - Depth를 Color에 정렬 (align)
    - 30fps 지원

    사용 예:
        camera = RealSenseCamera(width=640, height=480, fps=30)
        camera.connect()

        rgb_image = camera.read()  # RGB 이미지만
        rgb, depth = camera.read_rgbd()  # RGB + Depth

        camera.disconnect()
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
    ):
        """
        Args:
            width: 이미지 너비
            height: 이미지 높이
            fps: 프레임 레이트
            enable_depth: Depth 스트림 활성화 여부
        """
        if not REALSENSE_AVAILABLE:
            raise ImportError(
                "pyrealsense2가 설치되지 않았습니다.\n"
                "설치: pip install pyrealsense2"
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        # Pipeline 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Color stream 설정
        self.config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps
        )

        # Depth stream 설정 (선택사항)
        if enable_depth:
            self.config.enable_stream(
                rs.stream.depth,
                width,
                height,
                rs.format.z16,
                fps
            )

            # Depth를 Color에 정렬
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None

        self.is_connected = False

        logger.info(
            f"RealSense D435 초기화: {width}x{height} @ {fps}fps "
            f"(Depth: {enable_depth})"
        )

    def connect(self):
        """카메라 연결 시작"""
        try:
            # Pipeline 시작
            profile = self.pipeline.start(self.config)

            # 카메라 정보 출력
            device = profile.get_device()
            logger.info(f"RealSense 연결됨: {device.get_info(rs.camera_info.name)}")
            logger.info(f"Serial: {device.get_info(rs.camera_info.serial_number)}")

            # 첫 몇 프레임 skip (안정화)
            for _ in range(30):
                self.pipeline.wait_for_frames()

            self.is_connected = True
            logger.info("RealSense 카메라 준비 완료")

        except Exception as e:
            logger.error(f"RealSense 연결 실패: {e}")
            raise

    def disconnect(self):
        """카메라 연결 해제"""
        if self.is_connected:
            self.pipeline.stop()
            self.is_connected = False
            logger.info("RealSense 카메라 연결 해제됨")

    def read(self) -> np.ndarray:
        """
        RGB 이미지 읽기

        Returns:
            RGB 이미지 (H, W, 3), dtype=uint8
        """
        if not self.is_connected:
            raise RuntimeError("카메라가 연결되지 않았습니다. connect()를 먼저 호출하세요.")

        # Frames 읽기
        frames = self.pipeline.wait_for_frames()

        # Depth 정렬 (활성화된 경우)
        if self.align:
            frames = self.align.process(frames)

        # Color frame 추출
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("Color frame을 읽을 수 없습니다.")

        # numpy array로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # BGR → RGB 변환
        rgb_image = color_image[:, :, ::-1].copy()

        return rgb_image

    def read_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """
        RGB + Depth 이미지 읽기

        Returns:
            (rgb_image, depth_image)
            - rgb_image: (H, W, 3), dtype=uint8
            - depth_image: (H, W), dtype=uint16, 단위=mm
        """
        if not self.is_connected:
            raise RuntimeError("카메라가 연결되지 않았습니다. connect()를 먼저 호출하세요.")

        if not self.enable_depth:
            raise RuntimeError("Depth가 활성화되지 않았습니다.")

        # Frames 읽기
        frames = self.pipeline.wait_for_frames()

        # Depth 정렬
        aligned_frames = self.align.process(frames)

        # Color & Depth frames 추출
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Frame을 읽을 수 없습니다.")

        # numpy array로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # BGR → RGB 변환
        rgb_image = color_image[:, :, ::-1].copy()

        return rgb_image, depth_image

    def get_intrinsics(self) -> dict:
        """
        카메라 내부 파라미터 (intrinsics) 가져오기

        Returns:
            {
                "width": int,
                "height": int,
                "fx": float,  # focal length x
                "fy": float,  # focal length y
                "cx": float,  # principal point x
                "cy": float,  # principal point y
            }
        """
        if not self.is_connected:
            raise RuntimeError("카메라가 연결되지 않았습니다.")

        # Profile 가져오기
        profile = self.pipeline.get_active_profile()
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        return {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
        }

    def __enter__(self):
        """Context manager: with문 지원"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: with문 종료 시 자동 disconnect"""
        self.disconnect()
        return False


# 테스트 코드
if __name__ == "__main__":
    import cv2

    logging.basicConfig(level=logging.INFO)

    print("RealSense D435 테스트")
    print("=" * 50)

    # 카메라 초기화
    camera = RealSenseCamera(width=640, height=480, fps=30)
    camera.connect()

    # Intrinsics 출력
    intrinsics = camera.get_intrinsics()
    print(f"Intrinsics: {intrinsics}")

    # 이미지 캡처 테스트
    print("\n이미지 캡처 테스트 (ESC로 종료)")

    try:
        while True:
            # RGB + Depth 읽기
            rgb, depth = camera.read_rgbd()

            # Depth 시각화 (0-10m → 0-255)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # RGB는 BGR로 변환해서 표시
            bgr = rgb[:, :, ::-1]

            # 이미지 표시
            cv2.imshow("RGB", bgr)
            cv2.imshow("Depth", depth_colormap)

            # ESC로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        camera.disconnect()
        cv2.destroyAllWindows()
        print("\n테스트 완료")
