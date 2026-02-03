"""
DynamixelSDK 기반 모터 버스
Protocol 1.0 (AX-12A) 및 Protocol 2.0 (XL/XM) 지원
"""

import struct
import json
import logging
from pathlib import Path
from typing import Any

try:
    import dynamixel_sdk as dxl
except ImportError:
    raise ImportError(
        "dynamixel_sdk가 필요합니다. 설치: pip install dynamixel_sdk"
    )

from .motor_core import Motor, MotorCalibration, MotorNormMode

logger = logging.getLogger(__name__)


# ========================================
# Control Tables
# ========================================

# AX-12A (Protocol 1.0)
AX12_CTRL_TABLE = {
    "Torque_Enable": (24, 1),
    "Goal_Position": (30, 2),
    "Moving_Speed": (32, 2),
    "Present_Position": (36, 2),
    "Present_Speed": (38, 2),
    "Present_Load": (40, 2),
}

# XL/XM Series (Protocol 2.0)
X_SERIES_CTRL_TABLE = {
    "Torque_Enable": (64, 1),
    "Goal_Position": (116, 4),
    "Present_Position": (132, 4),
    "Present_Velocity": (128, 4),
    "Present_Load": (126, 2),
}

class DynamixelMotorsBus:
    def __init__(self, port: str, motors: dict[str, Motor], baudrate: int = 1000000, calibration: dict[str, MotorCalibration] | None = None):
        self.port = port
        self.motors = motors
        self.baudrate = baudrate
        self.calibration = calibration or {}
        self.is_calibrated = len(self.calibration) > 0

        # 핸들러 설정
        self.port_handler = dxl.PortHandler(self.port)
        self.packet_handler_1 = dxl.PacketHandler(1.0)
        self.packet_handler_2 = dxl.PacketHandler(2.0)
        
        self.is_connected = False
        self.last_raw_positions = {}

    def get_target_info(self, motor_id: int):
        """모터 객체의 모델 설정을 확인하여 프로토콜 및 테이블 결정"""
        # 1. ID에 해당하는 모터 객체를 찾습니다.
        target_motor = None
        for motor in self.motors.values():
            if motor.id == motor_id:
                target_motor = motor
                break
                
        # 2. 모터 모델 이름에 'xl'이나 'xm'이 포함되어 있으면 Protocol 2.0 적용
        if target_motor and ("xl" in target_motor.model.lower() or "xm" in target_motor.model.lower()):
            return self.packet_handler_2, X_SERIES_CTRL_TABLE, 4096, 2048
        
        # 3. 그 외(ID 5, 15 포함 기본값) 또는 모델명이 'ax'이면 Protocol 1.0 적용
        return self.packet_handler_1, AX12_CTRL_TABLE, 1024, 512
   
    def connect(self):
        if not self.port_handler.openPort(): raise RuntimeError(f"포트 실패: {self.port}")
        if not self.port_handler.setBaudRate(self.baudrate): raise RuntimeError("Baudrate 실패")
        self.is_connected = True
        logger.info(f"Dynamixel Bus 연결 완료 (Port: {self.port})")

    def disconnect(self, disable_torque=True):
        if self.is_connected:
            if disable_torque: self.disable_torque()
            self.port_handler.closePort()
            self.is_connected = False
    
    def ping(self, motor_id: int) -> bool:
        """모터 핑 테스트"""
        if not self.is_connected:
            return False

        model_number, result, error = self.packet_handler.ping(self.port_handler, motor_id)
        return result == dxl.COMM_SUCCESS

    # ========================================
    # Torque Control
    # ========================================

    def enable_torque(self, motor_names=None):
        target = motor_names or list(self.motors.keys())
        for n in target:
            m = self.motors[n]
            h, t, _, _ = self.get_target_info(m.id)
            h.write1ByteTxRx(self.port_handler, m.id, t["Torque_Enable"][0], 1)

    def disable_torque(self, motor_names=None):
        target = motor_names or list(self.motors.keys())
        for n in target:
            m = self.motors[n]
            h, t, _, _ = self.get_target_info(m.id)
            h.write1ByteTxRx(self.port_handler, m.id, t["Torque_Enable"][0], 0)

    # ========================================
    # Position Read/Write
    # ========================================

    def get_raw_position(self, motor_id: int) -> int:
        """단일 모터의 현재 Raw 위치(0~1023 또는 0~4095)를 읽어옵니다."""
        handler, table, _, _ = self.get_target_info(motor_id)
        addr, size = table["Present_Position"]

        if size == 4:
            val, res, err = handler.read4ByteTxRx(self.port_handler, motor_id, addr)
            if res == dxl.COMM_SUCCESS:
                val = struct.unpack('i', struct.pack('I', val & 0xFFFFFFFF))[0]
        else:
            val, res, err = handler.read2ByteTxRx(self.port_handler, motor_id, addr)

        if res != dxl.COMM_SUCCESS:
            logger.error(f"ID {motor_id} 위치 읽기 실패: {res}")
            return 0
        
        return val
    
    def sync_read(self, register_name: str) -> dict[str, float]:
        positions = {}
        for name, motor in self.motors.items():
            handler, table, res, center = self.get_target_info(motor.id)
            addr, size = table[register_name]

            if size == 4:
                val, res_code, err = handler.read4ByteTxRx(self.port_handler, motor.id, addr)
                if res_code == dxl.COMM_SUCCESS:
                    val = struct.unpack('i', struct.pack('I', val & 0xFFFFFFFF))[0]
            else:
                val, res_code, err = handler.read2ByteTxRx(self.port_handler, motor.id, addr)

            if res_code == dxl.COMM_SUCCESS:
                self.last_raw_positions[name] = val
            else:
                val = self.last_raw_positions.get(name, center)
            
            positions[name] = self.normalize(name, val)
        return positions

    def sync_write(self, register_name: str, values: dict[str, float]):
        for name, norm_value in values.items():
            if name not in self.motors: continue
            motor = self.motors[name]
            handler, table, _, _ = self.get_target_info(motor.id)
            addr, size = table[register_name]
            
            raw_value = self.denormalize(name, norm_value)

            if size == 4:
                handler.write4ByteTxRx(self.port_handler, motor.id, addr, raw_value)
            else:
                handler.write2ByteTxRx(self.port_handler, motor.id, addr, raw_value)

    # ========================================
    # Normalization
    # ========================================

    def normalize(self, motor_name: str, raw_value: int) -> float:
        motor = self.motors[motor_name]
        calib = self.calibration.get(motor_name)
        _, _, resolution, center = self.get_target_info(motor.id)

        if motor.norm_mode == MotorNormMode.RANGE_M100_100:
            if calib:
                range_span = calib.range_max - calib.range_min
                normalized = ((raw_value - calib.range_min) / range_span) * 200 - 100
            else:
                normalized = ((raw_value - center) / resolution) * 200
            return float(normalized)

        elif motor.norm_mode == MotorNormMode.RANGE_0_100:
            if calib:
                range_span = calib.range_max - calib.range_min
                normalized = ((raw_value - calib.range_min) / range_span) * 100
            else:
                normalized = (raw_value / resolution) * 100
            return float(normalized)

        elif motor.norm_mode == MotorNormMode.DEGREES:
            if calib:
                range_span = calib.range_max - calib.range_min
                normalized = ((raw_value - calib.range_min) / range_span) * 300 - 150
            else:
                normalized = (raw_value / resolution) * 300 - 150
            return float(normalized)

        return float(raw_value)

    def denormalize(self, motor_name: str, norm_value: float) -> int:
        motor = self.motors[motor_name]
        calib = self.calibration.get(motor_name)
        _, _, resolution, center = self.get_target_info(motor.id)

        if motor.norm_mode == MotorNormMode.RANGE_M100_100:
            if calib:
                range_span = calib.range_max - calib.range_min
                raw_value = int(((norm_value + 100) / 200) * range_span + calib.range_min)
            else:
                raw_value = int((norm_value / 200) * resolution + center)
            # return max(0, min(resolution - 1, raw_value))
            return int(raw_value)
        
        elif motor.norm_mode == MotorNormMode.RANGE_0_100:
            if calib:
                range_span = calib.range_max - calib.range_min
                raw_value = int((norm_value / 100) * range_span + calib.range_min)
            else:
                raw_value = int((norm_value / 100) * resolution)
            # return max(0, min(resolution - 1, raw_value))
            return int(raw_value)
        
        elif motor.norm_mode == MotorNormMode.DEGREES:
            if calib:
                range_span = calib.range_max - calib.range_min
                raw_value = int(((norm_value + 150) / 300) * range_span + calib.range_min)
            else:
                raw_value = int(((norm_value + 150) / 300) * resolution)
            # return max(0, min(resolution - 1, raw_value))
            return int(raw_value)
        
        return int(norm_value)

    # ========================================
    # Calibration
    # ========================================

    def save_calibration(self, filepath: Path):
        """캘리브레이션 저장"""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        calib_data = {}
        for name, calib in self.calibration.items():
            calib_data[name] = {
                "id": calib.id,
                "drive_mode": calib.drive_mode,
                "homing_offset": calib.homing_offset,
                "range_min": calib.range_min,
                "range_max": calib.range_max,
            }

        with open(filepath, "w") as f:
            json.dump(calib_data, f, indent=2)

        logger.info(f"캘리브레이션 저장: {filepath}")

    def load_calibration(self, filepath: Path):
        """캘리브레이션 로드"""
        if not filepath.exists():
            logger.warning(f"캘리브레이션 파일 없음: {filepath}")
            return

        with open(filepath, "r") as f:
            calib_data = json.load(f)

        self.calibration = {}
        for name, data in calib_data.items():
            self.calibration[name] = MotorCalibration(
                id=data["id"],
                drive_mode=data["drive_mode"],
                homing_offset=data["homing_offset"],
                range_min=data["range_min"],
                range_max=data["range_max"],
            )

        logger.info(f"캘리브레이션 로드: {filepath}")

    # ========================================
    # Low-level helpers
    # ========================================

    def _write_byte(self, motor_id: int, register_name: str, value: int):
        """1바이트 쓰기"""
        addr, size = self.ctrl_table[register_name]

        if size == 1:
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, addr, value
            )
        else:
            raise ValueError(f"_write_byte는 1바이트 레지스터만 지원: {register_name}")

        if result != dxl.COMM_SUCCESS:
            logger.error(f"Write 실패: {self.packet_handler.getTxRxResult(result)}")

    def __del__(self):
        """소멸자"""
        if self.is_connected:
            self.disconnect()