"""
LeRobot 호환 데이터 타입
RobotObservation, RobotAction, EnvTransition
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class RobotObservation:
    """
    로봇 관측 데이터 (LeRobot 호환)

    Example:
        obs = RobotObservation(
            state={"shoulder_pan.pos": 0.5, "shoulder_lift.pos": -0.3},
            images={"camera": np.array(...)}
        )
    """
    state: dict[str, float] = field(default_factory=dict)  # 모터 위치, 속도 등
    images: dict[str, np.ndarray] = field(default_factory=dict)  # 카메라 이미지
    extra: dict[str, Any] = field(default_factory=dict)  # 추가 데이터

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        result = {}
        result.update(self.state)
        result.update(self.images)
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotObservation":
        """딕셔너리에서 생성"""
        state = {}
        images = {}
        extra = {}

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                images[key] = value
            elif isinstance(value, (int, float)):
                state[key] = float(value)
            else:
                extra[key] = value

        return cls(state=state, images=images, extra=extra)


@dataclass
class RobotAction:
    """
    로봇 액션 데이터 (LeRobot 호환)

    Example:
        action = RobotAction(
            positions={"shoulder_pan.pos": 0.8, "shoulder_lift.pos": -0.2}
        )
    """
    positions: dict[str, float] = field(default_factory=dict)  # 목표 위치
    velocities: dict[str, float] = field(default_factory=dict)  # 목표 속도
    efforts: dict[str, float] = field(default_factory=dict)  # 목표 힘/토크
    extra: dict[str, Any] = field(default_factory=dict)  # 추가 데이터

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        result = {}
        result.update(self.positions)
        result.update(self.velocities)
        result.update(self.efforts)
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotAction":
        """딕셔너리에서 생성"""
        positions = {}
        velocities = {}
        efforts = {}
        extra = {}

        for key, value in data.items():
            if key.endswith(".pos") or key.endswith("_position"):
                positions[key] = float(value)
            elif key.endswith(".vel") or key.endswith("_velocity"):
                velocities[key] = float(value)
            elif key.endswith(".eff") or key.endswith("_effort"):
                efforts[key] = float(value)
            else:
                extra[key] = value

        return cls(
            positions=positions,
            velocities=velocities,
            efforts=efforts,
            extra=extra,
        )


@dataclass
class EnvTransition:
    """
    환경 전환 데이터 (LeRobot 호환)
    observation + action + reward + done

    Example:
        transition = EnvTransition(
            observation=obs,
            action=action,
            reward=1.0,
            done=False,
        )
    """
    observation: RobotObservation
    action: RobotAction
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }


def dict_to_observation(data: dict[str, Any]) -> RobotObservation:
    """
    간단한 딕셔너리를 RobotObservation으로 변환

    Args:
        data: {"motor.pos": value, "camera": np.ndarray, ...}

    Returns:
        RobotObservation
    """
    return RobotObservation.from_dict(data)


def dict_to_action(data: dict[str, Any]) -> RobotAction:
    """
    간단한 딕셔너리를 RobotAction으로 변환

    Args:
        data: {"motor.pos": value, ...}

    Returns:
        RobotAction
    """
    return RobotAction.from_dict(data)


def observation_to_dict(obs: RobotObservation) -> dict[str, Any]:
    """RobotObservation을 딕셔너리로 변환"""
    return obs.to_dict()


def action_to_dict(action: RobotAction) -> dict[str, Any]:
    """RobotAction을 딕셔너리로 변환"""
    return action.to_dict()
