"""
간단한 Policy 모델
MLP 기반 Behavior Cloning
"""

import torch
import torch.nn as nn
import numpy as np


class SimplePolicy(nn.Module):
    """
    간단한 MLP Policy

    입력: 현재 관측 (관절 위치)
    출력: 다음 액션 (목표 관절 위치)
    """

    def __init__(
        self,
        state_dim: int = 6,      # 관절 개수
        action_dim: int = 6,     # 액션 차원
        hidden_dim: int = 256,   # 은닉층 크기
        num_layers: int = 3,     # 레이어 수
    ):
        """
        Args:
            state_dim: 입력 차원 (관절 개수)
            action_dim: 출력 차원 (액션 개수)
            hidden_dim: 은닉층 크기
            num_layers: MLP 레이어 수
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # MLP 네트워크 구성
        layers = []

        # 첫 번째 레이어
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 중간 레이어들
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # 출력 레이어
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Tanh())  # -1 ~ 1 범위로 정규화

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            state: (batch, state_dim) 또는 (state_dim,)

        Returns:
            action: (batch, action_dim) 또는 (action_dim,)
        """
        return self.network(state)

    def predict(self, observation: dict[str, float]) -> dict[str, float]:
        """
        관측에서 액션 예측

        Args:
            observation: {"motor_name.pos": value, ...}

        Returns:
            action: {"motor_name.pos": value, ...}
        """
        # 관측을 텐서로 변환
        state_values = []
        motor_names = sorted([k for k in observation.keys() if k.endswith('.pos')])

        for key in motor_names:
            state_values.append(observation[key])

        state = torch.FloatTensor(state_values).unsqueeze(0)  # (1, state_dim)

        # 예측
        with torch.no_grad():
            action_tensor = self.forward(state)

        # 텐서를 딕셔너리로 변환
        action_values = action_tensor.squeeze(0).numpy()

        action = {}
        for i, key in enumerate(motor_names):
            # -1~1 범위를 -100~100으로 스케일링
            action[key] = float(action_values[i] * 100.0)

        return action

    def save(self, path: str):
        """모델 저장"""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)
        print(f"✓ 모델 저장됨: {path}")

    @classmethod
    def load(cls, path: str) -> 'SimplePolicy':
        """모델 로드"""
        checkpoint = torch.load(path)

        model = cls(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
        )
        model.load_state_dict(checkpoint['state_dict'])

        print(f"✓ 모델 로드됨: {path}")
        return model


class DatasetFromPickle(torch.utils.data.Dataset):
    """
    Pickle 파일에서 데이터셋 생성
    """

    def __init__(self, pickle_files: list[str]):
        """
        Args:
            pickle_files: Pickle 파일 경로 리스트
        """
        import pickle

        self.states = []
        self.actions = []

        # 모든 데이터 로드
        for pkl_file in pickle_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            observations = data['observations']
            actions = data['actions']

            # 관측과 액션 추출
            for obs, act in zip(observations, actions):
                # 관측에서 상태 추출 (.pos만)
                state = [v for k, v in sorted(obs.items()) if k.endswith('.pos')]
                action = [v for k, v in sorted(act.items()) if k.endswith('.pos')]

                self.states.append(state)
                self.actions.append(action)

        # NumPy 배열로 변환
        self.states = np.array(self.states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)

        # -100~100 범위를 -1~1로 정규화
        self.actions = self.actions / 100.0

        print(f"✓ 데이터셋 로드 완료: {len(self)} 샘플")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.actions[idx]),
        )
