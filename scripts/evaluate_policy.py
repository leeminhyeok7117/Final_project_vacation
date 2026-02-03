#!/usr/bin/env python
"""
Policy 평가 스크립트
시뮬레이션 또는 실제 로봇에서 테스트
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from policy import SimplePolicy
from simulation import SimpleSimEnv


def evaluate_in_sim():
    """시뮬레이션에서 평가"""

    # ========================================================================
    # 설정
    # ========================================================================
    # 프로젝트 루트 경로
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_PATH = PROJECT_ROOT / "examples" / "models" / "best_policy.pth"
    NUM_EPISODES = 10
    RENDER = True

    if not MODEL_PATH.exists():
        print(f"❌ 모델 파일 없음: {MODEL_PATH}")
        print(f"   먼저 학습하세요: python scripts/train_policy.py")
        return

    print("\n" + "="*70)
    print("  Policy 평가 (시뮬레이션)")
    print("="*70)
    print(f"  모델: {MODEL_PATH}")
    print(f"  에피소드: {NUM_EPISODES}")
    print("="*70 + "\n")

    # ========================================================================
    # 모델 로드
    # ========================================================================
    print("🧠 모델 로드 중...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePolicy.load(str(MODEL_PATH))
    model = model.to(device)
    model.eval()

    print(f"   디바이스: {device}\n")

    # ========================================================================
    # 환경 초기화
    # ========================================================================
    env = SimpleSimEnv(num_joints=6, max_steps=200)

    # ========================================================================
    # 평가 루프
    # ========================================================================
    print("🎮 평가 시작...\n")

    successes = 0
    total_rewards = []

    for episode in range(NUM_EPISODES):
        obs = env.reset(randomize=True)
        done = False
        episode_reward = 0
        step = 0

        while not done:
            # Policy로 액션 예측
            action = model.predict(obs)

            # 환경 스텝
            obs, reward, done, info = env.step(action)

            episode_reward += reward
            step += 1

        # 결과
        success = info['success']
        if success:
            successes += 1

        total_rewards.append(episode_reward)

        status = "✅ 성공" if success else "❌ 실패"
        print(f"Episode {episode+1:2d}/{NUM_EPISODES} | "
              f"{status} | "
              f"Steps: {step:3d} | "
              f"Reward: {episode_reward:6.2f} | "
              f"Distance: {info['distance']:.2f}")

        # 마지막 에피소드 렌더링
        if RENDER and episode == NUM_EPISODES - 1:
            save_path = Path.home() / "robot_results" / "last_episode.png"
            save_path.parent.mkdir(exist_ok=True)
            env.render(save_path=str(save_path))

    # ========================================================================
    # 통계
    # ========================================================================
    success_rate = successes / NUM_EPISODES * 100
    avg_reward = sum(total_rewards) / len(total_rewards)

    print("\n" + "="*70)
    print("  📊 평가 결과")
    print("="*70)
    print(f"   성공률: {success_rate:.1f}% ({successes}/{NUM_EPISODES})")
    print(f"   평균 보상: {avg_reward:.2f}")
    print("="*70 + "\n")


def evaluate_on_robot():
    """실제 로봇에서 평가"""

    # 프로젝트 루트 경로
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_PATH = PROJECT_ROOT / "examples" / "models" / "best_policy.pth"
    ROBOT_PORT = "/dev/ttyUSB0"
    MOTOR_MODEL = "ax-12a"

    if not MODEL_PATH.exists():
        print(f"❌ 모델 파일 없음: {MODEL_PATH}")
        return

    print("\n" + "="*70)
    print("  Policy 평가 (실제 로봇)")
    print("="*70)
    print(f"  모델: {MODEL_PATH}")
    print(f"  로봇: {ROBOT_PORT}")
    print("="*70 + "\n")

    # 모델 로드
    from policy import SimplePolicy
    model = SimplePolicy.load(str(MODEL_PATH))
    model.eval()

    # 로봇 연결
    from robots import SimpleRobot
    robot = SimpleRobot(
        port=ROBOT_PORT,
        motor_model=MOTOR_MODEL,
        robot_id="eval_robot",
    )

    try:
        robot.connect()

        print("\n🤖 로봇으로 Policy 실행 중...")
        print("   Ctrl+C로 종료\n")

        import time

        while True:
            # 현재 상태 읽기
            obs = robot.get_observation()

            # Policy로 액션 예측
            action = model.predict(obs)

            # 로봇에 전송
            robot.send_action(action)

            time.sleep(0.033)  # 30Hz

    except KeyboardInterrupt:
        print("\n\n⏸️  종료됨")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "robot":
        # 실제 로봇 평가
        evaluate_on_robot()
    else:
        # 시뮬레이션 평가 (기본)
        evaluate_in_sim()
