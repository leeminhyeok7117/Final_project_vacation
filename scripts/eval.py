# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# PKL 자동 구조 탐색 + 진단
# - demo_*.pkl 내부를 재귀 탐색해서 이미지/벡터 후보를 '키 경로(path)'로 찾아냄
# - 찾은 경로를 기반으로 RGB/BGR, state/action 범위, (가능하면) Hz까지 진단

# 실행:
# python3 eval.py --data_dir /home/lmh/minimal_lerobot/examples/data --pattern "demo_*.pkl" --max_files 5 --max_steps 200 --show 0
# """

# import argparse
# import pickle
# import glob
# from pathlib import Path
# import numpy as np

# try:
#     import cv2
# except Exception:
#     cv2 = None


# # -------------------------
# # 재귀 탐색 유틸
# # -------------------------
# def walk(obj, prefix=""):
#     """
#     obj 내부를 재귀적으로 탐색해서 (path, value) yield
#     path 예: root.steps[0].obs.image
#     """
#     yield prefix if prefix else "root", obj

#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             p = f"{prefix}.{k}" if prefix else f"root.{k}"
#             yield from walk(v, p)

#     elif isinstance(obj, (list, tuple)):
#         for i, v in enumerate(obj):
#             p = f"{prefix}[{i}]" if prefix else f"root[{i}]"
#             # 너무 깊은 대형 리스트는 앞부분만 탐색
#             if i >= 5:
#                 break
#             yield from walk(v, p)


# def is_img_candidate(x):
#     return (
#         isinstance(x, np.ndarray)
#         and x.ndim == 3
#         and x.shape[2] == 3
#         and x.dtype == np.uint8
#         and x.shape[0] >= 64
#         and x.shape[1] >= 64
#     )


# def is_vec_candidate(x, dim=6):
#     x = np.asarray(x) if isinstance(x, (list, tuple, np.ndarray)) else None
#     if x is None:
#         return False
#     if not isinstance(x, np.ndarray):
#         return False
#     if x.dtype.kind not in ("f", "i"):
#         return False
#     if x.ndim == 1 and x.shape[0] == dim:
#         return True
#     if x.ndim == 2 and x.shape[1] == dim:
#         return True
#     return False


# def get_by_path(root, path):
#     """
#     walk에서 나온 path 문자열을 실제로 따라가 value를 꺼냄
#     지원: dict key .k, list index [i]
#     """
#     if path == "root":
#         return root
#     assert path.startswith("root")
#     cur = root
#     rest = path[4:]  # remove "root"
#     i = 0
#     while i < len(rest):
#         if rest[i] == ".":
#             i += 1
#             j = i
#             while j < len(rest) and rest[j] not in ".[":
#                 j += 1
#             key = rest[i:j]
#             cur = cur[key]
#             i = j
#         elif rest[i] == "[":
#             i += 1
#             j = i
#             while j < len(rest) and rest[j] != "]":
#                 j += 1
#             idx = int(rest[i:j])
#             cur = cur[idx]
#             i = j + 1
#         else:
#             i += 1
#     return cur


# # -------------------------
# # RGB/BGR 휴리스틱(간단)
# # -------------------------
# def rgb_bgr_score(img_u8):
#     img = img_u8.astype(np.float32) / 255.0
#     c0 = img[..., 0].mean()
#     c2 = img[..., 2].mean()
#     v0 = img[..., 0].var()
#     v2 = img[..., 2].var()
#     return float((c2 - c0) + 0.5 * (v2 - v0))


# def summarize_vec(name, X):
#     X = np.asarray(X)
#     if X.ndim == 1:
#         X = X.reshape(-1, 1)
#     mn = X.min(axis=0)
#     mx = X.max(axis=0)
#     mean = X.mean(axis=0)
#     std = X.std(axis=0)
#     p01 = np.quantile(X, 0.01, axis=0)
#     p99 = np.quantile(X, 0.99, axis=0)
#     print(f"\n[{name}] shape={X.shape}")
#     print("  min :", np.round(mn, 5))
#     print("  max :", np.round(mx, 5))
#     print("  mean:", np.round(mean, 5))
#     print("  std :", np.round(std, 5))
#     print("  p01 :", np.round(p01, 5))
#     print("  p99 :", np.round(p99, 5))
#     return mn, mx


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_dir", type=str, required=True)
#     ap.add_argument("--pattern", type=str, default="demo_*.pkl")
#     ap.add_argument("--max_files", type=int, default=5)
#     ap.add_argument("--max_steps", type=int, default=200)  # step list가 있으면 앞부분만
#     ap.add_argument("--show", type=int, default=0)
#     args = ap.parse_args()

#     paths = sorted(glob.glob(str(Path(args.data_dir) / args.pattern)))
#     if len(paths) == 0:
#         raise SystemExit("매칭되는 pkl이 없습니다. --data_dir/--pattern 확인")

#     paths = paths[: args.max_files]
#     print(f"Found {len(paths)} files")

#     # 누적 통계용
#     rgb_scores = []
#     all_state = []
#     all_action = []

#     # 탐색 결과(가장 그럴듯한 경로)
#     img_paths = {}
#     vec6_paths = {}

#     for fp in paths:
#         with open(fp, "rb") as f:
#             obj = pickle.load(f)

#         # 1) 우선 구조 스냅샷: 최상위 타입/키
#         print(f"\n=== FILE: {fp} ===")
#         print("Top type:", type(obj))
#         if isinstance(obj, dict):
#             print("Top keys:", list(obj.keys())[:30])
#         elif isinstance(obj, list):
#             print("Top list len:", len(obj))
#             if len(obj) > 0:
#                 print("First elem type:", type(obj[0]))
#                 if isinstance(obj[0], dict):
#                     print("First elem keys:", list(obj[0].keys())[:30])

#         # 2) 전체 재귀 탐색(앞부분만)
#         found_imgs = []
#         found_vecs = []
#         for path, val in walk(obj):
#             if is_img_candidate(val):
#                 found_imgs.append((path, val.shape, val.dtype))
#                 img_paths[path] = img_paths.get(path, 0) + 1
#             if is_vec_candidate(val, dim=6):
#                 arr = np.asarray(val)
#                 found_vecs.append((path, arr.shape, arr.dtype))
#                 vec6_paths[path] = vec6_paths.get(path, 0) + 1

#         print("\n[Candidates] image uint8(H,W,3):")
#         for x in found_imgs[:10]:
#             print(" ", x)
#         print("[Candidates] vector dim=6:")
#         for x in found_vecs[:20]:
#             print(" ", x)

#         # 3) 후보가 있으면 실제 진단 샘플링
#         #    - 이미지: 가장 먼저 나온 후보 1개
#         if len(found_imgs) > 0:
#             p = found_imgs[0][0]
#             img = get_by_path(obj, p)
#             rgb_scores.append(rgb_bgr_score(img))
#             if args.show == 1 and cv2 is not None:
#                 cv2.imshow("raw", img)
#                 cv2.imshow("swap", img[:, :, ::-1])
#                 cv2.waitKey(500)

#         #    - vec6: 후보가 여러 개인데, state/action 구분을 위해 키 이름 기반 우선순위 부여
#         #      action 단어가 들어간 path를 action으로, qpos/state/joint가 들어간 path를 state로 우선
#         state_candidates = [p for (p, *_rest) in found_vecs if any(k in p.lower() for k in ["qpos", "state", "joint", "pos"])]
#         action_candidates = [p for (p, *_rest) in found_vecs if any(k in p.lower() for k in ["action", "act", "u", "cmd"])]

#         # fallback: 그냥 vec6 첫 번째를 state로 간주
#         if len(state_candidates) == 0 and len(found_vecs) > 0:
#             state_candidates = [found_vecs[0][0]]

#         # action fallback: state와 다른 vec6가 있으면 그걸 action으로
#         if len(action_candidates) == 0 and len(found_vecs) > 1:
#             action_candidates = [found_vecs[1][0]]

#         if len(state_candidates) > 0:
#             st = np.asarray(get_by_path(obj, state_candidates[0])).astype(np.float32)
#             st = st.reshape(-1, 6) if st.ndim == 2 else st.reshape(1, 6)
#             all_state.append(st)

#         if len(action_candidates) > 0:
#             ac = np.asarray(get_by_path(obj, action_candidates[0])).astype(np.float32)
#             ac = ac.reshape(-1, 6) if ac.ndim == 2 else ac.reshape(1, 6)
#             all_action.append(ac)

#     if args.show == 1 and cv2 is not None:
#         cv2.destroyAllWindows()

#     print("\n" + "=" * 60)
#     print("AGGREGATED DIAG RESULTS")
#     print("=" * 60)

#     # 1) RGB/BGR
#     if len(rgb_scores) > 0:
#         med = float(np.median(np.array(rgb_scores)))
#         print("\n[Color Channel Guess]")
#         print(f"  median score = {med:.6f}  ( >0: RGB likely, <0: BGR likely )")
#         if med > 0:
#             print("  -> 데이터 이미지가 RGB일 가능성이 큼. deploy에서 BGR이면 RGB 변환 필요.")
#         else:
#             print("  -> 데이터 이미지가 BGR(OpenCV)일 가능성이 큼. deploy 입력 채널 확인 필요.")
#     else:
#         print("\n[Color Channel Guess] 이미지 후보를 못 찾았습니다. (이미지가 uint8이 아닐 수도 있음)")

#     # 2) state/action 범위
#     if len(all_state) > 0:
#         st = np.concatenate(all_state, axis=0)
#         st_min, st_max = summarize_vec("STATE(guessed)", st)
#     else:
#         st = None
#         print("\n[STATE] 후보를 못 잡았습니다.")

#     if len(all_action) > 0:
#         ac = np.concatenate(all_action, axis=0)
#         ac_min, ac_max = summarize_vec("ACTION(guessed)", ac)
#     else:
#         ac = None
#         print("\n[ACTION] 후보를 못 잡았습니다.")

#     if st is not None and ac is not None:
#         st_rng = st_max - st_min
#         ac_rng = ac_max - ac_min
#         ratio = np.where(st_rng > 1e-6, ac_rng / st_rng, np.nan)
#         print("\n[State vs Action Range ratio(action/state)]", np.round(ratio, 4))

#         bad = np.sum((ratio < 0.2) | (ratio > 5.0))
#         if bad >= 2:
#             print("  -> 판정: action과 state 스케일 차이가 큼. state stats로 action 정규화하면 출력이 평균으로 수렴하기 쉬움.")
#         else:
#             print("  -> 판정: 스케일 차이가 치명적이진 않아 보임.")

#     # 3) 어떤 경로가 자주 나왔는지(포맷 확정용)
#     print("\n[Most frequent image paths]")
#     for p, c in sorted(img_paths.items(), key=lambda x: -x[1])[:10]:
#         print(f"  {c}x  {p}")

#     print("\n[Most frequent vec6 paths]")
#     for p, c in sorted(vec6_paths.items(), key=lambda x: -x[1])[:15]:
#         print(f"  {c}x  {p}")


# if __name__ == "__main__":
#     main()
import pickle, glob, numpy as np
p = sorted(glob.glob("/home/lmh/minimal_lerobot/examples/data/demo_*.pkl"))[0]
d = pickle.load(open(p,"rb"))
a0 = d["actions"][0]
v = np.array([a0[k] for k in ['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']], dtype=float)
print("first action vec:", v)
print("action min/max (first 200):")
A=[]
for i in range(min(200,len(d["actions"]))):
    ai=d["actions"][i]
    A.append([ai[k] for k in a0.keys()])
A=np.array(A,float)
print("min:", A.min(axis=0))
print("max:", A.max(axis=0))