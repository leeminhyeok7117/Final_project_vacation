import pickle
import numpy as np

PKL_PATH = "/home/lmh/minimal_lerobot/examples/data/demo_20260130_142104_10.pkl"

with open(PKL_PATH, "rb") as f:
    obj = pickle.load(f)

print("type(obj):", type(obj))

# dict면 key 보기
if isinstance(obj, dict):
    print("dict keys:", list(obj.keys())[:50])
    # 값 타입도 몇 개 찍기
    for k in list(obj.keys())[:10]:
        v = obj[k]
        print(f"  key={k!r} type={type(v)}")

# list/tuple면 첫 원소 타입
if isinstance(obj, (list, tuple)) and len(obj) > 0:
    print("len(obj):", len(obj))
    print("type(obj[0]):", type(obj[0]))
    if isinstance(obj[0], dict):
        print("obj[0] keys:", list(obj[0].keys())[:20])

# numpy 모양 찍기 시도
try:
    arr = np.asarray(obj)
    print("np.asarray(obj).shape:", getattr(arr, "shape", None), "dtype:", getattr(arr, "dtype", None))
except Exception as e:
    print("np.asarray(obj) failed:", e)

# callable이면 경고
if callable(obj):
    print("WARNING: obj is callable (function/method). You likely pickled a method instead of data.")