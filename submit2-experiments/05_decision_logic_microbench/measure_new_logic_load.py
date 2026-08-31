import pickle
import time
import os

t0 = time.perf_counter()
with open("/app/new_logic_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)
t1 = time.perf_counter()

size_bytes = os.path.getsize("/app/new_logic_pipeline.pkl")
print(f"file_size_bytes={size_bytes}")
print(f"load_time_ms={(t1 - t0) * 1000:.4f}")
