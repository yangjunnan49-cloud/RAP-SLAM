import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
cfg = KISSConfig()
cfg.data.deskew = False
o = KissICP(cfg)
attrs = [a for a in dir(o) if not a.startswith("__")]
print("attrs:", attrs)
rng = np.random.default_rng(0)
frame = rng.normal(size=(2000, 3)) * 10
ts = np.zeros(len(frame))
ret = o.register_frame(frame, ts)
print("register_frame returned:", type(ret))
for cand in ["poses", "get_poses", "pose", "last_pose"]:
    if hasattr(o, cand):
        v = getattr(o, cand)
        print(cand, "->", type(v), (v() if callable(v) else v))
