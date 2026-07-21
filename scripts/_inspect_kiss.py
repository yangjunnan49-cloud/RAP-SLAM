import inspect
from kiss_icp.kiss_icp import KissICP
from kiss_icp.config import KISSConfig
print("=== KissICP.register_frame signature ===")
print(inspect.signature(KissICP.register_frame))
print("=== KissICP __init__ ===")
print(inspect.signature(KissICP.__init__))
print("=== KISSConfig fields ===")
import dataclasses
try:
    for f in dataclasses.fields(KISSConfig):
        print(f.name, "=", f.default)
except Exception as e:
    print("not dataclass:", e)
    print([a for a in dir(KISSConfig) if not a.startswith("_")])
