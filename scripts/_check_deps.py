import importlib
for name in ["kiss_icp", "gtsam", "evo"]:
    try:
        m = importlib.import_module(name)
        print(f"{name} OK version={getattr(m, '__version__', 'n/a')}")
    except Exception as e:
        print(f"{name} FAILED: {e}")
try:
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    print("KissICP class import OK")
except Exception as e:
    print(f"KissICP class import FAILED: {e}")
