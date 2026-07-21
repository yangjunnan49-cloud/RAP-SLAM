from kiss_icp.config import KISSConfig
cfg = KISSConfig()
d = cfg.model_dump()
import json
print(json.dumps(d, indent=1, default=str))
