import numpy as np, open3d as o3d
pcd = o3d.io.read_point_cloud("/mnt/e/githouse/RAP-main/demo_example_data/map_kitti/kitti00.ply")
pts = np.asarray(pcd.points)
tree = o3d.geometry.KDTreeFlann(pcd)

def counts_for_circle(cx, cy, r, n=100, rng=50.0):
    cnts = []
    for k in range(n):
        a = 2*np.pi*k/n
        x, y = cx + r*np.cos(a), cy + r*np.sin(a)
        # query ball around (x, y, z_local) using xy radius: approximate with full 3D after z estimate
        idx = tree.search_radius_vector_3d(np.array([x, y, 5.0]), rng)[1]
        cnts.append(len(idx))
    return np.array(cnts)

for (cx, cy) in [(250, -30), (260, -20), (240, -20), (255, -10)]:
    for r in [70, 80, 90]:
        c = counts_for_circle(cx, cy, r)
        print(f"center=({cx},{cy}) r={r}: min={c.min()} p5={int(np.percentile(c,5))} med={int(np.median(c))} max={c.max()}")
