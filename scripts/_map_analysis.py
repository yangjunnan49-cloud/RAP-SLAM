import numpy as np, open3d as o3d, time
t0=time.time()
pcd = o3d.io.read_point_cloud("/mnt/e/githouse/RAP-main/demo_example_data/map_kitti/kitti00.ply")
pts = np.asarray(pcd.points)
print("load", time.time()-t0, "N=", len(pts))
print("bbox min", pts.min(0), "max", pts.max(0))
# 2D histogram of XY density, 10m cells
xmin,ymin = pts[:,0].min(), pts[:,1].min()
xmax,ymax = pts[:,0].max(), pts[:,1].max()
H, xe, ye = np.histogram2d(pts[:,0], pts[:,1], bins=[np.arange(xmin,xmax+10,10), np.arange(ymin,ymax+10,10)])
print("cells>2000 pts:", (H>2000).sum(), "of", H.size)
# print top-density cells
idx = np.dstack(np.unravel_index(np.argsort(H.ravel())[::-1][:30], H.shape))[0]
print("top cells (x0,y0,count):")
for i,j in idx[:20]:
    print(f"  x[{xe[i]:.0f},{xe[i+1]:.0f}] y[{ye[j]:.0f},{ye[j+1]:.0f}] -> {int(H[i,j])}")
# z stats
print("z percentiles 1/5/50/95/99:", np.percentile(pts[:,2],[1,5,50,95,99]))
