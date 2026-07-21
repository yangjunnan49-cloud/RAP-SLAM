#!/usr/bin/env python3
"""Download a frame range of the POSS odometry velodyne zip via HTTP range requests.

Resumable: skips files that already exist with the correct uncompressed size.
Runs on the Windows host (managed Python) — PKU server is fast from here.

Usage:
  python scripts/download_poss_segment.py --start 3850 --end 5268 --threads 4
"""
import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor"))
from remotezip import RemoteZip  # noqa: E402

URL = "http://www.poss.pku.edu.cn/OpenDataResource/poss_odomotry/data_odometry_velodyne.zip"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "poss", "dataset", "sequences", "00", "velodyne")

_tls = threading.local()


def get_rz():
    if not hasattr(_tls, "rz"):
        _tls.rz = RemoteZip(URL)
    return _tls.rz


def fetch(member):
    rz = get_rz()
    info = rz.getinfo(member)
    dest = os.path.join(OUT, os.path.basename(member))
    if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
        return os.path.basename(member), 0
    data = rz.read(member)
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return os.path.basename(member), len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=3850)
    ap.add_argument("--end", type=int, default=5268)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rz = RemoteZip(URL)
    members = [f"dataset/sequences/00/velodyne/{i:06d}.bin" for i in range(args.start, args.end + 1)]
    members = [m for m in members if m in set(rz.namelist())]
    print(f"{len(members)} files to fetch -> {OUT}")

    done = got = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(fetch, m): m for m in members}
        for fut in as_completed(futs):
            name, n = fut.result()
            done += 1
            got += n
            if done % 50 == 0 or done == len(members):
                print(f"[{done}/{len(members)}] downloaded {got/1e6:.0f} MB this session", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
