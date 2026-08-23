#!/bin/bash
# P4: Oxford Spires download helper (hf-mirror -> AWS CDN, ~3 MB/s, Range OK).
# Usage:
#   bash scripts/download_spires.sh gt               # all candidate gt-tum.txt
#   bash scripts/download_spires.sh clouds <seq>     # one sequence's lidar-clouds.zip + unzip
#   bash scripts/download_spires.sh clouds <seq> --keep-zip
# Verified 2026-08-23: huggingface.co blocked; hf-mirror.com API + resolve OK;
# LFS redirects to us.aws.cdn.hf.co (reachable, ~3 MB/s, supports Range).
set -u
MIRROR=https://hf-mirror.com
REPO=ori-drs/oxford_spires_dataset
ROOT=/mnt/e/githouse/Datasets/oxford_spires
# keble-college-01 and christ-church-01 have no gt-tum.txt (not core seqs).
SEQS="2024-03-12-keble-college-02 2024-03-12-keble-college-03 2024-03-12-keble-college-04 2024-03-12-keble-college-05 2024-03-13-observatory-quarter-01 2024-03-13-observatory-quarter-02 2024-03-14-blenheim-palace-01 2024-03-14-blenheim-palace-02 2024-03-14-blenheim-palace-05 2024-03-18-christ-church-02 2024-03-18-christ-church-03 2024-03-18-christ-church-05 2024-05-20-bodleian-library-02"

valid_gt() {  # first and last non-empty line must both have 8 TUM fields
  [ -s "$1" ] || return 1
  [ "$(head -1 "$1" | awk '{print NF}')" = "8" ] && \
  [ "$(tail -1 "$1" | awk '{print NF}')" = "8" ]
}

dl() {  # dl <url> <dst> <max-time> — 3 retries, returns 0 on HTTP 200
  local url=$1 dst=$2 mt=$3 code
  for att in 1 2 3; do
    code=$(curl -sL --max-time "$mt" -C - "$url" -o "$dst" -w '%{http_code}')
    [ "$code" = "200" ] && return 0
    sleep 5
  done
  echo "FAIL $dst (last http=$code)"
  return 1
}

case "${1:-}" in
  gt)
    mkdir -p "$ROOT/gt"
    for seq in $SEQS; do
      dst="$ROOT/gt/$seq.txt"
      if valid_gt "$dst"; then echo "SKIP $seq"; continue; fi
      if dl "$MIRROR/datasets/$REPO/resolve/main/sequences/$seq/processed/trajectory/gt-tum.txt" "$dst" 300 \
         && valid_gt "$dst"; then
        echo "OK   $seq $(stat -c %s "$dst") B"
      else
        echo "FAIL $seq (integrity)"; rm -f "$dst"
      fi
    done
    ;;
  clouds)
    seq="${2:?usage: clouds <seq> [--keep-zip]}"
    dst_dir="$ROOT/sequences/$seq/raw"
    mkdir -p "$dst_dir"
    zip="$dst_dir/lidar-clouds.zip"
    dl "$MIRROR/datasets/$REPO/resolve/main/sequences/$seq/raw/lidar-clouds.zip" "$zip" 14400 \
      || exit 1
    echo "unzipping $(stat -c %s "$zip") B ..."
    cd "$dst_dir" && unzip -q -o lidar-clouds.zip
    n=$(ls lidar-clouds/*.pcd 2>/dev/null | wc -l)
    echo "OK $seq: $n PCDs"
    if [ "${3:-}" != "--keep-zip" ]; then rm -f "$zip"; fi
    ;;
  *) echo "usage: $0 gt | clouds <seq> [--keep-zip]"; exit 1;;
esac
