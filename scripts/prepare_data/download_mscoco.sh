#!/usr/bin/env bash
# Download and extract the official MSCOCO 2014 images and annotations.
#
# Set DATA_ROOT to control where the data is written (defaults to
#   <repo>/data). The resulting layout is:
#
#   $DATA_ROOT/mscoco/train2014/          training images
#   $DATA_ROOT/mscoco/val2014/            validation images
#   $DATA_ROOT/mscoco/annotations/        captions / instances annotations
set -euo pipefail

ROOT_DIR="${BASH_SOURCE[0]%/*}/../.."
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
COCO_ROOT="$DATA_ROOT/mscoco"

IMG_BASE="http://images.cocodataset.org/zips"
ANN_BASE="http://images.cocodataset.org/annotations"

fetch_and_extract() {
  local url="$1" target_dir="$2"
  if [[ -d "$target_dir" ]]; then
    echo "Already present: $target_dir"
    return
  fi
  local zip_path
  zip_path="$COCO_ROOT/$(basename "$url")"
  echo "Downloading $url"
  curl -L --retry 3 --output "$zip_path" "$url"
  echo "Extracting $(basename "$url")"
  unzip -q "$zip_path" -d "$COCO_ROOT"
  rm -f "$zip_path"
}

mkdir -p "$COCO_ROOT"

fetch_and_extract "$IMG_BASE/train2014.zip" "$COCO_ROOT/train2014"
fetch_and_extract "$IMG_BASE/val2014.zip" "$COCO_ROOT/val2014"
fetch_and_extract "$ANN_BASE/annotations_trainval2014.zip" "$COCO_ROOT/annotations"

echo "MSCOCO data ready under $COCO_ROOT"
