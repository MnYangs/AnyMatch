#!/usr/bin/env bash

FILE_ID="1JCz52pM7On_YWCjBWr_VuTiXt1uOXPD_"

if ! command -v gdown &>/dev/null; then
  echo "错误: 未找到 gdown。请先安装: pip install gdown" >&2
  exit 1
fi

gdown --fuzzy "https://drive.google.com/uc?id=$FILE_ID" -c