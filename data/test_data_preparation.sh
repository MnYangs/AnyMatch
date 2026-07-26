#!/usr/bin/env bash

FILE_IDS=(
  "1M5bKE56N--SoA554xEpb4tBzvZjZMVDi"
  "12msTsEm-iRE9_I6GWMaU7lT04baFxFxf"
  "1fKJSnYv-WiX4GeTeFbgRfsQUQ5O_uoUs"
  "1KRV29uyuNfOGhSf9E-19yHbNpJbfMIV-"
  "1kxI-SI5Wpl37YaYbKXpMnfilLYQbJYVt"
)

for id in "${FILE_IDS[@]}"; do
  gdown --fuzzy "https://drive.google.com/uc?id=$id" -c
done