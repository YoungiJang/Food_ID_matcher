#!/usr/bin/env bash
# 원본 데이터를 data/raw/ 에 내려받는다. 저장소에는 원본을 포함하지 않는다.
#   - USDA FNDDS (FoodData Central, Survey Foods, 2024-10-31): 미국 정부 저작물(public domain)
#   - V-Nutri (HD-EPIC for V-Nutri): V-Nutri Annotation Data License (비상업적 이용, 논문 인용 필수)
set -euo pipefail
cd "$(dirname "$0")/.."

FNDDS_URL="https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip"
FNDDS_SHA256="5ccc25ec2777a8982fbb61378a42f415316173eb11e48c9a8ba4cb19f5a4f29c"
VNUTRI_COMMIT="a822d677f5f2157772dd3c60548c54e7ea2cde0a"
VNUTRI_URL="https://raw.githubusercontent.com/K624-YCK/V-Nutri/${VNUTRI_COMMIT}/HD-EPIC_for_V-Nutri"

mkdir -p data/raw/fndds data/raw/vnutri
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "FNDDS ..."
curl -fsSL -o "$tmp/fndds.zip" "$FNDDS_URL"
echo "$FNDDS_SHA256  $tmp/fndds.zip" | sha256sum -c --quiet
unzip -q -j -o "$tmp/fndds.zip" -d data/raw/fndds

echo "V-Nutri ..."
for f in dish_index.csv dish_ingredient_nutrition_map.csv; do
    curl -fsSL -o "data/raw/vnutri/$f" "$VNUTRI_URL/$f"
done

echo "done: data/raw/"
