# Food_ID_matcher

[V-Nutri](https://github.com/K624-YCK/V-Nutri) 데이터셋의 음식 ID(`dish_id` 1~51)를 USDA [FNDDS](https://fdc.nal.usda.gov/) 음식 하나에 매칭하고, 그 음식의 영양소 65개(100g당)를 가져오는 도구입니다.

vision 모델이 사진을 V-Nutri 음식 ID로 분류하면, 이 저장소의 매핑 테이블로 해당 음식의 영양소를 조회할 수 있습니다.

```
vision 모델 → dish_id(예: 1) → results/vnutri_fndds_map.csv → FNDDS fdc_id → 영양소 65개 (100g당)
```

최종 매핑 테이블(`results/vnutri_fndds_map.csv`)은 저장소에 포함돼 있습니다. 그래서 **영양소 조회만 하려면 GPU가 필요 없습니다.** 1장의 데이터 준비만 하고 3장으로 넘어가면 됩니다. 매핑을 직접 다시 만들려면 2장의 절차를 따릅니다(GPU 필요, 약 1시간).

1. 설치와 데이터 준비
2. 매핑 만들기 (GPU, 선택)
3. 영양소 조회 (GPU 불필요)

---

## 1. 설치와 데이터 준비

### 요구 사항
| 항목 | 내용 |
|---|---|
| Python | 3.14에서 검증 |
| GPU | 매핑을 만들 때만 필요. **24GB 이상 1장** (RTX A5000에서 검증). 모델을 GPU 여러 장에 나눠 올리지 않습니다 (7장 참고) |
| 디스크 | 모델 약 180GB (처음 실행할 때 Hugging Face에서 자동으로 받음) |
| 기타 | `curl`, `unzip`, `sha256sum` (데이터 다운로드용) |

```bash
git clone https://github.com/YoungiJang/Food_ID_matcher.git && cd Food_ID_matcher

bash scripts/download_data.sh                 # 원본: FNDDS 약 25MB, V-Nutri CSV 2개 → data/raw/
python food_matcher.py prepare                # 가공 데이터 → data/ (표준 라이브러리만 사용, 1분 이내)

# 매핑을 다시 만들 때만 (2장)
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 사용하는 모델 (모두 로컬 실행, 별도 승인 불필요, API 비용 없음)
| 역할 | 모델 | 라이선스 | 크기 | GPU 메모리 |
|---|---|---|---|---|
| 검색 임베딩 | `BAAI/bge-m3` | MIT | 4.6GB | 약 3GB |
| 재순위화 | `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 2.3GB | 약 2GB |
| 쿼리 재작성 | `Qwen/Qwen2.5-14B-Instruct` (8-bit) | Apache-2.0 | 29.6GB | 약 16GB |
| 선택 | `Qwen/Qwen2.5-32B-Instruct` (4-bit) | Apache-2.0 | 65.5GB | 약 20GB |
| 검토 | `microsoft/phi-4` (8-bit) | MIT | 29.3GB | 약 16GB |
| 평가 전용 심사 | `mistralai/Mistral-Small-24B-Instruct-2501` (4-bit) | Apache-2.0 | 47.2GB | 약 14GB |

선택, 검토, 심사 모델은 일부러 **서로 다른 계열**로 골랐습니다. 같은 계열 모델끼리는 서로의 답을 후하게 평가하는 경향이 있기 때문입니다.

---

## 2. 매핑 만들기 (GPU 필요, 선택)

저장소에 포함된 매핑을 다시 만드는 절차이고, 현재 최고 설정입니다. 각 단계는 앞 단계의 결과 파일을 읽습니다. 실행하면 `results/vnutri_fndds_map.csv`를 덮어씁니다.
```bash
PY=.venv/bin/python
$PY food_matcher.py rewrite                       # 1. 레시피마다 FNDDS 식 영문 설명 생성 → data/query_rewrites.csv
$PY food_matcher.py retrieve -k 40 --rerank       # 2. 후보 40개 검색        → data/retrieval_candidates.csv
$PY food_matcher.py llm-match --top 40 \
    --model Qwen/Qwen2.5-32B-Instruct --quantize 4bit   # 3. 선택 (5회 다수결)  → results/vnutri_fndds_map_select.csv
$PY food_matcher.py llm-review --top 40           # 4. 검토, 최종 매핑 확정  → results/vnutri_fndds_map.csv
$PY food_matcher.py evaluate                      # 5. 지표 계산 (선택)      → results/eval_report.json
$PY food_matcher.py judge results/vnutri_fndds_map.csv   # 6. 심사 점수 (선택) → results/judge/
```
- 1단계 결과(`data/query_rewrites.csv`)는 저장소에 포함돼 있어서, 1단계는 건너뛰어도 됩니다.
- 모든 LLM은 greedy 디코딩이라, 같은 모델과 같은 입력이면 같은 결과가 나옵니다.
- 각 명령의 옵션은 `python food_matcher.py <명령> --help`로 볼 수 있습니다.

**Slurm 클러스터에서 실행할 때**는 위 명령을 GPU 1장짜리 작업으로 감싸면 됩니다.
```bash
sbatch --gres=gpu:1 --mem=64G --time=12:00:00 -o run-%j.out --wrap "cd $PWD && .venv/bin/python food_matcher.py llm-match --top 40 --model Qwen/Qwen2.5-32B-Instruct --quantize 4bit"
```
노드마다 홈 디렉터리가 따로라면, 저장소와 모델 캐시(`~/.cache/huggingface`)가 있는 노드를 `-w`로 고정하세요.

### 자동 실험
`scripts/auto_experiment.py`는 지금까지 심사 점수가 가장 높은 실행의 설정에 변경 하나를 적용해 실행하고, 모든 실행을 다시 채점합니다. 기준이 될 실행이 필요하므로, 위 절차로 만든 결과를 먼저 `results/<이름>/`에 `config.json`과 함께 두고 한 번 채점해야 합니다.
```bash
.venv/bin/python scripts/auto_experiment.py v8 rerank=false      # 결과: results/v8/, 순위표: results/judge/leaderboard.md
```

---

## 3. 영양소 조회 (GPU 불필요)

```bash
python food_matcher.py lookup 1 8 --nutrients "Energy (KCAL)" "Protein (G)" "Total lipid (fat) (G)"
```

출력 예:
```
[1] Cacio e Pepe → Pasta, cooked (fdc_id 2708357, approx/low, per 100g)
    note: The candidate is basic cooked pasta without additional sauces or ingredients.
    Energy (KCAL): 157.0
    Protein (G): 5.76
    Total lipid (fat) (G): 0.92
[8] Pizza Dough → 매칭 없음 (none)
    ...
```

Python 코드에서 쓰려면:
```python
from food_matcher import NutrientLookup

lk = NutrientLookup()                    # 기본값: results/vnutri_fndds_map.csv
r = lk.lookup(1)                         # V-Nutri dish_id
r["fndds_description"], r["match_type"], r["confidence"], r["nutrients"]["Energy (KCAL)"]
```

- 영양소 열 이름 전체는 `data/fndds_nutrients.csv`의 첫 줄에 있습니다(65개).
- **모든 값은 100g당입니다.** 실제 섭취량을 구하려면 음식의 무게를 따로 곱해야 합니다.
- `match_type`이 `none`이면 매칭된 항목이 없어서 영양소가 `None`으로 나옵니다.
- `confidence`가 `low`인 항목은 근사 매칭이라 주의해서 써야 합니다.
- 다른 매핑 파일을 쓰려면 `--mapping <파일>` 또는 `NutrientLookup(mapping=...)`를 지정합니다.

---

## 4. 동작 방식

```
[오프라인, 처음 한 번]
V-Nutri 레시피 48종 (dish_id 51개)
 ① rewrite     LLM이 "Lentil curry with rice" 같은 FNDDS 식 일반 설명 생성
 ② retrieve    FNDDS 5,431개 중 후보 top-k
               - 임베딩(bge-m3): 쿼리(이름, 이름+식재료, 재작성 설명) × FNDDS 이름
               - 임베딩: 이름+식재료 × FNDDS 이름+구성 재료(input_food.csv)
               - 키워드(IDF) 순위
               → RRF로 합산, 같은 계열 변형은 최대 6개로 제한, 크로스 인코더로 재정렬
 ③ llm-match   선택 모델이 후보 순서를 섞어 5번 고르고 다수결. 득표율이 확신도
               (요리 종류 → 주재료 → 기름 사용 → 후보 압축 → 선택 순으로 생각하게 함)
 ④ llm-review  다른 계열 모델이 검토. 거부하면 정해진 규칙에 따라 교체하거나 확신도를 low로 낮춤
 → results/vnutri_fndds_map.csv

[온라인, 매번]
 lookup        dish_id → 매핑 테이블 → FNDDS 영양소
```

판단 규칙은 [docs/matching_rubric.md](docs/matching_rubric.md)에 있습니다(요리 종류 일치 > 주재료 > 조리 방식 > 일반 항목 우선 등). **매칭 과정에서는 V-Nutri 실측 영양값을 쓰지 않습니다.** 실측값은 평가에만 씁니다.

### 평가 (정답 매핑 없이)
- **심사 점수 (주 지표):** 평가 전용 모델(Mistral-Small-24B)이 요리 이름, 식재료, FNDDS 이름, 구성 재료만 보고 1~5점을 매깁니다. 매칭 방법이나 확신도는 보여 주지 않습니다. 매칭 없음은 1점으로 셉니다.
  - 5점: 같은 요리, 같은 주재료와 조리 방식
  - 4점: 같은 요리, 사소한 차이
  - 3점: 비슷한 요리이거나 주재료만 맞음
  - 2점: 관련은 있지만 다른 요리
  - 1점: 틀린 음식
- **실측 일관성 (보조 지표):** V-Nutri 실측값과 FNDDS의 단백질, 지방, 탄수화물 열량 비율 차이(L1, 낮을수록 좋음)입니다. V-Nutri에 무게가 없어서 비율만 비교합니다.
- `evaluate`는 확신도별 실측 일관성도 계산합니다. 확신도가 높을수록 거리가 작으면, 확신도를 믿을 수 있다는 뜻입니다.

---

## 5. 결과

개발할 때 실행한 결과입니다(48개 레시피, 심사 모델 Mistral-Small-24B). 결과 파일은 저장소에 포함하지 않습니다.

| 실행 | 설정 | 심사 평균 | 4점 이상 | 매칭 없음 |
|---|---|---|---|---|
| **v5 (2장의 설정)** | Qwen-32B 선택 + phi-4 검토, 재순위화, 후보 40개 | **3.31** | 48% | 7 |
| v4 | 재순위화 없음 | 3.10 | 42% | 11 |
| baseline | Qwen-14B, 손으로 쓴 쿼리, 후보 30개 | 3.17 | 46% | 11 |
| reference/b_claude_code | (참고) 같은 후보에서 Claude Code가 판단 | 3.52 | 46% | 2 |
| reference/v0_manual | (참고) 수작업. 매칭할 때 실측 비율을 참고함 | 3.58 | 48% | 0 |

`reference`는 비교용입니다. 같은 검색 후보에서 Claude Code가 판단하거나 사람이 직접 고른 결과라서, 이 저장소의 코드로는 다시 만들 수 없습니다.

---

## 6. 폴더 구조

```
food_matcher.py          모든 명령 (prepare, rewrite, retrieve, llm-match, llm-review, evaluate, judge, compare, lookup)
scripts/download_data.sh 원본 다운로드 (FNDDS는 SHA256 검증, V-Nutri는 커밋 고정)
scripts/auto_experiment.py  자동 실험 한 단계
docs/matching_rubric.md  매칭 판단 규칙
data/raw/                원본 (git에 올리지 않음)
data/                    가공 데이터 (query_rewrites.csv만 git에 포함)
results/                 실행 결과 (git에는 vnutri_fndds_map.csv만 포함)
  vnutri_fndds_map.csv   최종 매핑 (lookup 기본값)
  <run>/                 자동 실험의 실행별 결과: config.json, 후보, 선택, 최종 매핑, 평가
  judge/                 심사 점수 (실행별 csv, summary.json, leaderboard.md)
```

---

## 7. 한계와 주의 사항
- **값은 100g당입니다.** V-Nutri에는 무게 정보가 없어서, 접시 하나의 총 영양소는 무게를 따로 추정해야 합니다.
- **근사 매칭이 많습니다.** FNDDS는 미국 식단 기준이라, 인도·영국 가정식(Uttapam, Mangsho Bhuna 등)이나 복합 요리(French Fries and Beef Skewers)는 비슷한 항목으로 대체됩니다. `match_type`, `confidence`, `reason` 열을 함께 확인하세요.
- **평가 데이터가 48개뿐입니다.** 레시피 하나의 점수가 1점 바뀌면 평균이 약 0.02 바뀝니다. 심사 점수는 LLM의 판단이므로, 실행끼리 비교하는 상대 지표로 보는 게 맞습니다.
- **GPU 한 장만 씁니다.** 개발 서버에서는 GPU 간 직접 복사(P2P) 결과가 깨져서, `device_map="auto"`로 모델을 여러 장에 나누면 출력이 망가졌습니다. 48GB GPU라면 `--quantize none`으로 bf16을 쓸 수 있습니다.

---

## 8. 데이터 출처와 라이선스
- **USDA FoodData Central, FNDDS (Survey Foods, 2024-10-31):** 미국 정부 저작물(public domain). `scripts/download_data.sh`로 받습니다.
- **V-Nutri (HD-EPIC for V-Nutri):** V-Nutri Annotation Data License. 비상업적 이용만 허용하고, 결과를 공개할 때 논문 인용이 필수입니다. 원본은 이 저장소에 포함하지 않고 다운로드 스크립트로 받습니다. 저장소의 `data/query_rewrites.csv`에는 레시피 키가 들어 있습니다.
  ```bibtex
  @InProceedings{Yue_2026_CVPR,
    author    = {Yue, Chengkun and Xu, Chuanzhi and He, Jiangpeng},
    title     = {V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year      = {2026}
  }
  ```
  V-Nutri의 바탕이 된 HD-EPIC(Perrett et al., CVPR 2025)도 함께 인용하세요.
- **모델:** 위 표의 각 라이선스를 따릅니다.
