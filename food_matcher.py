"""V-Nutri 음식 ID → USDA FNDDS 음식 매칭 및 영양소 조회.

파이프라인 (오프라인, 처음 한 번. 위에서부터 순서대로):
    prepare      원본 → data/ 가공 데이터 (FNDDS 영양소 표, FNDDS 재료 문서, V-Nutri ID 목록)
    rewrite      LLM 이 레시피마다 FNDDS 식 일반 영문 설명을 생성 (검색 쿼리 재작성)
    retrieve     레시피마다 FNDDS 후보 top-k 검색 (로컬 임베딩 bge-m3 + 키워드, RRF)
    llm-match    선택 모델(Qwen2.5-14B)이 후보 순서를 섞어 5번 고르고 다수결
    llm-review   다른 계열 검토 모델(phi-4)이 선택 결과를 검토해 최종 매핑 확정
    evaluate     정답 매핑 없이 평가 (득표율, 검토 통과율, V-Nutri 실측 영양값과의 일관성)
    judge        평가 전용 심사 모델(Mistral-Small-24B)이 요리와 매칭 항목의 유사도를 1~5점으로 채점

실행 시 (온라인, 매번):
    lookup       V-Nutri dish_id → 매핑 테이블 → FNDDS 영양소 65개 (100g 당)

기타:
    compare      두 매핑 테이블의 차이 목록 생성

사용 예 (자세한 내용은 README.md):
    bash scripts/download_data.sh
    python food_matcher.py prepare
    python food_matcher.py lookup 1 8 --nutrients "Energy (KCAL)" "Protein (G)"

폴더:
    data/raw/   내려받은 원본 (scripts/download_data.sh). 저장소에 포함하지 않는다.
    data/       prepare 와 파이프라인이 만드는 중간 데이터
    results/    매핑 테이블, 평가 결과

prepare / evaluate / compare / lookup 은 표준 라이브러리만 쓴다.
rewrite / retrieve / llm-match / llm-review 는 requirements.txt 의 패키지와 GPU 가 필요하다.
"""
import argparse
import csv
import json
import math
import re
from collections import Counter, OrderedDict, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
DATA = HERE / "data"
RESULTS = HERE / "results"

FNDDS_SRC = RAW / "fndds"
VNUTRI_DISH_INDEX = RAW / "vnutri" / "dish_index.csv"
VNUTRI_INGREDIENTS = RAW / "vnutri" / "dish_ingredient_nutrition_map.csv"
FNDDS_TABLE = DATA / "fndds_nutrients.csv"       # FNDDS 음식 1행 × 영양소 65열 (100g 당)
FNDDS_RECIPES = DATA / "fndds_recipes.csv"       # FNDDS 음식별 주요 구성 재료 (검색용 문서)
VNUTRI_IDS = DATA / "vnutri_ids.csv"             # V-Nutri dish_id → recipe_key, recipe_name
QUERY_REWRITES = DATA / "query_rewrites.csv"     # rewrite 결과
CANDIDATES = DATA / "retrieval_candidates.csv"   # retrieve 결과
MAPPING_SELECT = RESULTS / "vnutri_fndds_map_select.csv"  # llm-match 결과 (선택 모델 다수결)
MAPPING = RESULTS / "vnutri_fndds_map.csv"                # llm-review 결과 (최종 매핑)

SELECT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
REVIEW_MODEL = "microsoft/phi-4"
JUDGE_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"  # 평가 전용. 선택·검토 모델과 다른 계열
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

META_COLUMNS = ("fdc_id", "description", "wweia_category", "basis")
MAPPING_COLUMNS = ["recipe_key", "recipe_name", "fdc_id", "fndds_description", "match_type", "confidence", "reason"]
MACROS = ("Protein (G)", "Total lipid (fat) (G)", "Carbohydrate, by difference (G)")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in records)


def load_recipes(rewrites=QUERY_REWRITES):
    """recipe_key → {name, ingredients[], rewrites[]} (vnutri_ids.csv 순서 유지)."""
    recipes = OrderedDict((r["recipe_key"], {"name": r["recipe_name"], "ingredients": [], "rewrites": []})
                          for r in read_csv(VNUTRI_IDS))
    for r in read_csv(VNUTRI_INGREDIENTS):
        items = recipes[r["recipe_key"]]["ingredients"]
        if r["ingredient_name"] not in items:
            items.append(r["ingredient_name"])
    if rewrites and Path(rewrites).exists():
        for r in read_csv(rewrites):
            recipes[r["recipe_key"]]["rewrites"].append(r["rewrite"])
    return recipes


# -------------------------------------------------------------------- prepare

def prepare(top_ingredients=5):
    """원본 → FNDDS 영양소 표, FNDDS 재료 문서, V-Nutri ID 목록.

    주의: FNDDS food_nutrient.nutrient_id 는 nutrient.id 가 아니라 nutrient.nutrient_nbr 과 매칭된다.
    """
    foods = read_csv(FNDDS_SRC / "food.csv")
    nutrients = {r["nutrient_nbr"]: r for r in read_csv(FNDDS_SRC / "nutrient.csv") if r["nutrient_nbr"]}
    category = {r["wweia_food_category"]: r["wweia_food_category_description"]
                for r in read_csv(FNDDS_SRC / "wweia_food_category.csv")}

    amounts = defaultdict(dict)
    for r in read_csv(FNDDS_SRC / "food_nutrient.csv"):
        amounts[r["fdc_id"]][r["nutrient_id"]] = r["amount"]
    used = {nbr for per_food in amounts.values() for nbr in per_food}
    nbrs = sorted(used, key=lambda n: float(nutrients[n]["rank"] or 1e9))
    columns = [f'{nutrients[n]["name"]} ({nutrients[n]["unit_name"]})' for n in nbrs]

    rows = []
    for food in foods:
        per_food = amounts.get(food["fdc_id"])
        if not per_food:  # 'Milk, human' 은 영양소 데이터가 없음
            continue
        rows.append([food["fdc_id"], food["description"], category.get(food["food_category_id"], ""), "100g"]
                    + [per_food.get(n, "") for n in nbrs])
    write_csv(FNDDS_TABLE, list(META_COLUMNS) + columns, rows)
    print(f"wrote {FNDDS_TABLE} ({len(rows)} foods x {len(columns)} nutrients)")

    # FNDDS 음식마다 무게가 큰 구성 재료 top_ingredients 개 (재료 이름은 괄호를 뺀 첫 쉼표 앞 부분만)
    recipe = defaultdict(list)
    for r in read_csv(FNDDS_SRC / "input_food.csv"):
        name = re.sub(r"\s*\([^)]*\)", "", r["sr_description"]).split(",")[0].strip()
        recipe[r["fdc_id"]].append((float(r["gram_weight"] or 0), name))
    doc_rows = []
    for fdc_id, desc, *_ in rows:
        names = []
        for _, name in sorted(recipe.get(fdc_id, []), reverse=True):
            if name and name.lower() not in (n.lower() for n in names):
                names.append(name)
        doc_rows.append([fdc_id, desc, "; ".join(names[:top_ingredients])])
    write_csv(FNDDS_RECIPES, ["fdc_id", "description", "ingredients"], doc_rows)
    print(f"wrote {FNDDS_RECIPES}")

    seen, ids = set(), []
    for r in read_csv(VNUTRI_DISH_INDEX):
        ids.append([r["dish_id"], r["recipe_key"], r["recipe_name"]])
        seen.add(r["recipe_key"])
    write_csv(VNUTRI_IDS, ["dish_id", "recipe_key", "recipe_name"], ids)
    print(f"wrote {VNUTRI_IDS} ({len(ids)} dishes, {len(seen)} recipes)")


# --------------------------------------------------------------- llm (공통)

def load_llm(model_name, quantize="8bit", gpu=0):
    """모델을 GPU 한 장에 올리고 generate(messages) -> str 함수를 돌려준다 (greedy 디코딩).

    GPU 한 장만 쓰는 이유: 개발 서버에서 GPU 간 직접 복사(P2P) 시 값이 깨져, device_map="auto" 로
    여러 장에 나누면 출력이 망가졌다. 14B 급은 bf16(약 29GB)이 24GB 에 안 들어가서 기본값을 8bit(약 16GB)로
    둔다. 48GB GPU 라면 --quantize none 으로 bf16 을 쓸 수 있다.
    """
    import logging

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)  # 8bit 연산마다 반복되는 dtype 변환 경고
    quant = {"none": None,
             "8bit": BitsAndBytesConfig(load_in_8bit=True),
             "4bit": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)}[quantize]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, quantization_config=quant,
                                                 device_map={"": gpu})

    def generate(messages, max_new_tokens=512):
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def ask_json(generate, messages, validate, max_retries=1):
    """generate 출력에서 JSON 을 꺼내 validate(ans) 로 검사한다. validate 는 문제가 있으면 오류 문자열을 돌려준다.

    실패하면 오류를 알려 주고 max_retries 번 다시 묻는다. (ans 또는 None, 시도 기록 list)
    """
    log = []
    for attempt in range(max_retries + 1):
        text = generate(messages)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            ans = json.loads(m.group(0)) if m else None
            error = validate(ans) if isinstance(ans, dict) else "no JSON object found"
        except json.JSONDecodeError as e:
            ans, error = None, f"invalid JSON: {e}"
        log.append({"attempt": attempt, "output": text, "error": error})
        if not error:
            return ans, log
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": f"Your answer was invalid ({error}). "
                                                           "Answer again with the JSON object only."}]
    return None, log


def dish_lines(recipe):
    return [f"Dish: {recipe['name']}", f"Ingredients: {', '.join(recipe['ingredients'])}"]


# -------------------------------------------------------------------- rewrite

REWRITE_PROMPT = """You describe a home-cooked dish the way the USDA FNDDS food database names its foods.
FNDDS names are generic and describe what the food is made of, for example:
  "Pasta with tomato-based sauce and meat, home recipe", "Egg omelet or scrambled egg, with vegetables, fat added",
  "Lentil curry with rice", "Cucumber salad, made with Italian dressing", "Fruit smoothie, with whole fruit and dairy".

Given the dish name and its ingredients, write 2 such descriptions:
1. the closest generic dish name (translate non-English or regional dish names into what the dish is),
2. a description of the main components and preparation.
Do not use brand names. Use the ingredient list when the dish name is misleading.

Answer with one JSON object only: {"descriptions": ["...", "..."]}"""


def validate_rewrite(ans):
    d = ans.get("descriptions")
    if not isinstance(d, list) or not 1 <= len(d) <= 3 or not all(isinstance(x, str) and x.strip() for x in d):
        return "descriptions must be a list of 1-3 non-empty strings"
    return None


def rewrite(out=QUERY_REWRITES, model_name=SELECT_MODEL, quantize="8bit", gpu=0):
    """LLM 이 레시피마다 FNDDS 식 일반 영문 설명을 만든다. retrieve 의 검색 쿼리로 쓴다."""
    recipes = load_recipes(rewrites=None)
    generate = load_llm(model_name, quantize, gpu)
    rows, log = [], []
    for key, r in recipes.items():
        ans, attempts = ask_json(generate, [{"role": "system", "content": REWRITE_PROMPT},
                                            {"role": "user", "content": "\n".join(dish_lines(r))}],
                                 validate_rewrite)
        log.append({"recipe_key": key, "attempts": attempts})
        for text in (ans or {}).get("descriptions", []):
            rows.append([key, text.strip()])
        print(f'{key} {r["name"]} -> {(ans or {}).get("descriptions")}', flush=True)
    write_csv(out, ["recipe_key", "rewrite"], rows)
    write_jsonl(Path(out).with_suffix(".jsonl"), log)
    print(f"wrote {out}")


# ------------------------------------------------------------------- retrieve

STOP = {"and", "with", "or", "of", "the", "a", "recipe", "nfs", "ns", "to", "as", "from", "type", "made"}
RRF_K = 60
DEPTH = 200  # 쿼리별로 RRF 에 넣을 순위 깊이


def tokens(text):
    out = []
    for t in re.findall(r"[a-z]+", text.lower()):
        if len(t) > 3 and t.endswith("es") and not t.endswith("ses"):
            t = t[:-2]
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        if t not in STOP:
            out.append(t)
    return out


def keyword_rank(query, fndds, idf, k):
    """IDF 가중 토큰 겹침(쿼리 기준 recall) + 문자열 유사도 보정으로 FNDDS 순위를 매긴다."""
    q = set(tokens(query))
    if not q:
        return []
    unseen = math.log(len(fndds))  # FNDDS 에 없는 토큰도 분모에는 포함
    q_weight = sum(idf.get(t, unseen) for t in q)
    scored = []
    for fdc_id, desc in fndds:
        overlap = sum(idf.get(t, 0) for t in q & set(tokens(desc))) / q_weight
        if overlap == 0:
            continue
        sim = SequenceMatcher(None, query.lower(), desc.lower()).ratio()
        scored.append((round(0.8 * overlap + 0.2 * sim, 4), fdc_id, desc))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


def diversify(order, descriptions, k, per_head):
    """설명의 첫 쉼표 앞 부분(예: "Pizza")이 같은 후보는 최대 per_head 개만 남긴다."""
    out, seen = [], Counter()
    for i in order:
        head = descriptions[i].split(",")[0].lower()
        if seen[head] < per_head:
            out.append(i)
            seen[head] += 1
            if len(out) == k:
                break
    return out


def retrieve(out=CANDIDATES, model_name=EMBED_MODEL, k=30, per_head=6, rewrites=QUERY_REWRITES,
             rerank=False, rerank_pool=80):
    """레시피마다 여러 순위표를 RRF(각 순위표에서 1/(60+순위) 합산)로 합쳐 후보 top-k 를 뽑는다.

    순위표:
    - 임베딩: 쿼리(이름, 이름+식재료, 재작성 설명들) × FNDDS 이름
    - 임베딩: 이름+식재료 × FNDDS 이름+구성 재료 (재료끼리 비교)
    - 키워드: 쿼리(이름, 재작성 설명들) × FNDDS 이름
    rerank 이면 RRF 상위 rerank_pool 개를 크로스 인코더(bge-reranker-v2-m3)로 다시 정렬한 뒤 top-k 를 고른다.
    크로스 인코더는 (레시피 이름+식재료, FNDDS 이름+구성 재료) 쌍을 함께 읽어 관련도를 매긴다.
    정답 영양값은 쓰지 않는다.
    """
    from sentence_transformers import SentenceTransformer

    docs = read_csv(FNDDS_RECIPES)
    fndds = [(r["fdc_id"], r["description"]) for r in docs]
    descriptions = [d for _, d in fndds]
    position = {fdc_id: i for i, (fdc_id, _) in enumerate(fndds)}
    recipes = load_recipes(rewrites)

    df = Counter(t for d in descriptions for t in set(tokens(d)))
    idf = {t: math.log(len(fndds) / c) for t, c in df.items()}

    model = SentenceTransformer(model_name)
    encode = lambda texts: model.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_tensor=True)
    name_emb = encode(descriptions)
    doc_texts = [f'{r["description"]}. Ingredients: {r["ingredients"]}' for r in docs]
    recipe_emb = encode(doc_texts)
    if rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANK_MODEL, max_length=512)

    rows = []
    for key, r in recipes.items():
        with_ingredients = f'{r["name"]}. Ingredients: {", ".join(r["ingredients"][:6])}'
        queries = [r["name"], with_ingredients] + r["rewrites"]
        rankings = (encode(queries) @ name_emb.T).topk(DEPTH, dim=1).indices.tolist()
        rankings += (encode([with_ingredients]) @ recipe_emb.T).topk(DEPTH, dim=1).indices.tolist()
        for text in [r["name"]] + r["rewrites"]:
            rankings.append([position[fdc_id] for _, fdc_id, _ in keyword_rank(text, fndds, idf, DEPTH)])

        score = defaultdict(float)
        for ranking in rankings:
            for rk, i in enumerate(ranking):
                score[i] += 1 / (RRF_K + rk)
        order = sorted(score, key=lambda i: -score[i])
        if rerank:
            pool = diversify(order, descriptions, rerank_pool, per_head)
            query = f'{r["name"]}. Ingredients: {", ".join(r["ingredients"])}'
            relevance = reranker.predict([(query, doc_texts[i]) for i in pool], batch_size=32)
            score = {i: float(s) for i, s in zip(pool, relevance)}
            order = sorted(pool, key=lambda i: -score[i])
        for rk, i in enumerate(diversify(order, descriptions, k, per_head), 1):
            rows.append([key, r["name"], "; ".join(r["ingredients"]), rk, round(score[i], 4),
                         fndds[i][0], descriptions[i]])
    write_csv(out, ["recipe_key", "recipe_name", "ingredients", "rank", "rrf_score", "fdc_id", "description"], rows)
    print(f"wrote {out} ({len(recipes)} recipes x top-{k})")


# ----------------------------------------------------- llm-match / llm-review

def fndds_ingredients():
    """fdc_id → FNDDS 구성 재료 문자열 (prepare 가 만든 fndds_recipes.csv)."""
    return {r["fdc_id"]: r["ingredients"] for r in read_csv(FNDDS_RECIPES)}


def load_candidates(top, path=CANDIDATES):
    candidates = defaultdict(list)
    for r in read_csv(path):
        if int(r["rank"]) <= top:
            candidates[r["recipe_key"]].append((r["fdc_id"], r["description"]))
    return candidates


def candidate_lines(candidates, ingredients):
    """후보마다 FNDDS 이름과 구성 재료를 함께 보여 준다 (이름만으로는 내용물을 알기 어려움)."""
    return ["Candidates (FNDDS name [its main recipe ingredients]):"] + [
        f"{i}. {desc} [{ingredients.get(fdc_id, '')}]" for i, (fdc_id, desc) in enumerate(candidates, 1)]


# docs/matching_rubric.md 의 규칙을 LLM 용 영문으로 옮긴 것. 둘을 함께 고친다.
RULES = """Rules, in priority order:
1. Choose only from the numbered candidates. Choose 0 when no candidate has the same dish type AND the same main components.
   A wrong dish type is worse than no match: never pick a raw ingredient for a cooked dish, a different dish for a sauce, etc.
2. Dish identity: the candidate must be the same kind of dish (curry, pizza, salad, soup, drink, ...).
3. Main ingredients: the ingredient list overrides the dish name (e.g. a "beef" dish made with plant-based burgers is vegetarian).
4. Preparation: if oil or butter is an ingredient prefer "fat added" / "made with oil"; otherwise "no added fat".
   The dish is home cooked: prefer "home recipe" over "restaurant", "ready-to-heat" or "frozen".
5. Prefer generic entries (NFS, plain) over brands or special variants (reduced sodium, gluten free, decaffeinated, egg white) unless the ingredients say so.
6. For a combination of two dishes, match the first dish named and mention the other in the reason.

match_type: "exact" = same dish and preparation; "approx" = similar dish or main component only; "none" = choice 0."""

SELECT_PROMPT = f"""You map a home-cooked dish to the single best matching entry of the USDA FNDDS food database.
You are given the dish name, its ingredient list, and a numbered list of FNDDS candidates (in random order),
each with the main ingredients of its FNDDS recipe. Compare ingredients, not only names.

{RULES}

Work in this order and write it into the JSON:
1. "dish_type": what kind of dish this is, in a few words.
2. "main_ingredients": the 1-4 ingredients that dominate the dish's nutrition.
3. "fat_added": true if oil, butter or ghee is an ingredient.
4. "shortlist": the numbers of up to 3 candidates with the same dish type (empty if none).
5. "choice", "match_type", "reason": the final decision among the shortlist (0 if the shortlist is empty).

Answer with one JSON object only:
{{"dish_type": "...", "main_ingredients": ["..."], "fat_added": true, "shortlist": [<n>, ...], "choice": <n or 0>, "match_type": "exact|approx|none", "reason": "<one short sentence>"}}"""


def validate_selection(n_candidates):
    def validate(ans):
        choice = ans.get("choice")
        if not isinstance(choice, int) or not 0 <= choice <= n_candidates:
            return f"choice must be an integer between 0 and {n_candidates}"
        if ans.get("match_type") not in ("exact", "approx", "none") or (choice == 0) != (ans["match_type"] == "none"):
            return 'match_type must be exact/approx/none, and "none" exactly when choice is 0'
        return None
    return validate


def vote_confidence(share):
    return "high" if share >= 0.8 else "medium" if share >= 0.6 else "low"


def llm_match(out=MAPPING_SELECT, model_name=SELECT_MODEL, top=30, runs=5, seed=0, quantize="8bit", gpu=0):
    """선택 모델이 레시피마다 후보 순서를 섞어 runs 번 고르고 다수결로 정한다. 정답 영양값은 쓰지 않는다.

    순서를 섞는 이유: LLM 은 목록 앞쪽을 고르는 경향(위치 편향)이 있다. 다수결 득표율(vote_share)을
    confidence 로 쓴다 (0.8 이상 high, 0.6 이상 medium, 그 외 low). 원본 응답은 <out>.jsonl 에 남긴다.
    """
    import random

    candidates = load_candidates(top)
    ingredients = fndds_ingredients()
    recipes = load_recipes()
    generate = load_llm(model_name, quantize, gpu)

    rows, log = [], []
    for key, r in recipes.items():
        cands = candidates[key]
        votes = []  # (fdc_id 또는 "", match_type, reason)
        for run in range(runs):
            order = list(range(len(cands)))
            random.Random(f"{seed}-{key}-{run}").shuffle(order)
            shown = [cands[i] for i in order]
            ans, attempts = ask_json(generate, [{"role": "system", "content": SELECT_PROMPT},
                                                {"role": "user", "content": "\n".join(
                                                    dish_lines(r) + candidate_lines(shown, ingredients))}],
                                     validate_selection(len(shown)))
            log.append({"recipe_key": key, "run": run, "order": order, "attempts": attempts})
            if ans:
                fdc_id = shown[ans["choice"] - 1][0] if ans["choice"] else ""
                votes.append((fdc_id, ans["match_type"], ans.get("reason", "")))

        if not votes:
            rows.append([key, r["name"], "", "", "none", "low", "LLM output invalid in every run", 0, ""])
            continue
        tally = Counter(v[0] for v in votes)
        rank = {fdc_id: i for i, (fdc_id, _) in enumerate(cands)}
        # 동점이면 검색 순위가 높은 후보, none("")은 동점에서 가장 뒤
        winner = max(tally, key=lambda f: (tally[f], -rank.get(f, len(cands))))
        winner_votes = [v for v in votes if v[0] == winner]
        match_type = Counter(v[1] for v in winner_votes).most_common(1)[0][0]
        share = round(tally[winner] / runs, 2)
        desc = dict(cands).get(winner, "")
        rows.append([key, r["name"], winner, desc, match_type, vote_confidence(share), winner_votes[0][2],
                     share, json.dumps(dict(tally))])
        print(f'{key} {r["name"]} -> {desc or "(none)"} [{match_type}, votes {share}]', flush=True)

    write_csv(out, MAPPING_COLUMNS + ["vote_share", "votes"], rows)
    write_jsonl(Path(out).with_suffix(".jsonl"), log)
    print(f"wrote {out}")


REVIEW_PROMPT = f"""You review a proposed mapping of a home-cooked dish to an entry of the USDA FNDDS food database.
Another model chose the proposed entry from the numbered candidates (each shown with the main ingredients of its
FNDDS recipe). Check it independently against the rules, comparing ingredients, not only names.

{RULES}

Decide:
- "accept" if the proposed entry is the best candidate under the rules (or, for a proposed 0, if truly no candidate fits).
- "reject" otherwise, and give "better_choice": the number of the best candidate, or 0 if no candidate fits.

Answer with one JSON object only:
{{"verdict": "accept|reject", "better_choice": <n, 0, or null when accepting>, "reason": "<one short sentence>"}}"""


def validate_review(n_candidates, proposed):
    def validate(ans):
        if ans.get("verdict") not in ("accept", "reject"):
            return "verdict must be accept or reject"
        if ans["verdict"] == "reject":
            better = ans.get("better_choice")
            if not isinstance(better, int) or not 0 <= better <= n_candidates:
                return f"when rejecting, better_choice must be an integer between 0 and {n_candidates}"
            if better == proposed:
                return "better_choice must differ from the proposed entry when rejecting"
        return None
    return validate


def llm_review(selection=MAPPING_SELECT, out=MAPPING, model_name=REVIEW_MODEL, top=30, quantize="8bit", gpu=0,
               runs=1, seed=0, none_policy="none"):
    """검토 모델(선택 모델과 다른 계열)이 선택 결과를 검토하고, 아래 규칙으로 최종 매핑을 정한다.

    - accept: 선택 결과 유지, confidence 는 선택 단계의 득표율 기준
    - reject + 다른 후보 c: 선택 단계에서 c 가 1표 이상 받았으면 c 로 교체, 아니면 유지. 둘 다 confidence low
    - reject + 0(맞는 후보 없음): none_policy="none" 이면 선택 득표율 0.8 미만일 때 none, 아니면 유지.
      none_policy="keep" 이면 항상 선택 결과 유지 (대략 맞는 항목이 매칭 없음보다 영양소 조회에 쓸모 있음).
      어느 쪽이든 confidence low
    runs > 1 이면 후보 순서를 섞어 runs 번 검토하고 판정(accept/reject)과 대안(better_choice)을 다수결로 정한다.
    원본 응답은 <out>.jsonl 에 남긴다.
    """
    import random

    candidates = load_candidates(top)
    ingredients = fndds_ingredients()
    recipes = load_recipes()
    generate = load_llm(model_name, quantize, gpu)

    rows, log = [], []
    for s in read_csv(selection):
        key, r = s["recipe_key"], recipes[s["recipe_key"]]
        cands = candidates[key]
        answers = []  # (verdict, better fdc_id 또는 "", reason)
        for run in range(runs):
            order = list(range(len(cands)))
            if run:  # 첫 실행은 검색 순서 그대로
                random.Random(f"review-{seed}-{key}-{run}").shuffle(order)
            shown = [cands[i] for i in order]
            number = {fdc_id: i for i, (fdc_id, _) in enumerate(shown, 1)}
            proposed = number.get(s["fdc_id"], 0)
            user = "\n".join(dish_lines(r) + candidate_lines(shown, ingredients)
                             + [f"Proposed: {proposed}. {s['fndds_description']}" if proposed else "Proposed: 0 (no match)"])
            one, attempts = ask_json(generate, [{"role": "system", "content": REVIEW_PROMPT},
                                                {"role": "user", "content": user}],
                                     validate_review(len(shown), proposed))
            log.append({"recipe_key": key, "run": run, "order": order, "attempts": attempts})
            if one:
                better = shown[one["better_choice"] - 1][0] if one["verdict"] == "reject" and one["better_choice"] else ""
                answers.append((one["verdict"], better, one.get("reason", "")))
        if answers:
            verdict_major = Counter(a[0] for a in answers).most_common(1)[0][0]
            chosen = [a for a in answers if a[0] == verdict_major]
            better_major = Counter(a[1] for a in chosen).most_common(1)[0][0] if verdict_major == "reject" else ""
            pick = next(a for a in chosen if a[1] == better_major)
            ans = {"verdict": verdict_major, "better_id": better_major, "reason": pick[2]}
        else:
            ans = None

        votes = json.loads(s["votes"]) if s["votes"] else {}
        share = float(s["vote_share"] or 0)
        fdc_id, desc, match_type, confidence = s["fdc_id"], s["fndds_description"], s["match_type"], s["confidence"]
        if ans is None:
            verdict, better_id, rule = "invalid", "", "검토 출력 오류: 선택 결과 유지(low)"
            confidence = "low"
        elif ans["verdict"] == "accept":
            verdict, better_id, rule = "accept", "", "검토 통과"
        else:
            verdict = "reject"
            better_id = ans["better_id"]
            confidence = "low"
            if better_id and votes.get(better_id):
                fdc_id, desc, match_type = better_id, dict(cands)[better_id], "approx"
                rule = "검토 거부: 선택 단계에서 표를 받은 검토 모델 제안으로 교체"
            elif not better_id and share < 0.8 and none_policy == "none":
                fdc_id, desc, match_type = "", "", "none"
                rule = "검토 거부: 맞는 후보 없음 + 선택 득표율 0.8 미만 → none"
            else:
                rule = "검토 거부: 근거 부족으로 선택 결과 유지"
        reason = s["reason"] if fdc_id == s["fdc_id"] else ans["reason"]
        rows.append([key, s["recipe_name"], fdc_id, desc, match_type, confidence, reason,
                     s["vote_share"], verdict, better_id, ans["reason"] if ans else "", rule])
        print(f'{key} {s["recipe_name"]} -> {desc or "(none)"} [{verdict}; {rule}]', flush=True)

    write_csv(out, MAPPING_COLUMNS + ["vote_share", "review_verdict", "review_choice", "review_reason", "final_rule"],
              rows)
    write_jsonl(Path(out).with_suffix(".jsonl"), log)
    print(f"wrote {out}")


# ---------------------------------------------------------------------- judge

JUDGE_PROMPT = """You are an independent judge of food-database mappings.
A home-cooked dish was mapped to one entry of the USDA FNDDS database, whose nutrient values will be used for the dish.
Rate how well the FNDDS entry represents the dish, considering dish type, main ingredients and preparation.

5 = same dish with the same main ingredients and preparation
4 = same dish; minor differences (variant, cooking fat, a missing side or topping)
3 = similar dish type or only the main component matches
2 = related food but a different dish (e.g. a raw ingredient for a cooked dish, a sauce for a full dish)
1 = wrong food

Answer with one JSON object only: {"score": <1-5>, "reason": "<one short sentence>"}"""


def validate_judgement(ans):
    if not isinstance(ans.get("score"), int) or not 1 <= ans["score"] <= 5:
        return "score must be an integer from 1 to 5"
    return None


def judge(mappings, out_dir=RESULTS / "judge", model_name=JUDGE_MODEL, quantize="4bit", gpu=0):
    """평가 전용 심사 모델(선택·검토 모델과 다른 계열)이 매핑마다 요리와 FNDDS 항목의 유사도를 1~5점으로 매긴다.

    심사 모델은 요리 이름·식재료와 FNDDS 이름·구성 재료만 본다 (매칭 방법, confidence, reason 은 보여 주지 않음).
    매칭 없음(none)은 1점으로 센다. 같은 (recipe_key, fdc_id) 쌍은 한 번만 채점해 여러 매핑을 같은 기준으로 비교한다.
    파이프라인 안에서는 쓰지 않는다 (심사 점수로 후보를 고르면 점수만 오르고 실제 품질은 보장되지 않음).
    """
    recipes = load_recipes()
    ingredients = fndds_ingredients()
    names = {r["fdc_id"]: r["description"] for r in read_csv(FNDDS_RECIPES)}
    tables = {f"{Path(m).parent.name}__{Path(m).stem}": read_csv(m) for m in mappings}  # 예: v2__vnutri_fndds_map
    pairs = sorted({(r["recipe_key"], r["fdc_id"]) for rows in tables.values() for r in rows if r["fdc_id"]})

    generate = load_llm(model_name, quantize, gpu)
    scores, log = {}, []
    for key, fdc_id in pairs:
        r = recipes[key]
        user = "\n".join(dish_lines(r) + [f"FNDDS entry: {names[fdc_id]}",
                                          f"FNDDS recipe ingredients: {ingredients.get(fdc_id, '')}"])
        ans, attempts = ask_json(generate, [{"role": "system", "content": JUDGE_PROMPT},
                                            {"role": "user", "content": user}], validate_judgement)
        log.append({"recipe_key": key, "fdc_id": fdc_id, "attempts": attempts})
        scores[key, fdc_id] = (ans["score"], ans.get("reason", "")) if ans else (None, "judge output invalid")
        print(f'{key} {r["name"]} | {names[fdc_id]} -> {scores[key, fdc_id][0]}', flush=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_jsonl(Path(out_dir) / "judge_log.jsonl", log)
    summary = {"judge_model": model_name}
    for name, rows in tables.items():
        out_rows, values = [], []
        for r in rows:
            score, reason = scores.get((r["recipe_key"], r["fdc_id"]), (1, "no match")) if r["fdc_id"] else (1, "no match")
            out_rows.append([r["recipe_key"], r["recipe_name"], r["fdc_id"], r["fndds_description"], score, reason])
            if score is not None:
                values.append(score)
        write_csv(Path(out_dir) / f"{name}.csv",
                  ["recipe_key", "recipe_name", "fdc_id", "fndds_description", "judge_score", "judge_reason"], out_rows)
        summary[name] = {"mean_score": round(sum(values) / len(values), 3), "n": len(values),
                         "score_distribution": dict(sorted(Counter(values).items())),
                         "share_ge_4": round(sum(v >= 4 for v in values) / len(values), 3)}
    with open(Path(out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ------------------------------------------------------- evaluate / compare

def energy_share(protein, fat, carb):
    energy = 4 * protein + 9 * fat + 4 * carb
    return (4 * protein / energy, 9 * fat / energy, 4 * carb / energy) if energy else None


def macro_distance_fn():
    """(fdc_id, recipe_key) → FNDDS 와 V-Nutri 실측값의 단백질·지방·탄수화물 열량 비율 L1 거리 (0~2).

    V-Nutri 에는 무게가 없어서 절대량 대신 비율을 비교한다. 매칭 단계에서는 쓰지 않는 독립 지표다.
    계산할 수 없으면(매칭 없음, 열량 0) "" 를 돌려준다.
    """
    fndds = {r["fdc_id"]: r for r in read_csv(FNDDS_TABLE)}
    truth = {}
    for r in read_csv(VNUTRI_DISH_INDEX):
        truth.setdefault(r["recipe_key"], energy_share(
            float(r["total_protein_g"]), float(r["total_fat_g"]), float(r["total_carbs_g"])))

    def distance(fdc_id, key):
        row = fndds.get(fdc_id)
        a, b = (energy_share(*(float(row[c]) for c in MACROS)) if row else None), truth[key]
        return round(sum(abs(x - y) for x, y in zip(a, b)), 2) if a and b else ""
    return distance


def evaluate(mapping=MAPPING, others=(), out=RESULTS / "eval_report.json"):
    """정답 매핑 없이 매핑 테이블을 평가한다.

    - 분포: match_type, confidence
    - 일관성: 선택 모델 득표율(순서를 섞어도 같은 답을 내는가), 검토 모델 통과율
    - 영양 일관성: V-Nutri 실측값과의 열량 비율 거리. 매칭에 쓰지 않은 독립 지표이고,
      confidence 가 높을수록 거리가 작아야 confidence 가 의미가 있다
    - 다른 매핑(others)과의 일치율 (참고용)
    """
    rows = read_csv(mapping)
    distance = macro_distance_fn()
    dists = {r["recipe_key"]: distance(r["fdc_id"], r["recipe_key"]) for r in rows}

    def mean(values):
        values = [v for v in values if v != ""]
        return round(sum(values) / len(values), 3) if values else None

    report = {
        "mapping": str(mapping),
        "n": len(rows),
        "match_type": dict(Counter(r["match_type"] for r in rows)),
        "confidence": dict(Counter(r["confidence"] for r in rows)),
        "macro_dist_mean": mean(dists.values()),
        "macro_dist_by_confidence": {c: mean(dists[r["recipe_key"]] for r in rows if r["confidence"] == c)
                                     for c in ("high", "medium", "low")},
    }
    if "vote_share" in rows[0]:
        report["vote_share_mean"] = mean(float(r["vote_share"]) for r in rows if r["vote_share"])
    if "review_verdict" in rows[0]:
        report["review_verdict"] = dict(Counter(r["review_verdict"] for r in rows))
        report["final_rule"] = dict(Counter(r["final_rule"] for r in rows))
    by_key = {r["recipe_key"]: r["fdc_id"] for r in rows}
    report["agreement"] = {}
    for other in others:
        other_rows = read_csv(other)
        report["agreement"][str(other)] = {
            "same_fdc_id": sum(by_key[o["recipe_key"]] == o["fdc_id"] for o in other_rows),
            "macro_dist_mean": mean(distance(o["fdc_id"], o["recipe_key"]) for o in other_rows),
        }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def compare(base, new, out):
    """두 매핑의 차이 목록: fdc_id 가 다르거나, new 의 confidence 가 low 이거나 match_type 이 none 인 항목."""
    base_rows = {r["recipe_key"]: r for r in read_csv(base)}
    new_rows = read_csv(new)
    distance = macro_distance_fn()

    header = ["recipe_key", "recipe_name", "flag",
              "new_fdc_id", "new_description", "new_match_type", "new_confidence", "new_reason", "new_macro_dist",
              "base_fdc_id", "base_description", "base_reason", "base_macro_dist"]
    queue, agree = [], 0
    for r in new_rows:
        key, old = r["recipe_key"], base_rows[r["recipe_key"]]
        same = r["fdc_id"] == old["fdc_id"]
        agree += same
        flags = [x for x, hit in [("different", not same), ("low", r.get("confidence") == "low"),
                                  ("none", r["match_type"] == "none")] if hit]
        if flags:
            queue.append([key, r["recipe_name"], "+".join(flags),
                          r["fdc_id"], r["fndds_description"], r["match_type"], r.get("confidence", ""),
                          r.get("reason", ""), distance(r["fdc_id"], key),
                          old["fdc_id"], old["fndds_description"], old.get("reason", ""), distance(old["fdc_id"], key)])
    write_csv(out, header, queue)
    print(f"agree {agree}/{len(new_rows)}, flagged {len(queue)} -> {out}")


# --------------------------------------------------------------------- lookup

def _to_number(value):
    return float(value) if value != "" else None


class NutrientLookup:
    """V-Nutri dish_id → vnutri_ids.csv(recipe_key) → 매핑 테이블(fdc_id) → FNDDS 영양소(100g 당)."""

    def __init__(self, mapping=MAPPING, vnutri_ids=VNUTRI_IDS, fndds=FNDDS_TABLE):
        self.dishes = {r["dish_id"]: r for r in read_csv(vnutri_ids)}
        self.mapping = {r["recipe_key"]: r for r in read_csv(mapping)}
        rows = read_csv(fndds)
        self.fndds = {r["fdc_id"]: r for r in rows}
        self.nutrient_columns = [c for c in rows[0] if c not in META_COLUMNS]

        missing = [k for k, m in self.mapping.items() if m["fdc_id"] and m["fdc_id"] not in self.fndds]
        if missing:
            raise ValueError(f"FNDDS 에 없는 fdc_id 가 매핑에 있음: {missing}")

    def lookup(self, dish_id, nutrients=None):
        """V-Nutri dish_id 하나에 대한 매칭 정보와 100g 당 영양소를 반환한다."""
        dish = self.dishes.get(str(dish_id))
        if dish is None:
            raise KeyError(f"알 수 없는 V-Nutri dish_id: {dish_id}")
        return {"dish_id": dish["dish_id"], **self.lookup_recipe(dish["recipe_key"], nutrients)}

    def lookup_recipe(self, recipe_key, nutrients=None):
        m = self.mapping.get(recipe_key)
        if m is None:
            raise KeyError(f"매핑되지 않은 recipe_key: {recipe_key}")
        columns = nutrients or self.nutrient_columns
        unknown = [c for c in columns if c not in self.nutrient_columns]
        if unknown:
            raise KeyError(f"알 수 없는 영양소 열: {unknown}")
        row = self.fndds.get(m["fdc_id"])  # match_type 'none' 이면 fdc_id 가 비어 있음
        return {
            "recipe_key": recipe_key,
            "recipe_name": m["recipe_name"],
            "fdc_id": m["fdc_id"] or None,
            "fndds_description": row["description"] if row else None,
            "match_type": m["match_type"],
            "confidence": m.get("confidence"),
            "note": m.get("reason", ""),
            "basis": row["basis"] if row else None,
            "nutrients": {c: _to_number(row[c]) if row else None for c in columns},
        }


def print_lookup(dish_ids, nutrients, as_json, mapping):
    lk = NutrientLookup(mapping=mapping)
    results = [lk.lookup(d, nutrients) for d in dish_ids]
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    for r in results:
        if r["fdc_id"] is None:
            print(f'[{r["dish_id"]}] {r["recipe_name"]} → 매칭 없음 ({r["match_type"]})')
        else:
            print(f'[{r["dish_id"]}] {r["recipe_name"]} → {r["fndds_description"]} '
                  f'(fdc_id {r["fdc_id"]}, {r["match_type"]}/{r["confidence"]}, per {r["basis"]})')
        if r["note"]:
            print(f'    note: {r["note"]}')
        for name, value in r["nutrients"].items():
            print(f"    {name}: {value}")


# ----------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="원본 → data/ 가공 데이터")

    def llm_options(s, model):
        s.add_argument("--model", default=model)
        s.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="8bit")
        s.add_argument("--gpu", type=int, default=0, help="모델을 올릴 GPU 번호 (한 장만 사용)")

    s = sub.add_parser("rewrite", help="LLM 쿼리 재작성 → data/query_rewrites.csv")
    s.add_argument("--out", type=Path, default=QUERY_REWRITES)
    llm_options(s, SELECT_MODEL)

    s = sub.add_parser("retrieve", help="후보 검색 → data/retrieval_candidates.csv")
    s.add_argument("--out", type=Path, default=CANDIDATES)
    s.add_argument("--model", default=EMBED_MODEL)
    s.add_argument("--rewrites", type=Path, default=QUERY_REWRITES)
    s.add_argument("-k", type=int, default=30)
    s.add_argument("--per-head", type=int, default=6)
    s.add_argument("--rerank", action="store_true", help="크로스 인코더로 후보 재정렬")
    s.add_argument("--rerank-pool", type=int, default=80)

    s = sub.add_parser("llm-match", help="선택 모델 다수결 → results/vnutri_fndds_map_select.csv")
    s.add_argument("--out", type=Path, default=MAPPING_SELECT)
    s.add_argument("--top", type=int, default=30, help="LLM 에 보여 줄 후보 수")
    s.add_argument("--runs", type=int, default=5, help="후보 순서를 섞어 실행할 횟수")
    s.add_argument("--seed", type=int, default=0)
    llm_options(s, SELECT_MODEL)

    s = sub.add_parser("llm-review", help="검토 모델 → results/vnutri_fndds_map.csv (최종)")
    s.add_argument("--selection", type=Path, default=MAPPING_SELECT)
    s.add_argument("--out", type=Path, default=MAPPING)
    s.add_argument("--top", type=int, default=30)
    s.add_argument("--runs", type=int, default=1, help="후보 순서를 섞어 검토할 횟수 (다수결)")
    s.add_argument("--none-policy", choices=["none", "keep"], default="none",
                   help="검토 모델이 '맞는 후보 없음'이라고 할 때: none 으로 바꿀지, 선택 결과를 유지할지")
    s.add_argument("--seed", type=int, default=0)
    llm_options(s, REVIEW_MODEL)

    s = sub.add_parser("judge", help="평가 전용 심사 모델이 매핑을 1~5점으로 채점 → results/judge/")
    s.add_argument("mappings", type=Path, nargs="+", help="채점할 매핑 CSV (여러 개 가능)")
    s.add_argument("--out-dir", type=Path, default=RESULTS / "judge")
    llm_options(s, JUDGE_MODEL)
    s.set_defaults(quantize="4bit")

    s = sub.add_parser("evaluate", help="정답 매핑 없이 평가 → results/eval_report.json")
    s.add_argument("--mapping", type=Path, default=MAPPING)
    s.add_argument("--others", type=Path, nargs="*", default=[], help="일치율을 비교할 다른 매핑")
    s.add_argument("--out", type=Path, default=RESULTS / "eval_report.json")

    s = sub.add_parser("compare", help="두 매핑의 차이 목록")
    s.add_argument("--base", type=Path, required=True)
    s.add_argument("--new", type=Path, default=MAPPING)
    s.add_argument("--out", type=Path, default=RESULTS / "compare.csv")

    s = sub.add_parser("lookup", help="V-Nutri dish_id → 영양소")
    s.add_argument("dish_ids", nargs="+", help="V-Nutri dish_id (1~51)")
    s.add_argument("--nutrients", nargs="+", help="반환할 영양소 열 이름 (기본: 전체 65개)")
    s.add_argument("--json", action="store_true", help="JSON 으로 출력")
    s.add_argument("--mapping", type=Path, default=MAPPING, help="매핑 테이블 CSV")

    a = p.parse_args()
    if a.command == "prepare":
        prepare()
    elif a.command == "rewrite":
        rewrite(a.out, a.model, a.quantize, a.gpu)
    elif a.command == "retrieve":
        retrieve(a.out, a.model, a.k, a.per_head, a.rewrites, a.rerank, a.rerank_pool)
    elif a.command == "llm-match":
        llm_match(a.out, a.model, a.top, a.runs, a.seed, a.quantize, a.gpu)
    elif a.command == "llm-review":
        llm_review(a.selection, a.out, a.model, a.top, a.quantize, a.gpu, a.runs, a.seed, a.none_policy)
    elif a.command == "judge":
        judge(a.mappings, a.out_dir, a.model, a.quantize, a.gpu)
    elif a.command == "evaluate":
        evaluate(a.mapping, a.others, a.out)
    elif a.command == "compare":
        compare(a.base, a.new, a.out)
    elif a.command == "lookup":
        print_lookup(a.dish_ids, a.nutrients, a.json, a.mapping)


if __name__ == "__main__":
    main()
