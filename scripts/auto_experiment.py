"""사람 없이 실험 한 단계를 실행한다: 지금까지 가장 점수가 높은 실행 설정에 변경 하나를 적용해 돌리고 전체를 다시 채점한다.

    python scripts/auto_experiment.py v4 select_model=Qwen/Qwen2.5-32B-Instruct select_quantize=4bit
    python scripts/auto_experiment.py v5 rerank=true

- 기준 설정: results/judge/summary.json 에서 mean_score(동점이면 share_ge_4)가 가장 높은 실행 중
  results/<run>/config.json 이 있는 것
- 실행: retrieve → llm-match → llm-review → evaluate, 결과는 results/<stage>/
- 채점: config.json 이 있는 모든 실행과 results/baseline, results/reference/* 를 judge 로 다시 채점
- 기록: results/judge/leaderboard.md
쿼리 재작성(data/query_rewrites.csv)은 기존 것을 그대로 쓴다.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PY = sys.executable
MAP = "vnutri_fndds_map.csv"


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([PY, ROOT / "food_matcher.py", *map(str, args)], check=True, cwd=ROOT)


def parse_value(v):
    return {"true": True, "false": False}.get(v.lower(), int(v) if v.isdigit() else v)


def configured_runs():
    return sorted(p.parent.name for p in RESULTS.glob("*/config.json"))


def best_run():
    summary = json.loads((RESULTS / "judge" / "summary.json").read_text())
    scored = [(summary[f"{r}__vnutri_fndds_map"]["mean_score"], summary[f"{r}__vnutri_fndds_map"]["share_ge_4"], r)
              for r in configured_runs() if f"{r}__vnutri_fndds_map" in summary]
    return max(scored)[2]


def leaderboard():
    summary = json.loads((RESULTS / "judge" / "summary.json").read_text())
    lines = ["| run | judge mean | ≥4 비율 | macro_dist | config |", "|---|---|---|---|---|"]
    for key, v in sorted(((k, v) for k, v in summary.items() if isinstance(v, dict)),
                         key=lambda kv: -kv[1]["mean_score"]):
        run_dir = RESULTS / key.split("__")[0]
        cfg = run_dir / "config.json"
        ev = run_dir / "eval_report.json"
        dist = json.loads(ev.read_text()).get("macro_dist_mean") if ev.exists() else ""
        lines.append(f'| {key} | {v["mean_score"]} | {v["share_ge_4"]} | {dist} | '
                     f'{cfg.read_text().strip() if cfg.exists() else ""} |')
    (RESULTS / "judge" / "leaderboard.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    stage, changes = sys.argv[1], dict(c.split("=", 1) for c in sys.argv[2:])
    base = best_run()
    config = json.loads((RESULTS / base / "config.json").read_text())
    config.update({k: parse_value(v) for k, v in changes.items()})
    config["base"] = base
    out = RESULTS / stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False))
    print(f"stage {stage}: base={base} config={config}", flush=True)

    if "Qwen2.5-32B" in config["select_model"]:  # 다운로드가 덜 끝났으면 이어받는다
        from huggingface_hub import snapshot_download
        snapshot_download(config["select_model"], allow_patterns=["*.json", "*.safetensors", "tokenizer*",
                                                                   "merges.txt", "vocab.json"])

    k = config["k"]
    run("retrieve", "-k", k, *(["--rerank"] if config.get("rerank") else []))
    (out / "retrieval_candidates.csv").write_bytes((ROOT / "data" / "retrieval_candidates.csv").read_bytes())
    run("llm-match", "--top", k, "--model", config["select_model"], "--quantize", config["select_quantize"],
        "--out", out / "vnutri_fndds_map_select.csv")
    run("llm-review", "--top", k, "--runs", config.get("review_runs", 1), "--none-policy", config.get("none_policy", "none"),
        "--selection", out / "vnutri_fndds_map_select.csv", "--out", out / MAP)
    run("evaluate", "--mapping", out / MAP, "--out", out / "eval_report.json")

    mappings = [RESULTS / "baseline" / MAP] + [RESULTS / r / MAP for r in configured_runs()]
    mappings += sorted((RESULTS / "reference").glob("*.csv"))
    run("judge", *[m for m in mappings if m.exists()])
    leaderboard()


if __name__ == "__main__":
    main()
