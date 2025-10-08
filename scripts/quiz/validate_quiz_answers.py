#!/usr/bin/env python3
"""Validate user answers against quiz & answer key (standalone copy)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List

def parse_args(argv: List[str]):
    p = argparse.ArgumentParser(description="Validate quiz answers")
    p.add_argument("--quiz", default="quiz.json")
    p.add_argument("--answers", default="answer_key.json")
    p.add_argument("--user")
    p.add_argument("--show-correct-first", action="store_true")
    p.add_argument("--raw", choices=["none", "summary", "full"], default="none",
                   help="How to display raw_response: none, summary, or full (default: none)")
    p.add_argument("--raw-truncate", type=int, default=1200,
                   help="Max characters to print for raw_output before truncating (default: 1200)")
    return p.parse_args(argv)

def load_json(path: Path): return json.loads(path.read_text(encoding="utf-8"))

def format_raw_response(raw: Any, mode: str, truncate: int) -> str:
    if mode == "none":
        return ""
    if not isinstance(raw, dict):
        s = str(raw)
        return (s[:truncate] + ("... [truncated]" if len(s) > truncate else "")) if s else ""
    obj = dict(raw)
    resp = obj.get("response")
    if isinstance(resp, str):
        text = resp.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                import json as _json
                obj["response"] = _json.loads(text)
            except Exception:
                obj["response"] = text.replace("\\n", "\n")
    def _summarize(o: Any) -> Any:
        if mode == "summary" and isinstance(o, list) and len(o) > 20:
            return f"<list of {len(o)} items>"
        return o
    if mode == "summary":
        for k in ["context", "tokens", "prompt", "prompt_token_ids"]:
            if k in obj and isinstance(obj[k], list):
                obj[k] = f"<list of {len(obj[k])} items>"
        if isinstance(obj.get("response"), list):
            short = []
            for item in obj["response"][:3]:
                if isinstance(item, dict):
                    short.append({
                        k: _summarize(v) if k in ("options",) else v
                        for k, v in item.items() if k in ("id", "question", "answer", "topic", "difficulty", "options")
                    })
                else:
                    short.append(item)
            total = len(obj["response"])  # type: ignore
            obj["response"] = {
                "items_preview": short,
                "total_items": total
            }
    import json as _json
    s = _json.dumps(obj, indent=2, ensure_ascii=False)
    return (s[:truncate] + ("... [truncated]" if len(s) > truncate else ""))

def interactive_collect(quiz: List[Dict[str, Any]]) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    print("Enter your answers (A-D). Press Enter to skip (counts as incorrect).\n")
    for q in quiz:
        print(f"{q['id']}: {q['question']}")
        for idx, opt in enumerate(q['options']):
            letter = chr(ord('A') + idx)
            print(f"  {letter}. {opt}")
        while True:
            val = input("Answer (A-D): ").strip().upper()
            if val == "": print("(skipped)\n"); break
            if val in ["A", "B", "C", "D"]: answers[q['id']] = val; print(); break
            print("Please enter A-D or leave blank to skip.")
    return answers

def score(quiz: List[Dict[str, Any]], key: Dict[str, Any], user_answers: Dict[str, str], show_correct_first: bool, raw_mode: str, raw_truncate: int) -> None:
    total = len(quiz); correct = 0
    print("\n===== Results =====\n")
    for q in quiz:
        qid = q['id']; user_ans = user_answers.get(qid); key_entry = key.get(qid)
        if not key_entry: print(f"[warn] Missing answer key for {qid}"); continue
        correct_ans = key_entry['answer']; is_correct = user_ans == correct_ans
        if is_correct: correct += 1
        header = f"{'✅' if is_correct else '❌'} {qid}  Your: {user_ans or '-'}  Correct: {correct_ans}"
        print(header)
        explanation = key_entry.get('explanation','').strip()
        if explanation:
            if show_correct_first:
                print(f"   Explanation: {explanation}")
            else:
                print(f"   {explanation}")
        raw_response = key_entry.get('raw_response', '')
        raw_out = format_raw_response(raw_response, raw_mode, raw_truncate)
        if raw_out:
            print("   [Model Raw Response]:")
            print(raw_out)
    pct = (correct / total) * 100 if total else 0
    print(f"\nScore: {correct}/{total} = {pct:.1f}%")
    missed = [q for q in quiz if user_answers.get(q['id']) != key.get(q['id'], {}).get('answer')]
    if missed:
        print("\nMissed Questions:")
        for q in missed:
            aid = q['id']; correct_ans = key.get(aid, {}).get('answer', '-')
            user_ans = user_answers.get(aid) or '-'
            print(f"  - {aid}: Your {user_ans} → Correct {correct_ans}: {q['question'][:120]}")

def main(argv):
    args = parse_args(argv)
    quiz_path = Path(args.quiz); ans_path = Path(args.answers)
    if not quiz_path.exists() or not ans_path.exists(): print("[error] quiz or answer key path does not exist"); return 1
    quiz = load_json(quiz_path); key = load_json(ans_path)
    if not isinstance(quiz, list) or not isinstance(key, dict): print("[error] Invalid JSON structure"); return 1
    user_answers = load_json(Path(args.user)) if args.user else interactive_collect(quiz)
    score(quiz, key, user_answers, args.show_correct_first, args.raw, args.raw_truncate); return 0

if __name__ == "__main__":  # pragma: no cover
    import sys
    try: raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt: print("\nInterrupted."); raise SystemExit(130)

