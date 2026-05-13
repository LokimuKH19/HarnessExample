"""
run.py — 本地调试脚本

使用 data/train_dev.jsonl 作为训练流，data/test_dev.jsonl 作为本地验证集，
测试你的 MyHarness 实现。

用法
----
  python run.py                         # 默认评测
  python run.py --workers 100           # 调整向 LLM 发消息的并发数，防止因超时等导致错误

注意
----
- 最终评分使用的训练集与测试集与此不同，请勿过拟合 DEV 集。
- 每次 call_llm 的 prompt token 超过 max_prompt_tokens 会被截断；
  正式评分系统行为相同。
"""

import argparse
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import call_llm as _raw_call_llm, count_tokens, count_messages_tokens, truncate_to_tokens
from solution import MyHarness


def make_controlled_llm(max_prompt_tokens: int, tracker: dict, lock: threading.Lock):
    def _call(messages: list[dict]) -> str:
        prompt_text = " ".join(m.get("content", "") for m in messages)
        n = count_tokens(prompt_text)
        if n > max_prompt_tokens:
            messages = list(messages)
            excess = n - max_prompt_tokens
            for i in range(len(messages) - 1, -1, -1):
                if excess <= 0:
                    break
                content = messages[i].get("content", "")
                msg_tokens = count_tokens(content)
                if msg_tokens <= excess:
                    messages[i] = {**messages[i], "content": ""}
                    excess -= msg_tokens
                else:
                    messages[i] = {**messages[i], "content": truncate_to_tokens(content, msg_tokens - excess)}
                    excess = 0
            truncated_by = n - max_prompt_tokens
            n = count_tokens(" ".join(m.get("content", "") for m in messages))
            print(f"[WARNING] prompt truncated by {truncated_by} tokens (budget={max_prompt_tokens})", file=sys.stderr)
        resp = _raw_call_llm(messages)
        with lock:
            tracker["prompt"]     += n
            tracker["completion"] += count_tokens(resp)
        return resp
    return _call


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",             default="data/train_dev.jsonl")
    parser.add_argument("--dev",               default="data/test_dev.jsonl")
    parser.add_argument("--workers",     type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--runs",        type=int, default=1)
    args = parser.parse_args()

    train = load_jsonl(args.train)
    dev   = load_jsonl(args.dev)

    print("=" * 60)
    print("  本地调试评测")
    print("=" * 60)
    print(f"  Train: {len(train)} 条 | Dev: {len(dev)} 条")
    print(f"  max_prompt_tokens: {args.max_prompt_tokens} | runs: {args.runs}\n")

    all_accuracies = []
    total_tracker  = {"prompt": 0, "completion": 0}
    total_elapsed  = 0.0

    for run_idx in range(args.runs):
        print(f"  [Run {run_idx + 1}/{args.runs}]")
        tracker = {"prompt": 0, "completion": 0}
        lock    = threading.Lock()
        llm     = make_controlled_llm(args.max_prompt_tokens, tracker, lock)

        harness = MyHarness(llm, count_tokens, count_messages_tokens, args.max_prompt_tokens)
        for item in train:
            harness.update(item["text"], item["label"])

        predictions = [None] * len(dev)
        error_log   = []
        t0 = time.time()

        def run_one(args_):
            idx, item = args_
            try:
                pred = harness.predict(item["text"])
                return idx, pred.strip(), None
            except Exception as e:
                return idx, "", str(e)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_one, (i, item)) for i, item in enumerate(dev)]
            done = 0
            for fut in as_completed(futures):
                idx, pred, err = fut.result()
                predictions[idx] = pred
                if err:
                    error_log.append((idx, err))
                done += 1
                sys.stdout.write(f"\r    进度: {done}/{len(dev)}")
                sys.stdout.flush()
        print()

        correct  = sum(1 for item, pred in zip(dev, predictions) if pred == item["label"])
        accuracy = correct / len(dev) * 100
        elapsed  = time.time() - t0
        all_accuracies.append(accuracy)
        total_tracker["prompt"]     += tracker["prompt"]
        total_tracker["completion"] += tracker["completion"]
        total_elapsed += elapsed

        print(f"    准确率={accuracy:.1f}%  耗时={elapsed:.1f}s", end="")
        if error_log:
            print(f"  错误={len(error_log)}", end="")
            for idx, err in sorted(error_log):
                print(f"\n      [#{idx}] {err[:120]}", end="")
        print()

    n       = len(dev)
    runs    = args.runs
    avg_acc = sum(all_accuracies) / runs
    print(f"\n{'=' * 60}")
    print(f"  平均准确率: {avg_acc:.1f}%  (各轮: {', '.join(f'{a:.1f}%' for a in all_accuracies)})")
    print(f"  prompt/条:  {total_tracker['prompt'] // (n * runs)} token")
    print(f"  compl/条:   {total_tracker['completion'] / (n * runs):.1f} token")
    print(f"  总耗时:     {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
