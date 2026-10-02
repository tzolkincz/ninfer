#!/usr/bin/env python3
"""Stream a chat completion and report timing metrics.

Usage: python3 bench_stream.py <port> <prompt> <max_tokens>
Output: "OK <prompt_tok> <comp_tok> <ttft_ms> <total_ms> <decode_tps> <prefill_tps>"
       or "FAIL <reason>"
"""
import json
import sys
import time
import urllib.request


def main():
    port = sys.argv[1]
    prompt = sys.argv[2]
    max_tokens = int(sys.argv[3])

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = json.dumps({
        "model": "quasar",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0

    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ttft is None and chunk.get("choices"):
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content") or delta.get("reasoning"):
                        ttft = time.monotonic() - start
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    except Exception as e:
        print(f"FAIL {e}")
        return

    total = time.monotonic() - start

    if ttft is None:
        print("FAIL no tokens received")
        return
    if completion_tokens < 2:
        print(f"FAIL only {completion_tokens} tokens")
        return

    decode_time = total - ttft
    prefill_tps = prompt_tokens / ttft if ttft > 0.001 else 0
    decode_tps = completion_tokens / decode_time if decode_time > 0.001 else 0
    print(f"OK {prompt_tokens} {completion_tokens} {ttft*1000:.0f} {total*1000:.0f} {decode_tps:.1f} {prefill_tps:.1f}")


if __name__ == "__main__":
    main()
