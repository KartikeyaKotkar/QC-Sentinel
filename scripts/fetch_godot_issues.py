import argparse
import json
import os
import time

import requests

GITHUB_API_URL = "https://api.github.com/repos/godotengine/godot/issues"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)


def clean_body_text(body: str, max_chars: int = 2000) -> str:
    if not body:
        return ""
    lines = body.splitlines()
    filtered = [line for line in lines if not line.strip().startswith("<!--") and not line.strip().endswith("-->")]
    return "\n".join(filtered).strip()[:max_chars]


def fetch_godot_bugs(target_count: int = 500, output_path: str = "data/raw/godot_qc_bugs.json"):
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "QC-Sentinel-Scraper",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
        print("Using GitHub token (5000 req/hr).")
    else:
        print("No GITHUB_TOKEN (60 req/hr).")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    bugs: list[dict] = []
    page = 1
    per_page = 100

    print(f"Targeting {target_count} closed bug reports...")

    while len(bugs) < target_count:
        params = {
            "state": "closed",
            "labels": "bug",
            "per_page": per_page,
            "page": page,
            "sort": "updated",
            "direction": "desc",
        }
        resp = requests.get(GITHUB_API_URL, headers=headers, params=params)

        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining is not None:
            print(f"  RateLimit remaining: {remaining}", end=" | ")

        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(0, reset - int(time.time())) + 2
            print(f"\n[RateLimit] sleep {wait}s...")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print(f"\nFailed page {page}: {resp.status_code} - {resp.text[:500]}")
            break

        items = resp.json()
        if not items:
            print("\nNo more issues.")
            break

        for item in items:
            if "pull_request" in item:
                continue
            labels = [lbl["name"] for lbl in item.get("labels", [])]
            subsystems = [lbl.replace("topic:", "").strip() for lbl in labels if lbl.startswith("topic:")]
            subsystem = subsystems[0] if subsystems else "Core"
            is_labeled_duplicate = any("duplicate" in lbl.lower() for lbl in labels)
            cleaned = clean_body_text(item.get("body", ""))
            if len(cleaned) < 50:
                continue
            bugs.append(
                {
                    "bug_id": f"GODOT-{item['number']}",
                    "title": item.get("title", ""),
                    "description": cleaned,
                    "subsystem": subsystem,
                    "labels": labels,
                    "is_labeled_duplicate": is_labeled_duplicate,
                    "closed_at": item.get("closed_at"),
                    "url": item.get("html_url"),
                }
            )
            if len(bugs) >= target_count:
                break

        print(f"Collected {len(bugs)}/{target_count} (page {page})", end="\r")
        page += 1
        time.sleep(0.5)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bugs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(bugs)} bugs to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Godot bug issues")
    parser.add_argument("--count", type=int, default=500, help="Target bug count")
    parser.add_argument("--output", type=str, default="data/raw/godot_qc_bugs.json", help="Output path")
    args = parser.parse_args()
    fetch_godot_bugs(target_count=args.count, output_path=args.output)
