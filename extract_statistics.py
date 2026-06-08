#!/usr/bin/env python3
"""
GitHub Repository Usage Statistics Extractor
Extracts usage statistics from https://github.com/C2SM/zonda-request

Usage:
    python extract_statistics.py

Optional environment variable:
    GITHUB_TOKEN  — Personal access token for higher rate limits (30 search req/min
                    vs 10 unauthenticated).  No special scopes required for public repos.
                    Create one at https://github.com/settings/tokens
"""

import csv
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_OWNER = "C2SM"
REPO_NAME  = "zonda-request"
API_BASE   = "https://api.github.com"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# Only count issues created on or after this date (YYYY-MM-DD, inclusive).
# Set to None to include all issues.
START_DATE = "2025-05-01"

OUTPUT_DIR    = "."
CSV_FILE      = os.path.join(OUTPUT_DIR, "issues_per_month.csv")
PLOT_FILE     = os.path.join(OUTPUT_DIR, "issues_per_month.png")
README_FILE   = os.path.join(OUTPUT_DIR, "README.md")

# Markers that wrap the auto-generated statistics block inside README.md.
# Everything between these two lines is replaced on every run.
STATS_START = "<!-- STATS:START -->"
STATS_END   = "<!-- STATS:END -->"


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h


def _get_json(url: str, params: dict):
    """GET with automatic rate-limit back-off (works for both REST and search endpoints)."""
    for _ in range(6):
        r = requests.get(url, headers=_headers(), params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429):
            if "Retry-After" in r.headers:
                wait = int(r.headers["Retry-After"]) + 1
            else:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 62))
                wait  = max(reset - int(time.time()), 1)
            print(f"  Rate-limited — waiting {wait}s …")
            time.sleep(wait)
            continue
        if r.status_code == 401:
            print("Authentication error — check GITHUB_TOKEN.")
            sys.exit(1)
        print(f"HTTP {r.status_code}: {r.text[:200]}")
        sys.exit(1)
    print("Exceeded retry budget.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Fetching issues via search API (avoids the 1000-item page limit)
# ---------------------------------------------------------------------------

SEARCH_URL = f"{API_BASE}/search/issues"


def _search_month(start: str, end: str, page: int) -> dict:
    return _get_json(SEARCH_URL, {
        "q":        f"repo:{REPO_OWNER}/{REPO_NAME} is:issue created:{start}..{end}",
        "per_page": 100,
        "page":     page,
        "sort":     "created",
        "order":    "asc",
    })


def _repo_date_bounds() -> tuple:
    """Return (first_issue_dt, last_issue_dt, total_count) via two search calls."""
    first = _get_json(SEARCH_URL, {
        "q": f"repo:{REPO_OWNER}/{REPO_NAME} is:issue",
        "sort": "created", "order": "asc", "per_page": 1,
    })
    last = _get_json(SEARCH_URL, {
        "q": f"repo:{REPO_OWNER}/{REPO_NAME} is:issue",
        "sort": "created", "order": "desc", "per_page": 1,
    })
    if not first.get("items") or not last.get("items"):
        return None, None, 0
    return (
        _parse_dt(first["items"][0]["created_at"]),
        _parse_dt(last["items"][0]["created_at"]),
        first.get("total_count", 0),
    )


def fetch_all_issues() -> List[dict]:
    """
    Fetch every issue from the repo using the GitHub Search API with monthly
    date-range chunking.  This avoids GitHub's 1000-item hard cap on page-based
    pagination and works for repositories of any size.

    The search API allows up to 1 000 results per individual query; since no
    month typically has > 1 000 issues, one query per month is sufficient.
    Rate limits:  10 req/min (unauthenticated)  |  30 req/min (token)
    """
    print(f"Detecting repository date range …")
    first_dt, last_dt, total = _repo_date_bounds()
    if first_dt is None:
        print("No issues found in the repository.")
        return []

    effective_start = START_DATE if START_DATE else first_dt.strftime("%Y-%m-%d")
    print(f"  Total issues in repo : {total}")
    print(f"  Fetching from        : {effective_start} → {last_dt.strftime('%Y-%m-%d')}")
    if not GITHUB_TOKEN:
        print("  Note: no GITHUB_TOKEN found — rate limited to 10 req/min (may take several minutes).")
        print("        Set GITHUB_TOKEN for 30 req/min.")

    issues_all: List[dict] = []
    seen_numbers: set      = set()

    # Iterate month by month; honour START_DATE if set
    floor_dt = (
        datetime.strptime(START_DATE, "%Y-%m-%d")
        if START_DATE else
        first_dt.replace(tzinfo=None)
    )
    current = max(
        first_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None),
        floor_dt.replace(day=1),
    )
    last_naive = last_dt.replace(tzinfo=None)

    print(f"\nFetching issues month by month …")
    while current <= last_naive:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)

        start_str = current.strftime("%Y-%m-%d")
        end_str   = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")
        month_str = current.strftime("%Y-%m")

        page         = 1
        month_new    = 0
        while True:
            data  = _search_month(start_str, end_str, page)
            items = data.get("items", [])
            for item in items:
                if item["number"] not in seen_numbers:
                    seen_numbers.add(item["number"])
                    issues_all.append(item)
                    month_new += 1
            if len(items) < 100:
                break
            if page >= 10:
                # Safety: > 1000 issues in one month — should never happen here
                print(f"  WARNING: {month_str} has >1000 issues, some may be missing.")
                break
            page += 1

        if month_new:
            print(f"  {month_str}: {month_new:4d} issues  (running total: {len(issues_all)})")

        current = next_month

    print(f"\nDone — {len(issues_all)} issues fetched.\n")
    return issues_all


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "—"


def _parse_dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def analyze_labels(issues: List[dict]) -> Counter:
    c: Counter = Counter()
    for issue in issues:
        for label in issue.get("labels", []):
            c[label["name"]] += 1
    return c


def analyze_users(issues: List[dict]) -> Counter:
    c: Counter = Counter()
    for issue in issues:
        login = (issue.get("user") or {}).get("login")
        if login:
            c[login] += 1
    return c


def analyze_monthly(issues: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for issue in issues:
        if issue.get("created_at"):
            counts[_parse_dt(issue["created_at"]).strftime("%Y-%m")] += 1
    return dict(sorted(counts.items()))


def analyze_yearly(issues: List[dict]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for issue in issues:
        if issue.get("created_at"):
            counts[_parse_dt(issue["created_at"]).year] += 1
    return dict(sorted(counts.items()))


def analyze_states(issues: List[dict]) -> Counter:
    return Counter(i.get("state", "unknown") for i in issues)


def analyze_resolution_time(issues: List[dict]) -> Optional[dict]:
    days_list = []
    for issue in issues:
        if issue.get("state") == "closed" and issue.get("closed_at"):
            delta = (_parse_dt(issue["closed_at"]) - _parse_dt(issue["created_at"])).days
            days_list.append(delta)
    if not days_list:
        return None
    days_sorted = sorted(days_list)
    n = len(days_sorted)
    return {
        "count":       n,
        "avg_days":    sum(days_list) / n,
        "median_days": days_sorted[n // 2],
        "min_days":    days_sorted[0],
        "max_days":    days_sorted[-1],
        "within_1d":   sum(1 for d in days_list if d <= 1),
        "within_7d":   sum(1 for d in days_list if d <= 7),
        "within_30d":  sum(1 for d in days_list if d <= 30),
    }


def analyze_label_combos(issues: List[dict]) -> Counter:
    """Most common multi-label combinations."""
    c: Counter = Counter()
    for issue in issues:
        names = tuple(sorted(l["name"] for l in issue.get("labels", [])))
        if len(names) > 1:
            c[names] += 1
    return c


def analyze_weekday(issues: List[dict]) -> Dict[str, int]:
    days   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts: Dict[str, int] = defaultdict(int)
    for issue in issues:
        if issue.get("created_at"):
            counts[days[_parse_dt(issue["created_at"]).weekday()]] += 1
    return {d: counts[d] for d in days}


def analyze_hour(issues: List[dict]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for issue in issues:
        if issue.get("created_at"):
            counts[_parse_dt(issue["created_at"]).hour] += 1
    return dict(sorted(counts.items()))


def issues_with_most_labels(issues: List[dict], top_n: int = 5) -> List[dict]:
    return sorted(issues, key=lambda i: len(i.get("labels", [])), reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# CSV + Plot
# ---------------------------------------------------------------------------

def save_csv(monthly: Dict[str, int]) -> None:
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Month", "Issues"])
        for month, count in monthly.items():
            w.writerow([month, count])
    print(f"CSV  → {CSV_FILE}")


def create_monthly_plot(csv_path: str) -> None:
    months, counts = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            months.append(row["Month"])
            counts.append(int(row["Issues"]))

    fig_w = max(16, len(months) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    bar_color = "#1565C0"
    bars = ax.bar(range(len(months)), counts, color=bar_color,
                  edgecolor="white", linewidth=0.4)

    font_sz = max(5, min(9, int(120 / len(months))))
    for bar, v in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            str(v),
            ha="center", va="bottom",
            fontsize=font_sz, fontweight="bold", color="#1a1a1a",
        )

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=60, ha="right",
                       fontsize=max(6, min(9, int(130 / len(months)))))
    ax.set_xlabel("Month", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Issues", fontsize=12)
    ax.set_title(
        f"Issues Opened per Month — {REPO_OWNER}/{REPO_NAME}",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {PLOT_FILE}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown(
    issues:         List[dict],
    label_counts:   Counter,
    user_counts:    Counter,
    monthly:        Dict[str, int],
    yearly:         Dict[int, int],
    states:         Counter,
    resolution:     Optional[dict],
    weekday_counts: Dict[str, int],
    hour_counts:    Dict[int, int],
    combo_counts:   Counter,
) -> None:
    total   = len(issues)
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    repo_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"

    def trow(label, count, pct_of=None):
        denom = pct_of if pct_of is not None else total
        return f"| {label} | {count} | {_pct(count, denom)} |"

    lines: List[str] = []

    # ── Sub-header (the README h1 is already above the marker) ──────────────
    lines += [
        f"> **Generated:** {now}  ",
        f"> **Repository:** [{REPO_OWNER}/{REPO_NAME}]({repo_url})",
        "",
        "---",
        "",
    ]

    # ── Overview ────────────────────────────────────────────────────────────
    open_count   = states.get("open", 0)
    closed_count = states.get("closed", 0)
    all_dates    = [i["created_at"][:10] for i in issues if i.get("created_at")]
    first_date   = min(all_dates) if all_dates else "—"
    last_date    = max(all_dates) if all_dates else "—"

    lines += [
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Issues | **{total}** |",
        f"| Open Issues | {open_count} ({_pct(open_count, total)}) |",
        f"| Closed Issues | {closed_count} ({_pct(closed_count, total)}) |",
        f"| Unique Contributors | {len(user_counts)} |",
        f"| Distinct Labels | {len(label_counts)} |",
        f"| Total Label Assignments | {sum(label_counts.values())} |",
        f"| First Issue | {first_date} |",
        f"| Latest Issue | {last_date} |",
        "",
    ]

    # ── Labels ──────────────────────────────────────────────────────────────
    lines += [
        "## Label Statistics",
        "",
        "| Label | Count | % of All Issues |",
        "|-------|------:|----------------:|",
    ]
    for label, count in label_counts.most_common():
        lines.append(trow(f"`{label}`", count))

    no_label = sum(1 for i in issues if not i.get("labels"))
    lines += [
        "",
        f"> **{no_label}** issues ({_pct(no_label, total)}) carry no label.",
        "",
    ]

    # ── Issue States ────────────────────────────────────────────────────────
    lines += [
        "## Issue States",
        "",
        "| State | Count | Percentage |",
        "|-------|------:|-----------:|",
    ]
    for state, count in states.most_common():
        lines.append(trow(state.capitalize(), count))
    lines.append("")

    # ── Resolution Time ─────────────────────────────────────────────────────
    if resolution:
        r = resolution
        lines += [
            "## Issue Resolution Time",
            "",
            f"Based on **{r['count']}** closed issues.",
            "",
            "| Metric | Days |",
            "|--------|-----:|",
            f"| Average | {r['avg_days']:.1f} |",
            f"| Median | {r['median_days']} |",
            f"| Fastest | {r['min_days']} |",
            f"| Slowest | {r['max_days']} |",
            "",
            "| SLA Bucket | Count | % of Closed |",
            "|------------|------:|------------:|",
            f"| Closed within 1 day | {r['within_1d']} | {_pct(r['within_1d'], r['count'])} |",
            f"| Closed within 7 days | {r['within_7d']} | {_pct(r['within_7d'], r['count'])} |",
            f"| Closed within 30 days | {r['within_30d']} | {_pct(r['within_30d'], r['count'])} |",
            "",
        ]

    # ── Top Contributors ──────────────────────────────────────────────────
    lines += [
        "## Top Contributors (by Issues Opened)",
        "",
        "| Rank | User | Issues | % of Total |",
        "|-----:|------|-------:|-----------:|",
    ]
    for rank, (user, count) in enumerate(user_counts.most_common(15), 1):
        lines.append(f"| {rank} | [{user}](https://github.com/{user}) | {count} | {_pct(count, total)} |")
    lines.append("")

    # ── Yearly ──────────────────────────────────────────────────────────────
    lines += [
        "## Issues per Year",
        "",
        "| Year | Count | % of Total |",
        "|------|------:|-----------:|",
    ]
    for year, count in yearly.items():
        lines.append(trow(year, count))
    lines.append("")

    # ── Monthly ─────────────────────────────────────────────────────────────
    counts_list = list(monthly.values())
    avg_mo  = sum(counts_list) / len(counts_list) if counts_list else 0
    peak_mo = max(monthly, key=monthly.get) if monthly else "—"
    low_mo  = min(monthly, key=monthly.get) if monthly else "—"

    lines += [
        "## Issues per Month",
        "",
        "![Issues Opened per Month](issues_per_month.png)",
        "",
        "### Monthly Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Average Issues / Month | {avg_mo:.1f} |",
        f"| Peak Month | {peak_mo} — {monthly.get(peak_mo, 0)} issues |",
        f"| Quietest Month | {low_mo} — {monthly.get(low_mo, 0)} issues |",
        f"| Months with Activity | {len(monthly)} |",
        "",
        "### Full Monthly Breakdown",
        "",
        "| Month | Count |",
        "|-------|------:|",
    ]
    for month, count in monthly.items():
        lines.append(f"| {month} | {count} |")
    lines.append("")

    # ── Day-of-Week ─────────────────────────────────────────────────────────
    lines += [
        "## Issues by Day of Week (UTC)",
        "",
        "| Day | Count | % of Total |",
        "|-----|------:|-----------:|",
    ]
    for day, count in weekday_counts.items():
        lines.append(trow(day, count))
    busiest_day = max(weekday_counts, key=weekday_counts.get)
    lines += [
        "",
        f"> Most issues are opened on **{busiest_day}**.",
        "",
    ]

    # ── Hour of Day ─────────────────────────────────────────────────────────
    lines += [
        "## Issues by Hour of Day (UTC)",
        "",
        "| Hour (UTC) | Count |",
        "|:----------:|------:|",
    ]
    for hour, count in hour_counts.items():
        lines.append(f"| {hour:02d}:00 | {count} |")
    busiest_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 0
    lines += [
        "",
        f"> Peak activity at **{busiest_hour:02d}:00 UTC**.",
        "",
    ]

    # ── Label Combos ─────────────────────────────────────────────────────────
    if combo_counts:
        lines += [
            "## Most Common Label Combinations",
            "",
            "| Labels | Count |",
            "|--------|------:|",
        ]
        for combo, count in combo_counts.most_common(10):
            label_str = " + ".join(f"`{l}`" for l in combo)
            lines.append(f"| {label_str} | {count} |")
        lines.append("")

    # ── Issues with Most Labels ──────────────────────────────────────────────
    top_labeled = issues_with_most_labels(issues)
    if any(len(i.get("labels", [])) > 0 for i in top_labeled):
        lines += [
            "## Issues Carrying the Most Labels",
            "",
            "| Issue | Title | Labels |",
            "|-------|-------|-------:|",
        ]
        for i in top_labeled:
            n     = len(i.get("labels", []))
            url   = i.get("html_url", "")
            num   = i.get("number", "")
            title = (i.get("title") or "")[:70]
            if n:
                lines.append(f"| [#{num}]({url}) | {title} | {n} |")
        lines.append("")

    # ── Footer ──────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        f"*Statistics generated by `extract_statistics.py` on {now}.*",
    ]

    stats_block = "\n".join(lines) + "\n"

    # Read existing README and splice the stats block between the markers.
    # If the markers are absent they are appended to the file.
    try:
        with open(README_FILE, "r", encoding="utf-8") as f:
            readme = f.read()
    except FileNotFoundError:
        readme = ""

    if STATS_START in readme and STATS_END in readme:
        before = readme[: readme.index(STATS_START) + len(STATS_START)]
        after  = readme[readme.index(STATS_END):]
        new_readme = before + "\n" + stats_block + after
    else:
        sep = "\n" if readme.endswith("\n") else "\n\n"
        new_readme = readme + sep + STATS_START + "\n" + stats_block + STATS_END + "\n"

    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(new_readme)

    print(f"README → {README_FILE}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    issues:       List[dict],
    label_counts: Counter,
    user_counts:  Counter,
    states:       Counter,
) -> None:
    total = len(issues)
    sep   = "─" * 52

    print(sep)
    print(f"  Total issues : {total}")
    print(f"  Open         : {states.get('open', 0)}  ({_pct(states.get('open', 0), total)})")
    print(f"  Closed       : {states.get('closed', 0)}  ({_pct(states.get('closed', 0), total)})")
    print(sep)
    print("  Top labels:")
    for label, count in label_counts.most_common(6):
        print(f"    {label:<38} {count:>5}  ({_pct(count, total)})")
    print(sep)
    print("  Top 3 power users:")
    for rank, (user, count) in enumerate(user_counts.most_common(3), 1):
        print(f"    {rank}. {user:<32} {count:>5}  ({_pct(count, total)})")
    print(sep)
    print(f"\nOutput files:")
    print(f"  {CSV_FILE}")
    print(f"  {PLOT_FILE}")
    print(f"  {README_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 52)
    print("  GitHub Statistics Extractor")
    print(f"  {REPO_OWNER}/{REPO_NAME}")
    if GITHUB_TOKEN:
        print("  Auth: token present — 30 search req/min")
    else:
        print("  Auth: none — 10 search req/min (set GITHUB_TOKEN to raise)")
    print("=" * 52)
    print()

    issues = fetch_all_issues()
    if not issues:
        print("No issues found — nothing to report.")
        sys.exit(0)

    print("Analysing …")
    label_counts   = analyze_labels(issues)
    user_counts    = analyze_users(issues)
    monthly        = analyze_monthly(issues)
    yearly         = analyze_yearly(issues)
    states         = analyze_states(issues)
    resolution     = analyze_resolution_time(issues)
    weekday_counts = analyze_weekday(issues)
    hour_counts    = analyze_hour(issues)
    combo_counts   = analyze_label_combos(issues)

    save_csv(monthly)
    create_monthly_plot(CSV_FILE)
    generate_markdown(
        issues, label_counts, user_counts, monthly, yearly,
        states, resolution, weekday_counts, hour_counts, combo_counts,
    )

    print_summary(issues, label_counts, user_counts, states)


if __name__ == "__main__":
    main()
