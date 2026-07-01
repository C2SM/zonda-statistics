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

OUTPUT_DIR  = "."
CSV_DIR     = os.path.join(OUTPUT_DIR, "csv")
README_FILE = os.path.join(OUTPUT_DIR, "README.md")

# Every table in the generated report is backed by its own CSV file. The
# extractor writes raw data to these files first; generate_markdown() reads
# them back in to build the report, so the CSVs are always the source of
# truth behind README.md.
CSV_OVERVIEW      = os.path.join(CSV_DIR, "overview.csv")
CSV_LABELS        = os.path.join(CSV_DIR, "labels.csv")
CSV_STATES        = os.path.join(CSV_DIR, "states.csv")
CSV_RESOLUTION    = os.path.join(CSV_DIR, "resolution.csv")
CSV_CONTRIBUTORS  = os.path.join(CSV_DIR, "contributors.csv")
CSV_YEARLY        = os.path.join(CSV_DIR, "yearly.csv")
CSV_MONTHLY       = os.path.join(CSV_DIR, "monthly.csv")
CSV_WEEKDAY       = os.path.join(CSV_DIR, "weekday.csv")
CSV_HOURLY        = os.path.join(CSV_DIR, "hourly.csv")
CSV_LABEL_COMBOS  = os.path.join(CSV_DIR, "label_combinations.csv")
CSV_TOP_LABELED   = os.path.join(CSV_DIR, "most_labeled_issues.csv")

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
# CSV persistence
#
# Every table that ends up in README.md is first written to its own CSV file
# under csv/. generate_markdown() reads these files back in, so the CSVs are
# always the authoritative data behind the report.
# ---------------------------------------------------------------------------

def _write_csv(path: str, header: List[str], rows: List[list]) -> None:
    os.makedirs(CSV_DIR, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _read_csv(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_all_csvs(
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
    top_labeled:    List[dict],
    generated_at:   str,
) -> None:
    total        = len(issues)
    open_count   = states.get("open", 0)
    closed_count = states.get("closed", 0)
    all_dates    = [i["created_at"][:10] for i in issues if i.get("created_at")]
    first_date   = min(all_dates) if all_dates else ""
    last_date    = max(all_dates) if all_dates else ""
    no_label     = sum(1 for i in issues if not i.get("labels"))

    _write_csv(CSV_OVERVIEW, ["Metric", "Value"], [
        ["Generated at", generated_at],
        ["Total issues", total],
        ["Open issues", open_count],
        ["Closed issues", closed_count],
        ["Unique contributors", len(user_counts)],
        ["Distinct labels", len(label_counts)],
        ["Total label assignments", sum(label_counts.values())],
        ["First issue date", first_date],
        ["Latest issue date", last_date],
        ["Unlabeled issues", no_label],
    ])

    _write_csv(CSV_LABELS, ["Label", "Count"],
               [[label, count] for label, count in label_counts.most_common()])

    _write_csv(CSV_STATES, ["State", "Count"],
               [[state.capitalize(), count] for state, count in states.most_common()])

    if resolution:
        r = resolution
        _write_csv(CSV_RESOLUTION, ["Metric", "Value"], [
            ["Count", r["count"]],
            ["Average days", f"{r['avg_days']:.4f}"],
            ["Median days", r["median_days"]],
            ["Min days", r["min_days"]],
            ["Max days", r["max_days"]],
            ["Within 1 day", r["within_1d"]],
            ["Within 7 days", r["within_7d"]],
            ["Within 30 days", r["within_30d"]],
        ])
    elif os.path.exists(CSV_RESOLUTION):
        os.remove(CSV_RESOLUTION)

    _write_csv(CSV_CONTRIBUTORS, ["User", "Issues"],
               [[user, count] for user, count in user_counts.most_common()])

    _write_csv(CSV_YEARLY, ["Year", "Count"],
               [[year, count] for year, count in yearly.items()])

    _write_csv(CSV_MONTHLY, ["Month", "Count"],
               [[month, count] for month, count in monthly.items()])

    _write_csv(CSV_WEEKDAY, ["Day", "Count"],
               [[day, count] for day, count in weekday_counts.items()])

    _write_csv(CSV_HOURLY, ["Hour", "Count"],
               [[f"{hour:02d}", count] for hour, count in hour_counts.items()])

    _write_csv(CSV_LABEL_COMBOS, ["Labels", "Count"],
               [[" + ".join(combo), count] for combo, count in combo_counts.most_common()])

    _write_csv(CSV_TOP_LABELED, ["Issue", "Title", "Labels", "URL"], [
        [i.get("number", ""), (i.get("title") or "")[:70],
         len(i.get("labels", [])), i.get("html_url", "")]
        for i in top_labeled if i.get("labels")
    ])

    print(f"CSV  → {CSV_DIR}/")


# ---------------------------------------------------------------------------
# Markdown report
#
# Everything below is read back from csv/ rather than passed in as Python
# objects, so the CSVs are the sole source of truth for the report.
# ---------------------------------------------------------------------------

def generate_markdown() -> None:
    overview = {row["Metric"]: row["Value"] for row in _read_csv(CSV_OVERVIEW)}
    total    = int(overview["Total issues"])
    repo_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"

    def trow(label, count, pct_of=None):
        denom = pct_of if pct_of is not None else total
        return f"| {label} | {count} | {_pct(int(count), int(denom))} |"

    lines: List[str] = []

    # ── Sub-header (the README h1 is already above the marker) ──────────────
    lines += [
        f"> **Generated:** {overview['Generated at']}  ",
        f"> **Repository:** [{REPO_OWNER}/{REPO_NAME}]({repo_url})",
        "",
        "---",
        "",
    ]

    # ── Overview ────────────────────────────────────────────────────────────
    lines += [
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Issues | **{total}** |",
        f"| Open Issues | {overview['Open issues']} ({_pct(int(overview['Open issues']), total)}) |",
        f"| Closed Issues | {overview['Closed issues']} ({_pct(int(overview['Closed issues']), total)}) |",
        f"| Unique Contributors | {overview['Unique contributors']} |",
        f"| Distinct Labels | {overview['Distinct labels']} |",
        f"| Total Label Assignments | {overview['Total label assignments']} |",
        f"| First Issue | {overview['First issue date']} |",
        f"| Latest Issue | {overview['Latest issue date']} |",
        "",
    ]

    # ── Labels ──────────────────────────────────────────────────────────────
    lines += [
        "## Label Statistics",
        "",
        "| Label | Count | % of All Issues |",
        "|-------|------:|----------------:|",
    ]
    for row in _read_csv(CSV_LABELS):
        lines.append(trow(f"`{row['Label']}`", row["Count"]))

    no_label = int(overview["Unlabeled issues"])
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
    for row in _read_csv(CSV_STATES):
        lines.append(trow(row["State"], row["Count"]))
    lines.append("")

    # ── Resolution Time ─────────────────────────────────────────────────────
    if os.path.exists(CSV_RESOLUTION):
        r = {row["Metric"]: row["Value"] for row in _read_csv(CSV_RESOLUTION)}
        lines += [
            "## Issue Resolution Time",
            "",
            f"Based on **{r['Count']}** closed issues.",
            "",
            "| Metric | Days |",
            "|--------|-----:|",
            f"| Average | {float(r['Average days']):.1f} |",
            f"| Median | {r['Median days']} |",
            f"| Fastest | {r['Min days']} |",
            f"| Slowest | {r['Max days']} |",
            "",
            "| SLA Bucket | Count | % of Closed |",
            "|------------|------:|------------:|",
            f"| Closed within 1 day | {r['Within 1 day']} | {_pct(int(r['Within 1 day']), int(r['Count']))} |",
            f"| Closed within 7 days | {r['Within 7 days']} | {_pct(int(r['Within 7 days']), int(r['Count']))} |",
            f"| Closed within 30 days | {r['Within 30 days']} | {_pct(int(r['Within 30 days']), int(r['Count']))} |",
            "",
        ]

    # ── Top Contributors ──────────────────────────────────────────────────
    lines += [
        "## Top Contributors (by Issues Opened)",
        "",
        "| Rank | User | Issues | % of Total |",
        "|-----:|------|-------:|-----------:|",
    ]
    for rank, row in enumerate(_read_csv(CSV_CONTRIBUTORS)[:15], 1):
        user, count = row["User"], row["Issues"]
        lines.append(f"| {rank} | [{user}](https://github.com/{user}) | {count} | {_pct(int(count), total)} |")
    lines.append("")

    # ── Yearly ──────────────────────────────────────────────────────────────
    lines += [
        "## Issues per Year",
        "",
        "| Year | Count | % of Total |",
        "|------|------:|-----------:|",
    ]
    for row in _read_csv(CSV_YEARLY):
        lines.append(trow(row["Year"], row["Count"]))
    lines.append("")

    # ── Monthly ─────────────────────────────────────────────────────────────
    monthly_rows = _read_csv(CSV_MONTHLY)
    counts_list  = [int(row["Count"]) for row in monthly_rows]
    avg_mo       = sum(counts_list) / len(counts_list) if counts_list else 0
    peak_row     = max(monthly_rows, key=lambda row: int(row["Count"])) if monthly_rows else None
    low_row      = min(monthly_rows, key=lambda row: int(row["Count"])) if monthly_rows else None

    lines += [
        "## Issues per Month",
        "",
        "### Monthly Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Average Issues / Month | {avg_mo:.1f} |",
        f"| Peak Month | {peak_row['Month'] if peak_row else '—'} — {peak_row['Count'] if peak_row else 0} issues |",
        f"| Quietest Month | {low_row['Month'] if low_row else '—'} — {low_row['Count'] if low_row else 0} issues |",
        f"| Months with Activity | {len(monthly_rows)} |",
        "",
        "### Full Monthly Breakdown",
        "",
        "| Month | Count |",
        "|-------|------:|",
    ]
    for row in monthly_rows:
        lines.append(f"| {row['Month']} | {row['Count']} |")
    lines.append("")

    # ── Day-of-Week ─────────────────────────────────────────────────────────
    weekday_rows = _read_csv(CSV_WEEKDAY)
    lines += [
        "## Issues by Day of Week (UTC)",
        "",
        "| Day | Count | % of Total |",
        "|-----|------:|-----------:|",
    ]
    for row in weekday_rows:
        lines.append(trow(row["Day"], row["Count"]))
    busiest_day = max(weekday_rows, key=lambda row: int(row["Count"]))["Day"]
    lines += [
        "",
        f"> Most issues are opened on **{busiest_day}**.",
        "",
    ]

    # ── Hour of Day ─────────────────────────────────────────────────────────
    hourly_rows = _read_csv(CSV_HOURLY)
    lines += [
        "## Issues by Hour of Day (UTC)",
        "",
        "| Hour (UTC) | Count |",
        "|:----------:|------:|",
    ]
    for row in hourly_rows:
        lines.append(f"| {row['Hour']}:00 | {row['Count']} |")
    busiest_hour = max(hourly_rows, key=lambda row: int(row["Count"]))["Hour"] if hourly_rows else "00"
    lines += [
        "",
        f"> Peak activity at **{busiest_hour}:00 UTC**.",
        "",
    ]

    # ── Label Combos ─────────────────────────────────────────────────────────
    combo_rows = _read_csv(CSV_LABEL_COMBOS)
    if combo_rows:
        lines += [
            "## Most Common Label Combinations",
            "",
            "| Labels | Count |",
            "|--------|------:|",
        ]
        for row in combo_rows[:10]:
            label_str = " + ".join(f"`{l}`" for l in row["Labels"].split(" + "))
            lines.append(f"| {label_str} | {row['Count']} |")
        lines.append("")

    # ── Issues with Most Labels ──────────────────────────────────────────────
    top_labeled_rows = _read_csv(CSV_TOP_LABELED)
    if top_labeled_rows:
        lines += [
            "## Issues Carrying the Most Labels",
            "",
            "| Issue | Title | Labels |",
            "|-------|-------|-------:|",
        ]
        for row in top_labeled_rows:
            lines.append(f"| [#{row['Issue']}]({row['URL']}) | {row['Title']} | {row['Labels']} |")
        lines.append("")

    # ── Footer ──────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        f"*Statistics generated by `extract_statistics.py` on {overview['Generated at']}.*",
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
    print(f"  {CSV_DIR}/")
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
    top_labeled    = issues_with_most_labels(issues)
    generated_at   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    save_all_csvs(
        issues, label_counts, user_counts, monthly, yearly, states,
        resolution, weekday_counts, hour_counts, combo_counts, top_labeled,
        generated_at,
    )
    generate_markdown()

    print_summary(issues, label_counts, user_counts, states)


if __name__ == "__main__":
    main()
