"""Scrape per-film genre, runtime, year and Cinepoint Flash score from the day popups of https://cinepoint.com/pages/tbo.

The daily scraper (cinepoint_scrape.py) kept only the first line of the title cell and ignored the Score column; the same popup rows also hold
genre, runtime, release year and the Cinepoint Flash score of each film. The score is as of the day this script runs (post-cutoff data, main.ipynb only).

Drives a real (visible) Chrome window: keep it on screen while it runs.

Usage:
    python external/cinepoint/cinepoint_film_info.py --start 2025-04-01 --end 2026-03-31 --step 3
    python external/cinepoint/cinepoint_film_info.py --dates 2025-06-02,2025-06-03          (explicit dates, for filling gaps)
Output: external/cinepoint/cinepoint_film_info.csv (appended, resumable; one row per film and scraped date)
"""

import argparse
import calendar
import csv
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from playwright.sync_api import sync_playwright

import cinepoint_scrape as cs

OUT_DEFAULT = Path(__file__).with_name("cinepoint_film_info.csv")
FIELDS = ["date", "rank", "title", "genre", "runtime_min", "year", "score", "daily_admission", "total_admission", "showtimes"]


def parse_title_cell(text):
    """first line = title, the rest 'Genre / Genre / 1h 43m / 2025' -> title, genre, runtime in minutes, year"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title, rest = lines[0], " / ".join(lines[1:])
    year = runtime = None
    m = re.search(r"(\d{4})\s*$", rest)
    if m:
        year, rest = int(m.group(1)), rest[:m.start()].rstrip(" /")
    m = re.search(r"(?:(\d+)h\s*(\d+)m|(\d+)h|(\d+)m)\s*$", rest)
    if m:
        runtime = int(m.group(1) or m.group(3) or 0) * 60 + int(m.group(2) or m.group(4) or 0)
        rest = rest[:m.start()].rstrip(" /")
    return title, rest, runtime, year


def read_info_rows(mask, iso):
    rows = []
    for i in range(mask.locator("tbody tr").count()):
        tds = mask.locator("tbody tr").nth(i).locator("td")
        if tds.count() < 7:
            continue
        title, genre, runtime, year = parse_title_cell(tds.nth(1).inner_text())
        score = re.findall(r"\d+(?:\.\d+)?", tds.nth(6).inner_text().replace("Cinepoint", "").replace("Flash", ""))
        rows.append({"date": iso, "rank": cs.to_int(tds.nth(0).inner_text()), "title": title, "genre": genre, "runtime_min": runtime, "year": year,
                     "score": float(score[-1]) if score else None, "daily_admission": cs.to_int(tds.nth(2).inner_text()),
                     "total_admission": cs.to_int(tds.nth(4).inner_text()), "showtimes": cs.to_int(tds.nth(5).inner_text())})
    return rows


cs.read_page_rows = read_info_rows   # read_popup_rows (pagination, 'rows per page') reuses the daily scraper's logic with our parser


def load_done(path):
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {r["date"] for r in csv.DictReader(f)}


def run(dates, out=OUT_DEFAULT):
    out = Path(out)
    done = load_done(out)
    todo = sorted(d for d in dates if d not in done)
    print(f"{len(dates)} dates, {len(todo)} to scrape", flush=True)
    by_month = {}
    for d in todo:
        by_month.setdefault(d[:7], []).append(d)
    new_file = not out.exists()
    with sync_playwright() as p, out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        browser = p.chromium.launch(channel="chrome", headless=False)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(cs.URL, wait_until="domcontentloaded")
        page.wait_for_selector("tbody a", timeout=30000)
        page.wait_for_timeout(1500)
        for ym, days in by_month.items():
            y, m = int(ym[:4]), int(ym[5:])
            ok = cs.select_period(page, y, m)
            if not ok:
                page.goto(cs.URL, wait_until="domcontentloaded")
                page.wait_for_selector("tbody a", timeout=30000)
                page.wait_for_timeout(1500)
                ok = cs.select_period(page, y, m)
            if not ok:
                print(f"{ym}: could not select period, skipped", file=sys.stderr, flush=True)
                continue
            days_in_month = calendar.monthrange(y, m)[1]
            for attempt in range(4):
                for _ in range(40):
                    if page.locator("tbody a").count() >= days_in_month:
                        break
                    page.wait_for_timeout(500)
                if page.locator("tbody a").count() >= days_in_month:
                    break
                cs.pick(page, r"\d{2,3}", "100", settle=2500)
            for iso in days:
                label = datetime.strptime(iso, "%Y-%m-%d").strftime("%b %-d, %Y") if sys.platform != "win32" else datetime.strptime(iso, "%Y-%m-%d").strftime("%b %#d, %Y")
                opened = False
                for _ in range(3):
                    cs.close_popup(page)
                    link = page.locator("tbody a", has_text=label)
                    if not link.count():
                        break
                    link.first.click()
                    try:
                        page.wait_for_selector("div.p-dialog-mask tbody tr", timeout=8000)
                        opened = True
                        break
                    except Exception:
                        page.wait_for_timeout(1000)
                if not opened:
                    print(f"{iso}: popup did not open", file=sys.stderr, flush=True)
                    continue
                got_iso, rows, total = cs.read_popup_rows(page)
                for r in rows:
                    r["date"] = got_iso
                writer.writerows(rows)
                f.flush()
                print(f"{got_iso}: {len(rows)} films (popup says {total})", flush=True)
                cs.close_popup(page)
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--dates", default="")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args()
    if a.dates:
        dates = [d.strip() for d in a.dates.split(",") if d.strip()]
    else:
        d0, d1 = datetime.strptime(a.start, "%Y-%m-%d"), datetime.strptime(a.end, "%Y-%m-%d")
        dates = [(d0 + timedelta(days=k)).strftime("%Y-%m-%d") for k in range(0, (d1 - d0).days + 1, a.step)]
    run(dates, a.out)
