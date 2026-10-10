"""Wikipedia daily pageviews (Wikimedia REST API, free) of every film's English and Indonesian article, from D1 - 35 days to D1 + 11 days.
Articles come from external/film_profile.csv (wikipedia_en / wikipedia_id); D1 = first date of the film in data/train.csv or data/test_history.csv.
Output: external/wiki_pageviews_window.csv (film, lang, article, date, views). Post-cutoff data: main.ipynb only."""
import time, urllib.parse, requests, pandas as pd
HEAD = {"User-Agent": "joints-2026-research/1.0 (student competition; contact via kaggle)"}
tr = pd.read_csv("data/train.csv", usecols=["movie_title", "date_show"], parse_dates=["date_show"])
th = pd.read_csv("data/test_history.csv", usecols=["movie_title", "date_show"], parse_dates=["date_show"])
d1 = pd.concat([tr, th]).groupby("movie_title")["date_show"].min()
prof = pd.read_csv("external/film_profile.csv").drop_duplicates("movie_title").set_index("movie_title")


def get(url, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, headers=HEAD, timeout=60)
            if r.status_code in (200, 404):
                return r
            time.sleep(3 * (k + 1))
        except requests.exceptions.RequestException:
            time.sleep(3 * (k + 1))
    return None


rows, missing = [], 0
for i, (film, day1) in enumerate(d1.items()):
    if film not in prof.index:
        continue
    a, b = (day1 - pd.Timedelta(days=35)).strftime("%Y%m%d"), (day1 + pd.Timedelta(days=11)).strftime("%Y%m%d")
    for lang, col in (("en", "wikipedia_en"), ("id", "wikipedia_id")):
        art = prof.loc[film, col]
        if not isinstance(art, str) or not art.strip():
            continue
        title = urllib.parse.quote(art.strip().replace(" ", "_"), safe="")
        r = get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{title}/daily/{a}/{b}")
        if r is None or r.status_code != 200:
            missing += 1; continue
        for it in r.json().get("items", []):
            rows.append((film, lang, art, pd.to_datetime(it["timestamp"][:8]).strftime("%Y-%m-%d"), it["views"]))
        time.sleep(0.15)
    if i % 50 == 0:
        print(i, len(d1), len(rows), flush=True)
out = pd.DataFrame(rows, columns=["movie_title", "lang", "article", "date", "views"]); out.to_csv("external/wiki_pageviews_window.csv", index=False)
print("saved", out.shape, "films", out.movie_title.nunique(), "| article-language pairs without data:", missing)
