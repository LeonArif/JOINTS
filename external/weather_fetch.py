"""Daily weather (precipitation sum, mean temperature) for every city of the competition from the Open-Meteo archive API (free, no key).
Cities are geocoded with the Open-Meteo geocoding API, preferring the result in the city's province (external/city_province.csv).
Output: external/weather_daily.csv (city_name, date, precip_mm, temp_c) for 2025-03-15 .. 2026-03-31, and external/city_coords.csv (the coordinates used)."""
import time, requests, pandas as pd


def get(url, params, tries=6):
    """GET with retries (the free API resets connections now and then)"""
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
            time.sleep(3 * (k + 1))
        except requests.exceptions.RequestException:
            time.sleep(3 * (k + 1))
    return None


cp = pd.read_csv("external/city_province.csv")
FIX = {"SOLO": "Surakarta", "LAMPUNG": "Bandar Lampung", "CIKARANG": "Cikarang", "ROKAN HILIR": "Bagansiapiapi", "KISARAN": "Kisaran", "TANJUNG PINANG": "Tanjung Pinang", "KUALA KAPUAS": "Kuala Kapuas", "PEMATANG SIANTAR": "Pematangsiantar", "RANTAU PRAPAT": "Rantauprapat"}
rows, fails = [], []
for city, prov in zip(cp["city_name"], cp["province"]):
    q = FIX.get(city, city.title())
    res = []
    for name in (q, q.split()[0]):
        resp = get("https://geocoding-api.open-meteo.com/v1/search", {"name": name, "count": 10, "country_code": "ID", "language": "en"})
        r = resp.json().get("results", []) if resp is not None else []
        if r:
            res = r; break
    if not res:
        fails.append(city); continue
    pick = next((x for x in res if prov.lower().replace("daerah istimewa ", "") in str(x.get("admin1", "")).lower() or str(x.get("admin1", "")).lower() in prov.lower()), res[0])
    rows.append({"city_name": city, "province": prov, "matched": pick["name"], "admin1": pick.get("admin1"), "lat": pick["latitude"], "lon": pick["longitude"]})
    time.sleep(0.2)
coords = pd.DataFrame(rows); coords.to_csv("external/city_coords.csv", index=False)
print("geocoded", len(coords), "failed", fails)
out = []
for _, c in coords.iterrows():
    for attempt in range(1):
        r = get("https://archive-api.open-meteo.com/v1/archive", {"latitude": c.lat, "longitude": c.lon, "start_date": "2025-03-15", "end_date": "2026-03-31",
                "daily": "precipitation_sum,temperature_2m_mean", "timezone": "Asia/Jakarta"})
        if r is not None:
            j = r.json()["daily"]; out.append(pd.DataFrame({"city_name": c.city_name, "date": j["time"], "precip_mm": j["precipitation_sum"], "temp_c": j["temperature_2m_mean"]})); break
    else:
        print("weather failed for", c.city_name)
    time.sleep(0.5)
w = pd.concat(out); w.to_csv("external/weather_daily.csv", index=False)
print(w.shape, w.city_name.nunique(), "cities |", w.date.min(), "..", w.date.max(), "| missing precip", int(w.precip_mm.isna().sum()))
