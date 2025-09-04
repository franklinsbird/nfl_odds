# nfl_odds_snapshot.py
# Fetches NFL odds from The Odds API v4, averages across bookmakers, and appends snapshots to CSV.
# Usage: python nfl_odds_snapshot.py --api-key YOUR_KEY --snapshot-label openers

from __future__ import annotations
import argparse, csv, datetime as dt, json, os, sys
from typing import Dict, List, Optional, Tuple
import requests

API_HOST = "https://api.the-odds-api.com"
SPORT_KEY = "americanfootball_nfl"

DEFAULT_BOOKMAKERS = ["draftkings","fanduel","betmgm","bovada","betonlineag","betrivers","mybookieag","lowvig"]
DEFAULT_OUTPUT = "nfl_odds_snapshots.csv"
DEFAULT_RAW_DIR = "raw_snapshots"

def american_to_prob(odds: int) -> float:
    if odds is None: return float("nan")
    if odds >= 0: return 100.0/(odds+100.0)
    return -odds/(-odds+100.0)

def prob_to_american(p: float) -> Optional[int]:
    if p is None or p<=0 or p>=1: return None
    if p>0.5: return int(round(-p*100.0/(1.0-p)))
    else: return int(round((1.0-p)*100.0/p))

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0, tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z")

def fetch_odds(api_key: str, sport_key: str=SPORT_KEY, regions: str="us,us2",
               bookmakers: Optional[List[str]]=None, markets: str="h2h",
               odds_format: str="american", commence_from: Optional[str]=None,
               commence_to: Optional[str]=None) -> Tuple[requests.Response, List[Dict]]:
    params = {"apiKey": api_key, "regions": regions, "markets": markets,
              "oddsFormat": odds_format, "dateFormat": "iso"}
    if bookmakers: params["bookmakers"] = ",".join(bookmakers)
    if commence_from: params["commenceTimeFrom"] = commence_from
    if commence_to: params["commenceTimeTo"] = commence_to
    url = f"{API_HOST}/v4/sports/{sport_key}/odds"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp, resp.json()

def extract_h2h_prices(event: Dict):
    away_prices, home_prices, per_book = [], [], {}
    for bk in event.get("bookmakers",[]):
        bk_key = bk.get("key")
        h2h = next((m for m in bk.get("markets",[]) if m.get("key")=="h2h"), None)
        if not h2h: continue
        away_odds, home_odds = None, None
        for out in h2h.get("outcomes",[]):
            if out.get("name")==event.get("away_team"): away_odds=out.get("price")
            elif out.get("name")==event.get("home_team"): home_odds=out.get("price")
        if away_odds is not None and home_odds is not None:
            per_book[bk_key]=(away_odds,home_odds)
            away_prices.append(away_odds); home_prices.append(home_odds)
    return away_prices, home_prices, per_book

def average_implied_prob(prices: List[int]) -> Optional[float]:
    probs=[american_to_prob(p) for p in prices if p is not None]
    probs=[p for p in probs if p==p]
    if not probs: return None
    return sum(probs)/len(probs)

def simple_average(prices: List[int]) -> Optional[float]:
    return sum(prices)/len(prices) if prices else None

def ensure_dir(path:str): os.makedirs(path, exist_ok=True)

def append_csv_row(path:str, row:Dict[str,object], fieldnames:List[str]):
    file_exists=os.path.isfile(path)
    with open(path,"a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        writer.writerow(row)

def save_raw_snapshot(raw_dir:str,label:str,timestamp_iso:str,payload:List[Dict]):
    ensure_dir(raw_dir)
    fname=f"odds_raw_{label}_{timestamp_iso.replace(':','').replace('-','').replace('Z','')}.json"
    fpath=os.path.join(raw_dir,fname)
    with open(fpath,"w",encoding="utf-8") as f: json.dump(payload,f,indent=2)
    return fpath

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("THE_ODDS_API_KEY"))
    parser.add_argument("--sport", default=SPORT_KEY)
    parser.add_argument("--regions", default="us,us2")
    parser.add_argument("--bookmakers", default=",".join(DEFAULT_BOOKMAKERS))
    parser.add_argument("--markets", default="h2h")
    parser.add_argument("--odds-format", default="american")
    parser.add_argument("--commence-from", default=None)
    parser.add_argument("--commence-to", default=None)
    parser.add_argument("--snapshot-label", default="snapshot")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    args=parser.parse_args()

    if not args.api_key:
        print("No API key provided"); sys.exit(2)
    bookmakers=[bk.strip() for bk in args.bookmakers.split(",") if bk.strip()]
    ts=now_iso()
    resp,data=fetch_odds(args.api_key,args.sport,args.regions,bookmakers,args.markets,args.odds_format,args.commence_from,args.commence_to)
    raw_path=save_raw_snapshot(args.raw_dir,args.snapshot_label,ts,data)
    print(f"Saved raw JSON to {raw_path}")

    fieldnames=["timestamp","snapshot_label","event_id","commence_time","home_team","away_team","bookmaker_count","avg_home_american","avg_away_american","avg_home_implied_prob","avg_away_implied_prob","consensus_home_american_from_prob","consensus_away_american_from_prob"]
    rows_written=0
    for event in data:
        away_prices,home_prices,per_book=extract_h2h_prices(event)
        if not away_prices or not home_prices: continue
        avg_home_odds=simple_average(home_prices)
        avg_away_odds=simple_average(away_prices)
        avg_home_prob=average_implied_prob(home_prices)
        avg_away_prob=average_implied_prob(away_prices)
        row={
            "timestamp":ts,"snapshot_label":args.snapshot_label,"event_id":event.get("id"),
            "commence_time":event.get("commence_time"),"home_team":event.get("home_team"),
            "away_team":event.get("away_team"),"bookmaker_count":len(per_book),
            "avg_home_american":round(avg_home_odds,2) if avg_home_odds is not None else None,
            "avg_away_american":round(avg_away_odds,2) if avg_away_odds is not None else None,
            "avg_home_implied_prob":round(avg_home_prob,5) if avg_home_prob is not None else None,
            "avg_away_implied_prob":round(avg_away_prob,5) if avg_away_prob is not None else None,
            "consensus_home_american_from_prob":prob_to_american(avg_home_prob) if avg_home_prob else None,
            "consensus_away_american_from_prob":prob_to_american(avg_away_prob) if avg_away_prob else None}
        append_csv_row(args.out,row,fieldnames); rows_written+=1
    print(f"Wrote {rows_written} rows to {args.out}")

if __name__=="__main__": main()
