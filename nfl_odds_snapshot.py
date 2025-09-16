# nfl_odds_snapshot.py
# Fetches NFL odds from The Odds API v4, averages across bookmakers, and appends snapshots to Google Sheets.
# Usage: python nfl_odds_snapshot.py --snapshot-label openers
# Reads THE_ODDS_API_KEY from ~/.bashrc if --api-key isn't provided.

from __future__ import annotations
import argparse, datetime as dt, json, os, sys
from typing import Dict, List, Optional, Tuple, Sequence
import requests
from googleapiclient.errors import HttpError
from zoneinfo import ZoneInfo

# Google Sheets imports
from google.oauth2 import service_account
from googleapiclient.discovery import build

API_HOST = "https://api.the-odds-api.com"
SPORT_KEY = "americanfootball_nfl"

DEFAULT_BOOKMAKERS = ["draftkings","fanduel","betmgm","bovada","betonlineag","betrivers","mybookieag","lowvig"]
DEFAULT_SHEET_ID = "10U1k9QeVYcm2Mwfvj2ZbtU0_KkEq7FlqeUOZWCf5MVw"
DEFAULT_SHEET_NAME = "Odds"
DEFAULT_RAW_DIR = "raw_snapshots"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
# creds = Credentials.from_service_account_file("/Users/fbird/Desktop/Testing/CSP/cspscraping-4e20669fcaf7.json", scopes=scope)
# client = gspread.authorize(creds)
# sheet = client.open_by_key(sheet_id).worksheet(tab_name)

def _api_key_from_bashrc(var_name: str = "THE_ODDS_API_KEY") -> Optional[str]:
    """Retrieve API key from ~/.bashrc.

    Looks for lines like `export THE_ODDS_API_KEY=VALUE` or `THE_ODDS_API_KEY=VALUE`.
    Returns the value if found, otherwise None.
    """
    bashrc = os.path.expanduser("~/.bashrc")
    if not os.path.isfile(bashrc):
        return None
    with open(bashrc, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if line.startswith(f"{var_name}="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                return value
    return None

def _default_api_key() -> Optional[str]:
    return os.getenv("THE_ODDS_API_KEY") or _api_key_from_bashrc()

def _default_gcp_creds() -> Optional[str]:
    return os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or _api_key_from_bashrc("GOOGLE_API_KEY")

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
    # Log the outgoing API call to avoid silent usage
    print(f"Calling Odds API: {url}")
    print(f"Parameters: { {k:v for k,v in params.items() if k!='apiKey'} } (apiKey omitted)")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    # Try to parse remaining credits from response headers (best-effort)
    def _parse_credits_from_headers(hdrs: dict) -> Optional[str]:
        if not hdrs:
            return None
        # normalize to lowercase keys
        l = {k.lower(): v for k, v in hdrs.items()}
        candidates = [
            'x-requests-remaining', 'x-requests-available', 'x-ratelimit-remaining',
            'x-credits-remaining', 'remaining-credits', 'x-request-limit-remaining',
            'x-ratelimit-remaining-requests'
        ]
        for key in candidates:
            if key in l:
                return l[key]
        # Some providers include a JSON field in body; we won't parse that here.
        return None

    credits = _parse_credits_from_headers(dict(resp.headers))
    if credits is not None:
        print(f"Odds API call completed. Credits remaining: {credits}")
    else:
        print("Odds API call completed. Credits remaining: (unknown)")

    return resp, resp.json()

def extract_h2h_prices(event: Dict):
    away_prices, home_prices, per_book = [], [], {}
    for bk in event.get("bookmakers",[]):
        bk_key = bk.get("key")
        h2h = next((m for m in bk.get("markets",[]) if m.get("key")=="h2h" or m.get("key")=="moneyline"), None)
        if not h2h: continue
        away_odds, home_odds = None, None
        for out in h2h.get("outcomes",[]):
            if out.get("name")==event.get("away_team"): away_odds=out.get("price")
            elif out.get("name")==event.get("home_team"): home_odds=out.get("price")
        if away_odds is not None and home_odds is not None:
            per_book[bk_key]=(away_odds,home_odds)
            away_prices.append(away_odds); home_prices.append(home_odds)
    return away_prices, home_prices, per_book


def extract_spread_values(event: Dict):
    """Extract per-book spreads for home and away teams. Returns (home_spreads, away_spreads).

    Spread values follow the convention where negative point means that team is the favorite.
    We normalize so the returned value is the numeric point applied to that team (negative if that
    team is favored). Some feeds use 'point' or 'handicap' or 'line' as the numeric key.
    """
    home_spreads, away_spreads = [], []
    def _get_point(o):
        for fk in ("point","handicap","line", "price", "value"):
            if fk in o and o.get(fk) is not None:
                try:
                    return float(o.get(fk))
                except Exception:
                    try:
                        return float(str(o.get(fk)).replace("+",""))
                    except Exception:
                        return None
        return None

    for bk in event.get("bookmakers", []):
        bk_key = bk.get("key")
        sp = next((m for m in bk.get("markets",[]) if m.get("key") in ("spreads","spread","point_spread")), None)
        if not sp:
            # sometimes spreads are provided under a different key or nested; skip gracefully
            continue
        outcomes = sp.get("outcomes") or sp.get("outcome") or []
        home_point = None
        away_point = None
        for out in outcomes:
            name = out.get("name")
            pt = _get_point(out)
            if name == event.get("home_team"):
                home_point = pt
            elif name == event.get("away_team"):
                away_point = pt
        # If market reports points as positive for favorite vs negative, we keep values as-is
        if home_point is not None:
            home_spreads.append(home_point)
        if away_point is not None:
            away_spreads.append(away_point)
    return home_spreads, away_spreads

def average_implied_prob(prices: List[int]) -> Optional[float]:
    probs=[american_to_prob(p) for p in prices if p is not None]
    probs=[p for p in probs if p==p]
    if not probs: return None
    return sum(probs)/len(probs)

def simple_average(prices: List[int]) -> Optional[float]:
    return sum(prices)/len(prices) if prices else None

def ensure_dir(path:str): os.makedirs(path, exist_ok=True)

def save_raw_snapshot(raw_dir:str,label:str,timestamp_iso:str,payload:List[Dict]):
    ensure_dir(raw_dir)
    fname=f"odds_raw_{label}_{timestamp_iso.replace(':','').replace('-','').replace('Z','')}.json"
    fpath=os.path.join(raw_dir,fname)
    with open(fpath,"w",encoding="utf-8") as f: json.dump(payload,f,indent=2)
    return fpath

def get_sheets_service(creds_path: Optional[str] = None):
    """Return a Sheets API service. If creds_path is None, fall back to the
    GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_API_KEY environment variable or ~/.bashrc.
    """
    if not creds_path:
        creds_path = _default_gcp_creds()
    if not creds_path:
        raise FileNotFoundError("No Google credentials path provided")
    creds = service_account.Credentials.from_service_account_file(creds_path, scopes=scope)
    return build('sheets','v4',credentials=creds, cache_discovery=False)

def _a1_sheet_range(sheet_name: str, cell_range: str) -> str:
    """Return an A1-style range with the sheet name properly quoted when needed.

    If the sheet name contains spaces or special chars, wrap in single quotes and
    escape any single quotes by doubling them (per A1 notation rules).
    """
    if sheet_name is None:
        sheet_name = ""
    # escape single quotes by doubling
    safe = sheet_name.replace("'", "''")
    # wrap in quotes if it contains whitespace or any of []:*?/\\
    if any(c.isspace() for c in sheet_name) or any(ch in sheet_name for ch in '[]:*?/\\'):
        safe = f"'{safe}'"
    return f"{safe}!{cell_range}"

def ensure_sheet_header(service, spreadsheet_id: str, sheet_name: str, headers: List[str]):
    # Use an explicit A1 range for the header row to avoid parsing issues
    range_ = _a1_sheet_range(sheet_name, "A1:Z1")
    try:
        result = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_).execute()
    except HttpError as e:
        # If the sheet name doesn't exist or the range can't be parsed, try to
        # ensure the sheet exists and create it if necessary, then retry.
        msg = str(e)
        if 'Unable to parse range' in msg or (hasattr(e, 'resp') and getattr(e.resp, 'status', None) == 400):
            # Fetch existing sheet titles
            meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id, fields='sheets.properties').execute()
            sheets = [s['properties']['title'] for s in meta.get('sheets', []) if 'properties' in s]
            if sheet_name not in sheets:
                print(f"Sheet '{sheet_name}' not found in spreadsheet; creating it.")
                service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={
                    'requests': [{'addSheet': {'properties': {'title': sheet_name}}}]
                }).execute()
            else:
                # sheet exists but range parsing failed for another reason; re-raise
                raise
            # retry getting the range after creating sheet
            result = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_).execute()
        else:
            raise

    values = result.get('values', [])
    if not values or values[0] != headers:
        body = {'values': [headers]}
        service.spreadsheets().values().update(spreadsheetId=spreadsheet_id, range=range_, valueInputOption='RAW', body=body).execute()

def append_sheet_row(service, spreadsheet_id: str, sheet_name: str, row_values: Sequence[object]):
    range_ = _a1_sheet_range(sheet_name, "A:Z")
    body = {'values': [row_values]}
    service.spreadsheets().values().append(spreadsheetId=spreadsheet_id, range=range_, valueInputOption='USER_ENTERED', insertDataOption='INSERT_ROWS', body=body).execute()

def append_sheet_rows(service, spreadsheet_id: str, sheet_name: str, rows: List[List[object]], batch_size: int = 50, max_retries: int = 5):
    """Append rows in batches to reduce number of write requests and retry on rate limits.

    batch_size controls how many rows are sent in a single append request. Adjust down
    if you still hit per-request limits. Uses exponential backoff on HttpError 429.
    """
    import time, random
    if not rows:
        return
    # ensure rows are lists
    batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]
    for batch in batches:
        body = {'values': batch}
        attempt = 0
        while True:
            try:
                range_ = _a1_sheet_range(sheet_name, "A:Z")
                service.spreadsheets().values().append(spreadsheetId=spreadsheet_id, range=range_, valueInputOption='USER_ENTERED', insertDataOption='INSERT_ROWS', body=body).execute()
                break
            except HttpError as e:
                status = None
                try:
                    status = int(getattr(e.resp, 'status', 0))
                except Exception:
                    status = None
                # retry on 429 or 503
                if status in (429, 503) and attempt < max_retries:
                    sleep_time = (2 ** attempt) + random.random()
                    print(f"Rate-limited (status={status}), retrying in {sleep_time:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    attempt += 1
                    continue
                # otherwise re-raise
                raise

def format_commence_pacific(iso_str: Optional[str]) -> Optional[str]:
    """Convert an ISO8601 UTC timestamp to America/Los_Angeles and format as "MM/DD/YYYY hh:mm AM/PM TZ".

    If input is falsy, returns None. If parsing fails, returns the original string.
    """
    if not iso_str:
        return None
    try:
        s = iso_str
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dtobj = dt.datetime.fromisoformat(s)
        if dtobj.tzinfo is None:
            dtobj = dtobj.replace(tzinfo=dt.timezone.utc)
        pac = dtobj.astimezone(ZoneInfo("America/Los_Angeles"))
        # Example: 09/07/2025 01:00 PM PDT
        fmt = pac.strftime("%m/%d/%Y %I:%M %p")
        tz = pac.tzname() or "PT"
        return f"{fmt} {tz}"
    except Exception:
        return iso_str

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--api-key", default=_default_api_key())
    parser.add_argument("--sport", default=SPORT_KEY)
    parser.add_argument("--regions", default="us")
    # parser.add_argument("--regions", default="us,us2")
    parser.add_argument("--bookmakers", default=",".join(DEFAULT_BOOKMAKERS))
    parser.add_argument("--markets", default="h2h,spreads")
    parser.add_argument("--odds-format", default="american")
    # parser.add_argument("--markets", default="h2h,spreads,totals")
    parser.add_argument("--commence-from", default=None)
    parser.add_argument("--commence-to", default=None)
    parser.add_argument("--snapshot-label", default="snapshot")
    parser.add_argument("--sheet-id", default=os.getenv("SHEET_ID", DEFAULT_SHEET_ID))
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument("--gcp-creds", default=_default_gcp_creds())
    parser.add_argument("--raw-file", default=None, help="Path to a previously saved raw snapshot JSON to use instead of calling the Odds API")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    args=parser.parse_args()

    if not args.api_key:
        print("No API key provided"); sys.exit(2)
    bookmakers=[bk.strip() for bk in args.bookmakers.split(",") if bk.strip()]
    ts=now_iso()

    # If a cached raw file is provided, use it and do not call the Odds API.
    if args.raw_file:
        if not os.path.isfile(args.raw_file):
            print(f"Raw file {args.raw_file} not found"); sys.exit(2)
        print(f"Using cached raw snapshot: {args.raw_file}")
        with open(args.raw_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        resp = None
        raw_path = args.raw_file
    else:
        resp,data=fetch_odds(args.api_key,args.sport,args.regions,bookmakers,args.markets,args.odds_format,args.commence_from,args.commence_to)
        raw_path=save_raw_snapshot(args.raw_dir,args.snapshot_label,ts,data)
        print(f"Saved raw JSON to {raw_path}")

    # Updated column names: record moneyline averages + spread averages
    fieldnames = [
        "timestamp", "snapshot_label", "event_id", "commence_time", "home_team", "away_team", "bookmaker_count",
        "avg_home_moneyline", "avg_away_moneyline", "avg_home_implied_prob", "avg_away_implied_prob",
        "consensus_home_american_from_prob", "consensus_away_american_from_prob",
        "avg_home_spread", "avg_away_spread"
    ]

    # initialize Sheets service and ensure header (use env var GOOGLE_APPLICATION_CREDENTIALS if not passed)
    try:
        service = get_sheets_service(args.gcp_creds)
    except Exception:
        print("No Google credentials provided; set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_API_KEY (env var or in ~/.bashrc), or pass --gcp-creds pointing to a service account JSON file")
        sys.exit(2)

    ensure_sheet_header(service, args.sheet_id, args.sheet_name, fieldnames)

    rows_to_append: List[Sequence[object]] = []
    for event in data:
        away_prices, home_prices, per_book = extract_h2h_prices(event)
        # Get spreads per team (may be empty)
        home_spreads, away_spreads = extract_spread_values(event)

        # Skip events without moneyline info
        if not away_prices or not home_prices:
            continue

        avg_home_moneyline = simple_average(home_prices)
        avg_away_moneyline = simple_average(away_prices)
        avg_home_prob = average_implied_prob(home_prices)
        avg_away_prob = average_implied_prob(away_prices)
        avg_home_spread = simple_average(home_spreads) if home_spreads else None
        avg_away_spread = simple_average(away_spreads) if away_spreads else None

        row = {
            "timestamp": ts,
            "snapshot_label": args.snapshot_label,
            "event_id": event.get("id"),
            "commence_time": format_commence_pacific(event.get("commence_time")),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "bookmaker_count": len(per_book),
            "avg_home_moneyline": round(avg_home_moneyline, 2) if avg_home_moneyline is not None else None,
            "avg_away_moneyline": round(avg_away_moneyline, 2) if avg_away_moneyline is not None else None,
            "avg_home_implied_prob": round(avg_home_prob, 5) if avg_home_prob is not None else None,
            "avg_away_implied_prob": round(avg_away_prob, 5) if avg_away_prob is not None else None,
            "consensus_home_american_from_prob": prob_to_american(avg_home_prob) if avg_home_prob else None,
            "consensus_away_american_from_prob": prob_to_american(avg_away_prob) if avg_away_prob else None
        }
        # attach spread fields
        row["avg_home_spread"] = round(avg_home_spread, 2) if avg_home_spread is not None else None
        row["avg_away_spread"] = round(avg_away_spread, 2) if avg_away_spread is not None else None

        # prepare ordered values, convert None to empty string
        values = [row.get(k) if row.get(k) is not None else "" for k in fieldnames]
        rows_to_append.append(values)

    # append all rows in batches to avoid per-row write quota limits
    if rows_to_append:
        append_sheet_rows(service, args.sheet_id, args.sheet_name, [list(r) for r in rows_to_append], batch_size=50)
    print(f"Wrote {len(rows_to_append)} rows to Google Sheet {args.sheet_id}")


if __name__ == "__main__":
    main()
