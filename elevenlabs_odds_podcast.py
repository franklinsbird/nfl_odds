import os
import sys
import math
import statistics
import textwrap
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional
import base64

import requests
import json

# Reuse the helpers from nfl_odds_snapshot to get the Odds API key and fetching logic
from nfl_odds_snapshot import _default_api_key, fetch_odds

# ----- Config and defaults -----
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
VOICE_SEARCH_DEFAULT = os.getenv("VOICE_SEARCH", "american sports announcer")
DAYS_AHEAD_DEFAULT = int(os.getenv("DAYS_AHEAD", "8"))

def iso_range_for_next_days(days: int):
    now = datetime.now(timezone.utc)
    start = now.replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(days=days)
    print(f"[debug] iso_range_for_next_days: start={start.isoformat()} end={end.isoformat()}")
    return start.isoformat(), end.isoformat()

# Note: fetch_odds from nfl_odds_snapshot accepts commence_from/commence_to args; we'll
# compute and pass them so we get the same window behavior as the original script.
def fetch_nfl_odds(days=8):
    start_iso, end_iso = iso_range_for_next_days(days)
    api_key = _default_api_key() or ""
    print(f"[debug] fetch_nfl_odds: start={start_iso} end={end_iso} api_key_set={bool(api_key)}")
    resp, events = fetch_odds(api_key, commence_from=start_iso, commence_to=end_iso)
    print(f"[debug] fetch_nfl_odds: resp_type={type(resp).__name__} events_type={type(events).__name__} events_len={len(events) if hasattr(events, '__len__') else 'n/a'}")
    return events, getattr(resp, 'headers', {}) if resp is not None else {}

def median_or_none(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None

def round_half(x, ndigits=1):
    # avoid ugly 6.499999 prints from floats
    return float(f"{round(x, ndigits):g}")

def summarize_events(events):
    """
    Build a consensus favorite (by bookmaker majority) and median spread/total.
    Events schema has: home_team, away_team, commence_time, bookmakers[] with markets[spreads, totals].
    """
    print(f"[debug] summarize_events: received {len(events) if hasattr(events, '__len__') else 'n/a'} events")
    summaries = []
    for idx, ev in enumerate(events):
        home, away = ev.get("home_team"), ev.get("away_team")
        start_iso = ev.get("commence_time")
        books = ev.get("bookmakers", [])
        # print(f"[debug] event {idx}: home={home} away={away} kickoff={start_iso} bookmakers={len(books)}")

        # Aggregate favorites across books
        fav_counts = {}
        fav_spreads = {}  # team -> list of absolute spread points
        totals_points = []

        for bk in books:
            # Build a map of markets supporting multiple key names
            mkts = {}
            for m in bk.get("markets", []):
                k = m.get("key") or m.get("market_key") or m.get("market")
                if k:
                    mkts[k] = m
            # if mkts:
            #     print(f"[debug]  book {bk.get('key','?')} markets_found={list(mkts.keys())}")

            # helper to parse a numeric point from various possible fields
            def _parse_point(o):
                for fk in ("point", "handicap", "line", "price", "value"):
                    if fk in o and o.get(fk) is not None:
                        try:
                            return float(o.get(fk))
                        except Exception:
                            # sometimes the point is nested or non-numeric
                            try:
                                return float(str(o.get(fk)).replace("+", ""))
                            except Exception:
                                return None
                return None

            # Try several possible market keys for spreads
            spread_keys = [k for k in ("spreads", "spread", "point_spread") if k in mkts]
            for sk in spread_keys:
                outcomes = mkts[sk].get("outcomes") or mkts[sk].get("outcome") or []
                if not outcomes:
                    continue
                # Find favorite (negative point) for this book
                book_fav = None
                book_fav_point = None
                for o in outcomes:
                    pt = _parse_point(o)
                    name = o.get("name")
                    if pt is None or name is None:
                        continue
                    if pt < 0:
                        book_fav = name
                        book_fav_point = abs(pt)
                        break
                if book_fav:
                    fav_counts[book_fav] = fav_counts.get(book_fav, 0) + 1
                    fav_spreads.setdefault(book_fav, []).append(book_fav_point)

            # totals: try common keys and outcome name variations
            total_keys = [k for k in ("totals", "total", "over_under", "ou") if k in mkts]
            for tk in total_keys:
                tout = mkts[tk].get("outcomes") or mkts[tk].get("outcome") or []
                # Prefer the "Over" line to record a single number
                over = next((o for o in tout if o.get("name", "").lower().startswith("over")), None)
                if over:
                    pt = _parse_point(over)
                    if pt is not None:
                        totals_points.append(pt)
            # If we found no totals via outcomes, some feeds put the line on the market itself
            if not totals_points:
                for tk in total_keys:
                    m = mkts.get(tk)
                    if m and m.get("point") is not None:
                        try:
                            totals_points.append(float(m.get("point")))
                        except Exception:
                            pass

            # If no favs found in this book, optionally log first outcomes for diagnosis
            if not fav_counts:
                # print just one sample for debugging if markets exist
                if mkts:
                    sample_k = next(iter(mkts))
                    sample_out = (mkts[sample_k].get("outcomes") or mkts[sample_k].get("outcome") or [])
                    if sample_out:
                        print(f"[debug]  sample outcomes for book {bk.get('key','?')} market {sample_k}: {[{ 'name': o.get('name'), 'point': o.get('point'), 'price': o.get('price') } for o in sample_out[:3]]}")

            # Collect moneyline info (price) to determine favorite when spreads missing
            ml_keys = [k for k in ("moneyline", "ml", "h2h", "head_to_head") if k in mkts]
            for mk in ml_keys:
                outs = mkts[mk].get("outcomes") or mkts[mk].get("outcome") or []
                if len(outs) < 2:
                    continue
                # parse price for each outcome
                parsed = []
                for o in outs:
                    name = o.get("name")
                    # price often under 'price' or 'prob' or 'price_usd' etc; reuse _parse_point
                    price = None
                    if o.get("price") is not None:
                        try:
                            price = float(o.get("price"))
                        except Exception:
                            try:
                                price = float(str(o.get("price")).replace("+", ""))
                            except Exception:
                                price = None
                    else:
                        price = _parse_point(o)
                    if price is None or name is None:
                        continue
                    parsed.append((name, price))
                if len(parsed) >= 2:
                    # choose the outcome with numerically smaller price (more negative or smaller positive)
                    parsed.sort(key=lambda np: np[1])
                    book_ml_fav, book_ml_price = parsed[0]
                    # record
                    fav_counts[book_ml_fav] = fav_counts.get(book_ml_fav, 0) + 1
                    fav_spreads.setdefault(book_ml_fav, []).append(None)
                    # also store moneyline values for summary
                    totals_points.append(None)  # keep totals_points unaffected; we'll store ML separately below
                    # Use a separate structure to collect ML prices per team
                    # create if missing
                    if 'ml_prices' not in locals():
                        ml_prices = {}
                    ml_prices.setdefault(book_ml_fav, []).append(book_ml_price)

            # end per-book loop

        # Decide consensus favorite
        if fav_counts:
            consensus_fav = max(fav_counts.items(), key=lambda kv: kv[1])[0]
            # If spreads exist for consensus_fav, compute median spread ignoring None
            spread_med = median_or_none([s for s in fav_spreads.get(consensus_fav, []) if s is not None])
        else:
            consensus_fav, spread_med = None, None

        total_med = median_or_none(totals_points)

        # derive a representative moneyline if collected
        rep_ml = None
        if 'ml_prices' in locals() and consensus_fav in ml_prices:
            try:
                # median of collected ML prices for consensus favorite
                rep_ml = statistics.median(ml_prices[consensus_fav])
                # prefer showing as int if whole number
                if float(rep_ml).is_integer():
                    rep_ml = int(rep_ml)
            except Exception:
                rep_ml = ml_prices[consensus_fav][0]

        summaries.append({
            "home": home,
            "away": away,
            "kickoff_iso": start_iso,
            "favorite": consensus_fav,
            "spread": round_half(spread_med, 1) if spread_med is not None else None,
            "total": round_half(total_med, 1) if total_med is not None else None,
            "moneyline": rep_ml,
        })
    # sort by kickoff time
    summaries.sort(key=lambda g: g["kickoff_iso"] or "")
    print(f"[debug] summarize_events: returning {len(summaries)} summaries")
    return summaries

# ----- Copywriting: build an energetic radio script -----
def fmt_kickoff_pt(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except Exception as e:
        print(f"[debug] fmt_kickoff_pt: failed to parse {iso_str}: {e}")
        return "Kickoff time TBA"
    local = dt.astimezone(ZoneInfo("America/Los_Angeles"))
    wkday = local.strftime("%A")
    hour = local.strftime("%-I")
    minute = local.minute
    # Speak :00 times as "one o'clock Pacific" to avoid TTS reading "PT" oddly
    if minute == 0:
        return f"{wkday} {hour} o'clock Pacific"
    # for common quarter/half times, use natural phrases
    if minute == 15:
        return f"{wkday} {hour} quarter past Pacific"
    if minute == 30:
        return f"{wkday} {hour} thirty Pacific"
    if minute == 45:
        # express as quarter to next hour
        next_hour = (local + timedelta(hours=1)).strftime("%-I")
        return f"{wkday} quarter to {next_hour} Pacific"
    # fallback: numeric minute + AM/PM spelled-out timezone
    return f"{wkday} {local.strftime('%-I:%M %p')} Pacific"

def line_for_game(g):
    away, home = g["away"], g["home"]
    ko = fmt_kickoff_pt(g["kickoff_iso"])
    fav, spread, total, ml = g.get("favorite"), g.get("spread"), g.get("total"), g.get("moneyline")

    def _format_ml(m):
        if m is None:
            return None
        try:
            mf = float(m)
            # convert to int when it's essentially integral
            if float(mf).is_integer():
                mi = int(mf)
                return f"{mi:+d}"
            return f"{mf:+.0f}"
        except Exception:
            return str(m)

    ml_str = _format_ml(ml)

    # Build text: prefer spread if available, otherwise use moneyline to indicate favorite
    if fav and spread is not None:
        if fav == home:
            vs = f"{away} at {home}: {home} -{spread}"
        else:
            vs = f"{away} at {home}: {fav} -{spread}"
        if ml_str:
            vs += f" (ML {ml_str})"
    elif fav and ml_str:
        # report favorite by moneyline with requested phrasing
        # e.g. "the Eagles are favored and their moneyline bet is minus 135"
        # Use lowercase 'minus' for negative ml when speaking
        ml_spoken = ml_str.replace('+', '').replace('-', 'minus ')
        vs = f"{away} at {home}: the {fav} are favored and their moneyline bet is {ml_spoken}"
    else:
        vs = f"{away} at {home}: pick'em"

    if total is not None:
        vs += f", total {total}"
    return f"{ko} — {vs}."

def build_script(games):
    intro = (
        "Welcome to your NFL Odds Radio Rundown. "
        "Here are this week’s consensus point spreads and totals across major US books."
    )
    body = "\n".join(line_for_game(g) for g in games)
    outro = "That’s your slate—good luck and enjoy the games."
    return textwrap.dedent(f"""{intro}

    {body}

    {outro}
    """).strip()

# ----- ElevenLabs TTS -----
# API: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
# Python SDK: https://github.com/elevenlabs/elevenlabs-python
from elevenlabs import ElevenLabs

def choose_voice_id(client: ElevenLabs, search_term: str = "announcer") -> str:
    """
    Try to find an American (en-US) voice first, then fall back to search term and defaults.
    """
    voices = []
    try:
        # prefer search first (higher relevance)
        res = client.voices.search(search=search_term, page_size=100)
        voices = getattr(res, "voices", []) or []
    except Exception as e:
        print(f"[debug] choose_voice_id: search failed: {e}")
        voices = []
    try:
        if not voices:
            res2 = client.voices.list()
            voices = getattr(res2, "voices", []) or []
    except Exception:
        pass

    if voices:
        # helper to get string metadata
        def _meta_str(v):
            parts = []
            for k in ("language", "locale", "accent", "name", "title", "voice_id"):
                val = None
                try:
                    val = v.get(k) if isinstance(v, dict) else getattr(v, k, None)
                except Exception:
                    val = None
                if isinstance(val, str):
                    parts.append(val)
            return " ".join(parts).lower()

        # prefer anything that explicitly mentions US/American or en-US
        for v in voices:
            meta = _meta_str(v)
            if "us" in meta or "american" in meta or "en-us" in meta:
                vid = v.get('voice_id') if isinstance(v, dict) else getattr(v, 'voice_id', None)
                if vid:
                    print(f"[debug] choose_voice_id: selected US voice from metadata: {meta}")
                    return vid

        # fallback: prefer English-language voices
        for v in voices:
            meta = _meta_str(v)
            if "en" in meta:
                vid = v.get('voice_id') if isinstance(v, dict) else getattr(v, 'voice_id', None)
                if vid:
                    print(f"[debug] choose_voice_id: selected English voice: {meta}")
                    return vid

        # final fallback: first voice
        v = voices[0]
        vid = v.get('voice_id') if isinstance(v, dict) else getattr(v, 'voice_id', None)
        if vid:
            print(f"[debug] choose_voice_id: falling back to first available voice")
            return vid

    # Safe default from docs
    print("[debug] choose_voice_id: no voices available, falling back to hardcoded default")
    return "JBFqnCBsd6RMkjVDRZzb"

def tts_to_mp3(script_text: str, out_path: str, voice_id: str):
    print(f"[debug] tts_to_mp3: starting conversion, voice_id={voice_id}, script_len={len(script_text)}")
    client = ElevenLabs(api_key=ELEVEN_API_KEY)
    try:
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_44100_128",
            text=script_text,
            model_id="eleven_multilingual_v2",
        )
    except Exception as e:
        print(f"[debug] tts_to_mp3: exception calling ElevenLabs convert: {e}")
        raise

    data = None
    # Various SDKs or errors can return different types: bytes, bytearray, str (base64),
    # an object with .content, a dict, or a generator/iterator yielding chunks.
    try:
        if audio is None:
            print("[debug] tts_to_mp3: convert returned None")
        elif isinstance(audio, (bytes, bytearray)):
            data = bytes(audio)
            print(f"[debug] tts_to_mp3: received bytes len={len(data)}")
        elif isinstance(audio, str):
            # Some SDKs may return a base64 string
            try:
                data = base64.b64decode(audio)
                print(f"[debug] tts_to_mp3: decoded base64 string to bytes len={len(data)}")
            except Exception:
                print(f"[debug] tts_to_mp3: received str (len={len(audio)}) but base64 decode failed")
        elif hasattr(audio, "content"):
            data = getattr(audio, "content")
            print(f"[debug] tts_to_mp3: used .content attribute, len={len(data) if hasattr(data,'__len__') else 'n/a'}")
        elif isinstance(audio, dict):
            print(f"[debug] tts_to_mp3: SDK returned dict/error payload: {audio}")
        else:
            # Support generator/streaming responses
            import types
            if isinstance(audio, types.GeneratorType) or hasattr(audio, '__iter__'):
                chunks = []
                try:
                    for i, chunk in enumerate(audio):
                        if chunk is None:
                            continue
                        # bytes directly
                        if isinstance(chunk, (bytes, bytearray)):
                            chunks.append(bytes(chunk))
                            continue
                        # base64-encoded string chunk
                        if isinstance(chunk, str):
                            try:
                                chunks.append(base64.b64decode(chunk))
                                continue
                            except Exception:
                                # not base64; skip
                                continue
                        # dict-like chunk might carry audio under common keys
                        if isinstance(chunk, dict):
                            for k in ('audio', 'data', 'content'):
                                if k in chunk:
                                    val = chunk[k]
                                    if isinstance(val, (bytes, bytearray)):
                                        chunks.append(bytes(val))
                                        break
                                    if isinstance(val, str):
                                        try:
                                            chunks.append(base64.b64decode(val))
                                            break
                                        except Exception:
                                            continue
                        # object with .content
                        if hasattr(chunk, 'content'):
                            c = getattr(chunk, 'content')
                            if isinstance(c, (bytes, bytearray)):
                                chunks.append(bytes(c))
                            elif isinstance(c, str):
                                try:
                                    chunks.append(base64.b64decode(c))
                                except Exception:
                                    pass
                    if chunks:
                        data = b"".join(chunks)
                        print(f"[debug] tts_to_mp3: collected {len(chunks)} chunks from generator, total_bytes={len(data)}")
                    else:
                        print(f"[debug] tts_to_mp3: generator produced no usable chunks (sample first chunk type unknown)")
                except Exception as e:
                    print(f"[debug] tts_to_mp3: error iterating generator: {e}")
            else:
                # last resort: try to coerce
                try:
                    data = bytes(audio)
                    print(f"[debug] tts_to_mp3: coerced audio to bytes len={len(data)}")
                except Exception:
                    print(f"[debug] tts_to_mp3: unknown audio return type {type(audio)}")
    except Exception as e:
        print(f"[debug] tts_to_mp3: error while inspecting audio return value: {e}")

    if not data or len(data) == 0:
        print("[debug] tts_to_mp3: no audio bytes produced; aborting write and dumping diagnostics")
        # Try to provide more context if possible
        try:
            if isinstance(audio, dict):
                print("[debug] tts_to_mp3: error payload keys:", list(audio.keys()))
        except Exception:
            pass
        raise RuntimeError("No audio returned from ElevenLabs TTS (empty result)")

    # SDK returns bytes; write to file
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"[debug] tts_to_mp3: wrote file {out_path}")

def football_week_iso_range(now: Optional[datetime] = None):
    """Return (start_iso, end_iso) for the current football week.

    Football week is defined as Thursday 00:00 (local UTC day) through the following
    Monday 23:59:59. If today is Tuesday or Wednesday, this returns the upcoming
    Thursday->Monday window. For Thu-Mon it returns the current window that contains today.
    """
    now = now or datetime.now(timezone.utc)
    wd = now.weekday()  # Mon=0 .. Sun=6
    # If today is Tue(1) or Wed(2) we select upcoming Thursday; otherwise select the
    # Thursday on-or-before today (which handles Thu-Fri-Sat-Sun-Mon)
    if wd in (1, 2):
        days_until_thu = 3 - wd
        thursday = (now + timedelta(days=days_until_thu)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        if wd >= 3:
            days_back = wd - 3
        else:
            # Monday (0) -> go back 4 days to Thursday
            days_back = 4
        thursday = (now - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0)

    # End is the following Monday at 23:59:59
    end = thursday + timedelta(days=4, hours=23, minutes=59, seconds=59)
    print(f"[debug] football_week_iso_range: thursday={thursday.isoformat()} end={end.isoformat()} now={now.isoformat()}")
    return thursday.isoformat(), end.isoformat()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate an NFL odds podcast MP3 using ElevenLabs TTS.")
    parser.add_argument("--days", type=int, default=DAYS_AHEAD_DEFAULT, help="Days ahead to include")
    parser.add_argument("--raw-file", default=None, help="Optional raw snapshot JSON (saved by nfl_odds_snapshot) to use instead of calling the Odds API")
    parser.add_argument("--voice-search", default=VOICE_SEARCH_DEFAULT, help="Search term to choose ElevenLabs voice")
    parser.add_argument("--voice-id", default=None, help="Explicit ElevenLabs voice id to use (overrides search)")
    parser.add_argument("--eleven-api-key", default=os.getenv("ELEVENLABS_API_KEY"), help="ElevenLabs API key (or set ELEVENLABS_API_KEY)")
    parser.add_argument("--dry-run", action="store_true", help="Run the script without making external calls (for testing)")
    args = parser.parse_args()

    # Determine ElevenLabs API key
    eleven_api_key = args.eleven_api_key or ELEVEN_API_KEY
    if not eleven_api_key:
        print("Missing ElevenLabs API key"); sys.exit(2)

    # Determine source of odds: either cached raw file or call fetch_odds from nfl_odds_snapshot
    if args.raw_file:
        if not os.path.isfile(args.raw_file):
            print(f"Raw file {args.raw_file} not found"); sys.exit(2)
        with open(args.raw_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
        headers = {}
        print(f"Loaded {len(events)} events from cached file {args.raw_file}")
        print(f"[debug] raw_file type: {type(events).__name__}")
        if isinstance(events, list) and events and isinstance(events[0], dict):
            print(f"[debug] raw_file first event keys: {sorted(events[0].keys())}")
    else:
        # Use the same API key resolution as nfl_odds_snapshot
        odds_api_key = _default_api_key()
        if not odds_api_key:
            print("No Odds API key found; set THE_ODDS_API_KEY or add to ~/.bashrc"); sys.exit(2)

        commence_from, commence_to = iso_range_for_next_days(args.days)
        print(f"[debug] main: fetching odds from {commence_from} to {commence_to} using api_key_set={bool(odds_api_key)}")
        resp, events = fetch_odds(odds_api_key, commence_from=commence_from, commence_to=commence_to)
        headers = getattr(resp, 'headers', {}) if resp is not None else {}
        print(f"[debug] main: fetched events type={type(events).__name__} len={len(events) if hasattr(events, '__len__') else 'n/a'} headers_keys={list(headers.keys())[:10]}")

    # Restrict to the current football week (Thu -> Mon)
    start_iso, end_iso = football_week_iso_range()
    try:
        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    except Exception:
        # fallback: parse naive
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = datetime.fromisoformat(end_iso)

    def event_in_week(ev):
        k = ev.get("commence_time")
        if not k:
            return False
        try:
            dt = datetime.fromisoformat(k.replace("Z", "+00:00"))
        except Exception:
            return False
        return (dt >= start_dt) and (dt <= end_dt)

    if isinstance(events, list):
        original_len = len(events)
        events = [e for e in events if event_in_week(e)]
        print(f"[debug] filtered events to football week: {len(events)} of {original_len} remain")
    else:
        print(f"[debug] events is not a list; skipping football-week filtering (type={type(events).__name__})")

    games = summarize_events(events)
    print(f"Summarized into {len(games)} games")
    if games:
        print("First game summary:", games[0])
    if not games:
        print("No NFL games returned in the selected window.")
        return
    script = build_script(games)
    print("\n--- Generated Script ---\n")
    print(script)
    print("\n------------------------\n")

    if args.dry_run:
        print("Dry run enabled -- skipping ElevenLabs TTS and file write.")
        return

    # pick a voice
    try:
        client = ElevenLabs(api_key=eleven_api_key)
        if args.voice_id:
            voice_id = args.voice_id
            print(f"Using explicit voice id: {voice_id}")
        else:
            voice_id = choose_voice_id(client, args.voice_search)
            print(f"Using voice id: {voice_id}")
    except Exception as e:
        print("Error creating ElevenLabs client or choosing voice:", e)
        return
    out_file = "weekly_odds_rundown.mp3"
    try:
        tts_to_mp3(script, out_file, voice_id)
        print(f"Saved audio to: {out_file}")
    except Exception as e:
        print("Error during TTS or file write:", e)
        return

if __name__ == "__main__":
    main()
