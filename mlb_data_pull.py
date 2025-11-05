#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLB pitch + event pipeline (API-first, SQLite output)

- Writes TWO tables to SQLite: `events` and `pitches`
- Adds a `year` column (== season) to both tables
- No CSV files are produced

Dependencies:
    pip install pandas pyarrow requests tqdm python-dateutil sqlalchemy
"""

from __future__ import annotations

import os
import io
import time
import random
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Dict

import pandas as pd
import requests
from tqdm import tqdm
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from sqlalchemy import create_engine, text

# ----------------------------- Config -----------------------------

DB_PATH = "mlb_stats.db"           # SQLite file to write
LOG_DIR = "logs"

# Set the range of seasons you want.
# NOTE: Pre-2008 will not have Statcast tracking; those columns will be blank.
START_SEASON = 2008
END_SEASON = date.today().year

# Endpoints
SAVANT_BASE = "https://baseballsavant.mlb.com/statcast_search/csv"
MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
MLB_FEED = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"

# Statcast fetch window (<=6 days is safest)
STATCAST_WINDOW_DAYS = 5

# Retry / networking
HTTP_TIMEOUT = 60
MAX_RETRIES = 6                     # total attempts per request
BACKOFF_BASE = 1.6                  # exponential backoff base
STATUS_FORCELIST = (429, 500, 502, 503, 504)


# ------------------------- Helpers / Setup ------------------------

def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)


def season_dates(season: int) -> Tuple[date, date]:
    # Conservative window that comfortably includes preseason/postseason
    return date(season, 2, 1), date(season, 12, 1)


def chunk_dates(start: date, end: date, step_days: int = STATCAST_WINDOW_DAYS) -> List[Tuple[date, date]]:
    chunks = []
    d = start
    while d <= end:
        e = min(d + timedelta(days=step_days - 1), end)
        chunks.append((d, e))
        d = e + timedelta(days=1)
    return chunks


def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,   # retry DNS/connect issues
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=0,      # we do our own jittered sleep below
        status_forcelist=STATUS_FORCELIST,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "mlb-pitch-pipeline/2.0-sqlite"})
    return s


SESSION = build_session()


def http_get(url: str, *, params: Dict | None = None, timeout: int = HTTP_TIMEOUT, expect_json: bool = False):
    """
    GET with explicit retry + jittered backoff, covering DNS failures/timeouts and 5xx/429.
    Returns Response (or parsed JSON if expect_json=True).
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            # If rate-limited or server error, allow retry (urllib3 Retry already helps too)
            if r.status_code in STATUS_FORCELIST or (not r.content and attempt < MAX_RETRIES - 1):
                raise requests.exceptions.RetryError(f"Retryable status {r.status_code}")
            if expect_json:
                return r.json()
            return r
        except Exception as e:
            last_exc = e
            sleep = (BACKOFF_BASE ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep)
    # Exhausted attempts
    raise last_exc


# ----------------------- MLB Stats API calls ----------------------

def list_game_pks(season: int) -> List[int]:
    """
    List all MLB gamePk values for the given season via the schedule endpoint.
    """
    start, end = season_dates(season)
    pks: List[int] = []

    d = start
    while d <= end:
        # Walk month-by-month to keep responses small
        month_end = (d.replace(day=1) + relativedelta(months=1) - timedelta(days=1))
        window_end = min(month_end, end)
        params = {"sportId": 1, "startDate": d.isoformat(), "endDate": window_end.isoformat()}
        js = http_get(MLB_SCHEDULE, params=params, expect_json=True)
        for date_block in js.get("dates", []) or []:
            for g in date_block.get("games", []) or []:
                pk = g.get("gamePk")
                if isinstance(pk, int):
                    pks.append(pk)
        d = window_end + timedelta(days=1)

    return sorted(set(pks))


def _safeget(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def parse_feed_to_events_and_pitches(game_pk: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Call the game feed and return (events_df, pitches_df) for one game.
    """
    url = MLB_FEED.format(gamePk=game_pk)
    js = http_get(url, expect_json=True)

    plays = _safeget(js, "liveData", "plays", "allPlays", default=[]) or []
    ev_rows, pi_rows = [], []

    for play_index, play in enumerate(plays):
        about = play.get("about", {}) or {}
        result = play.get("result", {}) or {}
        matchup = play.get("matchup", {}) or {}
        count_end = play.get("count", {}) or {}

        # Stable event_id: prefer playId; else synthesize from gamePk + index
        event_id = play.get("playId") or f"{game_pk}:{play_index}"

        # Plate appearance / event row
        ev_rows.append({
            "event_id": event_id,
            "season": None,  # filled later
            "game_pk": game_pk,
            "at_bat_index": play.get("atBatIndex"),
            "start_time": about.get("startTime"),
            "end_time": about.get("endTime"),
            "inning": about.get("inning"),
            "half_inning": about.get("halfInning"),
            "is_scoring_play": about.get("isScoringPlay"),
            "event": result.get("event"),
            "event_type": result.get("eventType"),
            "rbi": result.get("rbi"),
            "away_score": result.get("awayScore"),
            "home_score": result.get("homeScore"),
            "description": result.get("description"),
            "batter_id": _safeget(matchup, "batter", "id"),
            "pitcher_id": _safeget(matchup, "pitcher", "id"),
            "bat_side": _safeget(matchup, "batSide", "code"),
            "pitch_hand": _safeget(matchup, "pitchHand", "code"),
            "balls_end": count_end.get("balls"),
            "strikes_end": count_end.get("strikes"),
            "outs_end": count_end.get("outs"),
        })

        # Pitch rows (count BEFORE each pitch is in each playEvent.count)
        pitch_no = 0
        for ev in play.get("playEvents", []) or []:
            if not ev.get("isPitch"):
                continue
            pitch_no += 1
            det = ev.get("details", {}) or {}
            pdta = ev.get("pitchData", {}) or {}
            coords = pdta.get("coordinates", {}) or {}
            brk = pdta.get("breaks", {}) or {}

            pi_rows.append({
                "event_id": event_id,
                "season": None,  # filled later
                "game_pk": game_pk,
                "at_bat_index": play.get("atBatIndex"),
                "pitch_number": pitch_no,
                # call / description / pitch type (feed)
                "call_code": _safeget(det, "call", "code"),
                "call_desc": _safeget(det, "call", "description"),
                "des": det.get("description"),
                "pitch_type_code": _safeget(pdta, "type", "code"),
                "pitch_type_desc": _safeget(pdta, "type", "description"),
                # feed speeds & location (not as complete as Statcast)
                "start_speed": pdta.get("startSpeed"),
                "end_speed": pdta.get("endSpeed"),
                "plate_x": coords.get("pX"),
                "plate_z": coords.get("pZ"),
                "sz_top": pdta.get("strikeZoneTop"),
                "sz_bot": pdta.get("strikeZoneBottom"),
                "break_x": brk.get("breakHorizontal"),
                "break_z": brk.get("breakVertical"),
                "spin_rate": _safeget(pdta, "breaks", "spinRate"),
                # count before this pitch
                "balls_before": _safeget(ev, "count", "balls"),
                "strikes_before": _safeget(ev, "count", "strikes"),
                "outs_before": _safeget(ev, "count", "outs"),
            })

    events_df = pd.DataFrame(ev_rows)
    pitches_df = pd.DataFrame(pi_rows)
    return events_df, pitches_df


# -------------------- Baseball Savant (Statcast) ------------------

def fetch_statcast_chunk(start_dt: date, end_dt: date) -> pd.DataFrame:
    """
    Pull per-pitch tracking for [start_dt, end_dt] (inclusive).
    We use player_type=pitcher & type=details for the richest columns.
    """
    params = {
        "all": "true",
        "player_type": "pitcher",
        "type": "details",
        "min_pitches": 0,
        "min_results": 0,
        "hfGT": "R|",
        "game_date_gt": start_dt.isoformat(),
        "game_date_lt": end_dt.isoformat(),
    }
    r = http_get(SAVANT_BASE, params=params)
    content = r.content.decode("utf-8", errors="ignore").strip()
    if not content:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(content))

    # Normalize join keys
    if "at_bat_number" in df.columns and "at_bat_index" not in df.columns:
        df["at_bat_index"] = df["at_bat_number"]

    for key in ("game_pk", "at_bat_index", "pitch_number"):
        if key not in df.columns:
            df[key] = pd.NA

    return df


def fetch_statcast_season(season: int) -> pd.DataFrame:
    start, end = season_dates(season)
    frames = []
    for s, e in tqdm(chunk_dates(start, end, STATCAST_WINDOW_DAYS), desc=f"Statcast {season}"):
        try:
            chunk = fetch_statcast_chunk(s, e)
            if not chunk.empty:
                chunk["season"] = season
                frames.append(chunk)
        except Exception as e:
            print(f"[WARN] Statcast chunk {s}..{e} failed: {e}")

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame(columns=["season", "game_pk", "at_bat_index", "pitch_number"])


# ------------------------ SQLite utilities -----------------------

def _sqlite_type_from_dtype(dtype) -> str:
    """Basic pandas dtype -> SQLite affinity mapping."""
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"  # 0/1
    return "TEXT"


def ensure_table_exists_and_expand(conn, table: str, df: pd.DataFrame):
    """
    Create table if not exists (via pandas on first write).
    If the table exists and df has new columns, ALTER TABLE to add them.
    """
    cols = conn.execute(text(f'PRAGMA table_info("{table}")')).fetchall()
    existing = {row[1] for row in cols}  # row[1] = column name
    if not existing:
        # Table doesn't exist yet — create with current schema
        df.head(0).to_sql(table, conn, if_exists="replace", index=False)
        existing = set(df.columns)

    missing = [c for c in df.columns if c not in existing]
    for c in missing:
        sql_type = _sqlite_type_from_dtype(df[c].dtype)
        conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{c}" {sql_type}'))


def _sqlite_safe_chunksize(ncols: int, max_params: int = 999) -> int:
    """
    For SQLite's ~999-parameter limit, pick a rows-per-batch so that
    rows * ncols <= max_params. Guarantee at least 1 row per batch.
    """
    return max(1, max_params // max(1, ncols))


def write_df_append(engine, table: str, df: pd.DataFrame):
    """Append df into table, expanding schema if new columns appear, respecting SQLite param cap."""
    if df.empty:
        return

    ncols = len(df.columns)
    safe_chunk = _sqlite_safe_chunksize(ncols)  # e.g., with 120 cols → 8 rows/batch

    with engine.begin() as conn:
        ensure_table_exists_and_expand(conn, table, df)

        try:
            # Fast path: multi-row INSERTs within SQLite's param cap
            df.to_sql(
                name=table,
                con=conn,
                if_exists="append",
                index=False,
                chunksize=safe_chunk,
                method="multi",
            )
        except Exception:
            # Fallback: row-at-a-time if something still complains (slower but robust)
            df.to_sql(
                name=table,
                con=conn,
                if_exists="append",
                index=False,
                chunksize=1,
                method=None,
            )

        # helpful indices
        if "year" in df.columns:
            conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_{table}_year ON "{table}" ("year")'))
        if table == "pitches":
            for col in ("game_pk", "at_bat_index", "pitch_number"):
                if col in df.columns:
                    conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON "{table}" ("{col}")'))


# -------------------------- Orchestration -------------------------

def build_for_season(season: int, engine):
    print(f"=== Season {season} ===")
    ensure_dirs()

    failed: list[tuple[int, str]] = []
    all_events = []
    all_feed_pitches = []

    # 1) Enumerate games, then pull feeds
    pks = list_game_pks(season)
    for pk in tqdm(pks, desc=f"MLB feed {season}"):
        try:
            ev_df, pi_df = parse_feed_to_events_and_pitches(pk)
            if not ev_df.empty:
                ev_df["season"] = season
                ev_df["year"] = season          # <-- explicit year column
                all_events.append(ev_df)
            if not pi_df.empty:
                pi_df["season"] = season
                pi_df["year"] = season          # <-- explicit year column
                all_feed_pitches.append(pi_df)
        except Exception as e:
            failed.append((pk, str(e)))

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    feed_pitches_df = pd.concat(all_feed_pitches, ignore_index=True) if all_feed_pitches else pd.DataFrame()

    # 2) Statcast tracking (2008+ only)
    if season >= 2008:
        statcast_df = fetch_statcast_season(season)
        # Ensure a broad set of Statcast columns exists, even if missing in some years.
        stat_keep = [
            "season", "game_pk", "at_bat_index", "pitch_number",
            "game_date", "batter", "pitcher",
            "events", "description",
            "pitch_type", "release_speed", "release_pos_x", "release_pos_y", "release_pos_z",
            "pfx_x", "pfx_z", "plate_x", "plate_z", "zone",
            "sz_top", "sz_bot", "spin_rate_deprecated", "release_extension",
            "vx0", "vy0", "vz0", "ax", "ay", "az",
            "hc_x", "hc_y", "launch_speed", "launch_angle", "effective_speed",
            "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
        ]
        for c in stat_keep:
            if c not in statcast_df.columns:
                statcast_df[c] = pd.NA
        statcast_df = statcast_df[stat_keep]
        statcast_df["year"] = season
    else:
        statcast_df = pd.DataFrame(columns=["season", "game_pk", "at_bat_index", "pitch_number", "year"])

    # 3) Merge feed-derived pitches with Statcast on join keys (outer to keep everything)
    if statcast_df.empty and feed_pitches_df.empty:
        merged_pitches = pd.DataFrame(columns=["season", "game_pk", "at_bat_index", "pitch_number", "year"])
    elif statcast_df.empty:
        merged_pitches = feed_pitches_df.copy()
    elif feed_pitches_df.empty:
        merged_pitches = statcast_df.copy()
    else:
        merged_pitches = pd.merge(
            feed_pitches_df,
            statcast_df,
            on=["season", "game_pk", "at_bat_index", "pitch_number", "year"],
            how="outer",
            suffixes=("_feed", "_statcast"),
        )

    # 4) Append to SQLite (no CSVs)
    write_df_append(engine, "events", events_df)
    write_df_append(engine, "pitches", merged_pitches)

    # 5) Log failures (if any)
    if failed:
        with open(os.path.join(LOG_DIR, f"failed_game_pks_{season}.txt"), "w") as f:
            for pk, err in failed:
                f.write(f"{pk}\t{err}\n")
        print(f"[INFO] Season {season}: {len(failed)} game feeds failed (see {LOG_DIR}/failed_game_pks_{season}.txt)")


def main():
    ensure_dirs()

    engine = create_engine(f"sqlite:///{os.path.abspath(DB_PATH)}", future=True)

    # Pragmas for better write throughput
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        conn.exec_driver_sql("PRAGMA temp_store=MEMORY;")

        # fresh DB each run: drop tables if they exist (optional; comment out to keep appending)
        conn.execute(text('DROP TABLE IF EXISTS "events"'))
        conn.execute(text('DROP TABLE IF EXISTS "pitches"'))

    for season in range(START_SEASON, END_SEASON + 1):
        build_for_season(season, engine)

    print(f"Done. Wrote tables 'events' and 'pitches' to {DB_PATH}")


if __name__ == "__main__":
    main()
