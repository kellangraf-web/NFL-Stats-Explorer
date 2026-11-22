"""
Builds players.json for the QB First-N-Seasons Explorer.

Downloads nflverse game-level player stats (1999 -> current season), splits every
player's career into season 1, 2, 3, ... (counted from his rookie year), and saves
games / TDs / yards per season, separated into regular season, playoffs and
wild-card playoff games. Pre-1999 careers (Marino, Favre, Kelly, etc.) are added
by hand at the bottom.

Run:  python build_data.py        (writes players.json next to this file)
"""
import datetime as dt
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
OUT = Path(__file__).with_name("players.json")
MAXK = 23            # longest window the slider offers (Tom Brady: 2000-2022)


# ---------------------------------------------------------------------------
# 1. Work out which NFL season is "current"
# ---------------------------------------------------------------------------
def current_season(today=None):
    """NFL seasons start in September; Jan-Aug games/offseason belong to last year's season."""
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


# ---------------------------------------------------------------------------
# 2. Download helpers
# ---------------------------------------------------------------------------
def fetch_parquet(url, tries=4):
    """Download a parquet file into a DataFrame; return None if it doesn't exist (404)."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return pd.read_parquet(io.BytesIO(r.read()))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                      # e.g. new season not published yet
            if attempt == tries - 1:
                raise
        except urllib.error.URLError:
            if attempt == tries - 1:
                raise
        time.sleep(5 * (attempt + 1))           # back off and retry


def load_weekly(cur):
    """Stack every season's weekly player stats, 1999 -> cur."""
    cols = ["player_id", "player_display_name", "season", "week", "season_type",
            "passing_tds", "rushing_tds", "receiving_tds",
            "passing_yards", "rushing_yards", "receiving_yards"]
    frames = []
    for yr in range(1999, cur + 1):
        df = fetch_parquet(f"{BASE}/stats_player/stats_player_week_{yr}.parquet")
        if df is None:
            print(f"  {yr}: not published yet, skipping")
            continue
        frames.append(df[cols])
        print(f"  {yr}: {len(df):,} rows")
    wk = pd.concat(frames, ignore_index=True)
    # Regular season + playoffs only (preseason is excluded)
    return wk[wk.season_type.isin(["REG", "POST"])].fillna(0)


# ---------------------------------------------------------------------------
# 3. Build per-player, per-season rows
# ---------------------------------------------------------------------------
def build():
    cur = current_season()
    print(f"Current season: {cur}")
    wk = load_weekly(cur)

    # --- Career start = earliest of rookie season, draft year, first game in data ---
    pl = fetch_parquet(f"{BASE}/players/players.parquet")[["gsis_id", "rookie_season", "draft_year"]]
    first_seen = wk.groupby("player_id").season.min().rename("first_seen")
    wk = wk.merge(pl, left_on="player_id", right_on="gsis_id", how="left").merge(first_seen, on="player_id")
    wk["start"] = wk[["rookie_season", "draft_year", "first_seen"]].min(axis=1)

    # Careers that began before 1999: fix the start year; seasons before 1999 are added by hand below
    START = {"Peyton Manning": 1998, "Kurt Warner": 1998, "Brett Favre": 1991, "Dan Marino": 1983}
    wk["start"] = wk.player_display_name.map(START).fillna(wk.start)
    # Any other pre-1999 career would be incomplete, so leave it out
    wk = wk[(wk.start >= 1999) | wk.player_display_name.isin(START.keys())]

    # --- Season index within the career (1 = rookie year) ---
    wk["k"] = (wk.season - wk.start + 1).astype(int)
    wk = wk[wk.k <= MAXK]

    # --- Game type: 0 = regular season, 1 = playoff, 2 = wild-card playoff ---
    # The wild-card round is the first playoff week of each season.
    wc_week = wk[wk.season_type == "POST"].groupby("season").week.min()
    wk["pt"] = 0
    wk.loc[wk.season_type == "POST", "pt"] = 1
    wk.loc[(wk.season_type == "POST") & (wk.week == wk.season.map(wc_week)), "pt"] = 2

    # --- Totals: total TDs / yards = passing + rushing + receiving ---
    wk["ttd"] = wk.passing_tds + wk.rushing_tds + wk.receiving_tds
    wk["tyd"] = wk.passing_yards + wk.rushing_yards + wk.receiving_yards
    agg = wk.groupby(["player_id", "player_display_name", "start", "k", "pt"]).agg(
        g=("week", "size"),                    # games with at least one recorded stat
        ptd=("passing_tds", "sum"), ttd=("ttd", "sum"),
        pyd=("passing_yards", "sum"), tyd=("tyd", "sum")).reset_index()

    # --- Keep players who could plausibly chart (keeps the file small) ---
    tot = agg.groupby("player_id").tyd.sum()
    best = agg.groupby(["player_id", "k"]).tyd.sum().groupby("player_id").max()
    keep = set(tot[tot >= 5000].index) | set(best[best >= 1500].index)
    agg = agg[agg.player_id.isin(keep)]

    # --- One record per player; each row = [season#, gameType, G, passTD, totalTD, passYds, totalYds] ---
    players = []
    for (pid, name, start), grp in agg.groupby(["player_id", "player_display_name", "start"]):
        rows = [[int(r.k), int(r.pt), int(r.g), int(r.ptd), int(r.ttd), int(r.pyd), int(r.tyd)]
                for r in grp.itertuples()]
        players.append({"n": name, "s": int(start), "r": rows})

    add_hand_entered(players)

    # --- Label for "data through": last week with games in the current season ---
    cur_rows = wk[wk.season == cur]
    if len(cur_rows):
        last = cur_rows.sort_values(["week"]).iloc[-1]
        through = (f"Week {int(last.week)}" if last.season_type == "REG"
                   else "the playoffs")
    else:
        through = "the start of the season"

    out = {"meta": {"season": cur, "through": through}, "players": players}
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {OUT.name}: {len(players)} players, {cur} data through {through}")


# ---------------------------------------------------------------------------
# 4. Hand-entered seasons before 1999 (NFL.com, Pro Football Reference, StatMuse)
#    Rushing figures and a few playoff per-game splits are approximate.
# ---------------------------------------------------------------------------
def season(k, g, ptd, pyd, rtd, ryd, post=0):
    """One row: season #k, games, pass TD, pass yds, rush TD, rush yds; post 0=reg, 1=playoff, 2=wild card."""
    return [k, post, g, ptd, ptd + rtd, pyd, pyd + ryd]


def add_hand_entered(players):
    def P(name):
        return next(p for p in players if p["n"] == name)

    # Rookie (1998) seasons that predate the nflverse data
    P("Peyton Manning")["r"].append([1, 0, 16, 26, 26, 3739, 3739 + 62])
    P("Kurt Warner")["r"].append([1, 0, 1, 0, 0, 39, 39])

    # Marino 1983-1998 (1999 comes from nflverse). 1994 playoffs = remainder of career playoff totals.
    marino = [season(1,11,20,2210,2,45), season(2,16,48,5084,0,-7), season(3,16,30,4137,0,-24),
              season(4,16,44,4746,0,-3), season(5,12,26,3245,1,-5), season(6,16,28,4434,0,-17),
              season(7,16,24,3997,2,-7), season(8,16,21,3563,0,29), season(9,16,25,3970,1,32),
              season(10,16,24,4116,0,66), season(11,5,8,1218,1,-4), season(12,16,30,4453,1,-6),
              season(13,14,24,3668,0,14-6), season(14,13,17,2795,0,-3), season(15,16,16,3780,0,-14),
              season(16,16,23,3497,1,-3),
              season(1,1,2,193,0,0,1), season(2,3,8,1001,0,0,1), season(3,2,2,486,0,0,1),
              season(8,1,2,221,0,0,2), season(8,1,3,323,0,0,1), season(10,2,4,435,0,0,1),
              season(12,1,2,257,0,0,2), season(12,1,4,262,0,0,1),
              season(13,1,2,422,0,0,2), season(15,1,0,141,0,0,2),
              season(16,1,1,235,0,0,2), season(16,1,0,243,0,0,1)]
    # Favre 1991-1998 (1999-2010 from nflverse)
    favre = [season(1,2,0,0,0,0), season(2,15,18,3227,1,198-7), season(3,16,19,3303,1,216),
             season(4,16,33,3882,2,202), season(5,16,38,4413,3,181), season(6,16,39,3899,2,136),
             season(7,16,35,3867,1,187), season(8,16,31,4212,1,133),
             season(3,1,3,204,0,0,2), season(3,1,2,331,0,0,1), season(4,1,0,262,0,0,2),
             season(4,1,0,211,0,0,1), season(5,1,3,199,0,0,2), season(5,2,5,606,0,0,1),
             season(6,3,5,617,0,0,1), season(7,3,5,668,0,0,1), season(8,1,2,292,0,0,2)]
    # Kelly 1986-1996 (whole career)
    kelly = [season(1,16,22,3593,0,199), season(2,12,19,2798,0,133), season(3,16,15,2403,0,154),
             season(4,13,25,3130,2,137), season(5,14,24,2829,0,63), season(6,15,33,3844,1,45),
             season(7,16,23,3457,1,53), season(8,16,18,3382,0,102), season(9,14,22,3114,1,77),
             season(10,15,22,3130,0,33), season(11,13,14,2810,0,0),
             season(3,2,1,407,0,0,1), season(4,1,4,405,0,0,1), season(5,3,5,851,0,0,1),
             season(6,3,5,665,0,0,1), season(7,2,1,259,0,0,1), season(8,3,2,707,0,0,1),
             season(10,1,1,195,0,0,2), season(10,1,1,135,0,0,1), season(11,1,1,239,0,0,2)]

    P("Dan Marino")["r"] += marino;  P("Dan Marino")["approx"] = 1
    P("Brett Favre")["r"] += favre;  P("Brett Favre")["approx"] = 1
    players.append({"n": "Jim Kelly", "s": 1986, "r": kelly, "approx": 1})


if __name__ == "__main__":
    build()
