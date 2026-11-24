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
OUT = Path(__file__).with_name("players.json")        # per-season totals (loads first)
OUT_GAMES = Path(__file__).with_name("games.json")    # per-game rows (loaded when the slider counts games)
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
    cols = ["player_id", "player_display_name", "position", "season", "week", "season_type",
            "passing_tds", "rushing_tds", "receiving_tds",
            "passing_yards", "rushing_yards", "receiving_yards", "carries", "receptions"]
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
    wk = wk[wk.season_type.isin(["REG", "POST"])]
    num = wk.columns.drop(["player_id", "player_display_name", "position", "season_type"])
    wk[num] = wk[num].fillna(0)            # missing stat = 0
    wk["position"] = wk["position"].fillna("")
    return wk


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
    wk["syd"] = wk.rushing_yards + wk.receiving_yards          # scrimmage yards (RB filter)

    # --- Primary position = the position a player is listed at most often ---
    pos = wk[wk.position != ""].groupby("player_id").position.agg(lambda x: x.mode().iat[0])
    modern = wk.start >= 1999                                  # complete careers only

    # Each group: who's in it, which stats (output name -> source column), and a size filter
    qb, qb_games = build_group(wk, wk.index == wk.index,
        [("ptd", "passing_tds"), ("ttd", "ttd"), ("pyd", "passing_yards"), ("tyd", "tyd")],
        keep_col="tyd", total_min=5000, season_min=1500)
    add_hand_entered(qb, qb_games)
    rb, rb_games = build_group(wk, modern & wk.player_id.map(pos).eq("RB"),
        [("att", "carries"), ("ryd", "rushing_yards"), ("rtd", "rushing_tds"),
         ("recyd", "receiving_yards"), ("rectd", "receiving_tds")],
        keep_col="syd", total_min=3000, season_min=800)
    wr, wr_games = build_group(wk, modern & wk.player_id.map(pos).eq("WR"),
        [("rec", "receptions"), ("recyd", "receiving_yards"), ("rectd", "receiving_tds")],
        keep_col="receiving_yards", total_min=3000, season_min=800)
    # Tight ends: same receiving stats as WRs, lower size filter (TEs catch fewer yards)
    te, te_games = build_group(wk, modern & wk.player_id.map(pos).eq("TE"),
        [("rec", "receptions"), ("recyd", "receiving_yards"), ("rectd", "receiving_tds")],
        keep_col="receiving_yards", total_min=2000, season_min=500)
    finalize_games(rb_games); finalize_games(wr_games); finalize_games(te_games)          # (QB list finalized with hand data)

    # --- Label for "data through": last week with games in the current season ---
    cur_rows = wk[wk.season == cur]
    if len(cur_rows):
        last = cur_rows.sort_values(["week"]).iloc[-1]
        through = (f"Week {int(last.week)}" if last.season_type == "REG"
                   else "the playoffs")
    else:
        through = "the start of the season"

    out = {"meta": {"season": cur, "through": through}, "players": qb, "rbs": rb, "wrs": wr, "tes": te}
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    games = {"players": qb_games, "rbs": rb_games, "wrs": wr_games, "tes": te_games}
    OUT_GAMES.write_text(json.dumps(games, separators=(",", ":")))
    print(f"Wrote {OUT.name}: {len(qb)} QB-list, {len(rb)} RBs, {len(wr)} WRs, {len(te)} TEs; "
          f"{OUT_GAMES.name}; {cur} data through {through}")


def build_group(wk, mask, stats, keep_col, total_min, season_min):
    """Per-season rows and per-game rows for one group of players.

    Season row: [season#, gameType, G, stat1, stat2, ...]
    Game row:   [season#, gameType, stat1, stat2, ...]   (chronological)
    gameType: 0 = regular season, 1 = playoff, 2 = wild-card playoff.
    """
    w = wk[mask]
    # Keep players who could plausibly chart: enough total yards, or one big season
    tot = w.groupby("player_id")[keep_col].sum()
    best = w.groupby(["player_id", "k"])[keep_col].sum().groupby("player_id").max()
    keep = set(tot[tot >= total_min].index) | set(best[best >= season_min].index)
    w = w[w.player_id.isin(keep)]
    names = [n for n, _ in stats]

    # Season totals
    agg = w.groupby(["player_id", "player_display_name", "start", "k", "pt"]).agg(
        g=("week", "size"), **{n: (src, "sum") for n, src in stats}).reset_index()
    seasons = []
    for (pid, name, start), grp in agg.groupby(["player_id", "player_display_name", "start"]):
        rows = [[int(r.k), int(r.pt), int(r.g)] + [int(getattr(r, n)) for n in names]
                for r in grp.itertuples()]
        seasons.append({"id": pid, "n": name, "s": int(start), "r": rows})

    # Per-game rows, oldest first (o = sort key within a season)
    w = w.sort_values(["season", "week"])
    games = {}
    for pid, grp in w.groupby("player_id", sort=False):
        games[pid] = [[int(r.k), int(r.pt)] + [int(getattr(r, src)) for _, src in stats] + [int(r.week)]
                      for r in grp.itertuples()]

    dedupe_names(seasons)
    # Align game lists with season list order, keyed by the (deduped) name
    game_list = []
    for p in seasons:
        rows = games.pop(p["id"], [])
        game_list.append({"n": p["n"], "g": rows})
        del p["id"]
    return seasons, game_list


def finalize_games(game_list):
    """Sort each player's games by (season#, order) and drop the order column."""
    for p in game_list:
        p["g"] = [r[:-1] for r in sorted(p["g"], key=lambda r: (r[0], r[-1]))]


def dedupe_names(lst):
    """Two players with one name (e.g. Adrian Peterson) -> add rookie year: 'Adrian Peterson (2007)'."""
    from collections import Counter
    counts = Counter(p["n"] for p in lst)
    for p in lst:
        if counts[p["n"]] > 1:
            p["n"] = f'{p["n"]} ({p["s"]})'


# ---------------------------------------------------------------------------
# 4. Hand-entered seasons before 1999 (NFL.com, Pro Football Reference, StatMuse)
#    Rushing figures and a few playoff per-game splits are approximate.
# ---------------------------------------------------------------------------
def season(k, g, ptd, pyd, rtd, ryd, post=0):
    """One row: season #k, games, pass TD, pass yds, rush TD, rush yds; post 0=reg, 1=playoff, 2=wild card."""
    return [k, post, g, ptd, ptd + rtd, pyd, pyd + ryd]


def add_hand_entered(players, game_list):
    def P(name):
        return next(p for p in players if p["n"] == name)

    def spread(name, rows):
        """Per-game rows for hand seasons: each season's totals split evenly over its games
        (approximate). Regular-season games come first, then wild card, then later rounds."""
        G = next(p for p in game_list if p["n"] == name)["g"]
        for k, pt, g, *vals in rows:
            order0 = {0: 0, 2: 100, 1: 200}[pt]
            # Whole numbers that add back exactly: each game gets v // g, and the
            # remainder is handed out one per game from the first game on.
            split = [[v // g + (1 if i < v - (v // g) * g else 0) for i in range(g)] for v in vals]
            for i in range(g):
                G.append([k, pt] + [col[i] for col in split] + [order0 + i])

    # Rookie (1998) seasons that predate the nflverse data
    manning98, warner98 = [[1, 0, 16, 26, 26, 3739, 3739 + 62]], [[1, 0, 1, 0, 0, 39, 39]]
    P("Peyton Manning")["r"] += manning98; spread("Peyton Manning", manning98)
    P("Kurt Warner")["r"] += warner98;     spread("Kurt Warner", warner98)

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

    P("Dan Marino")["r"] += marino;  P("Dan Marino")["approx"] = 1;  spread("Dan Marino", marino)
    P("Brett Favre")["r"] += favre;  P("Brett Favre")["approx"] = 1;  spread("Brett Favre", favre)
    players.append({"n": "Jim Kelly", "s": 1986, "r": kelly, "approx": 1})
    game_list.append({"n": "Jim Kelly", "g": []}); spread("Jim Kelly", kelly)
    finalize_games(game_list)


if __name__ == "__main__":
    build()
