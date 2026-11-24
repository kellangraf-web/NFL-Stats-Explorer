"""
Builds players.json for the QB First-N-Seasons Explorer.

Downloads nflverse game-level player stats (1999 -> current season), splits every
player's career into season 1, 2, 3, ... (counted from his rookie year), and saves
games / TDs / yards per season, separated into regular season, playoffs and
wild-card playoff games. Games from 1970-1998 come from a saved snapshot of the Kaggle
"NFL football player stats" game logs (data/), cross-checked against fantasydatapros.

Run:  python build_data.py        (writes players.json next to this file)
"""
import datetime as dt
import io
import json
import re
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
    cols = ["player_id", "player_display_name", "position", "team", "season", "week", "season_type",
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
    num = wk.columns.drop(["player_id", "player_display_name", "position", "team", "season_type"])
    wk[num] = wk[num].fillna(0)            # missing stat = 0
    wk["position"] = wk["position"].fillna("")
    wk["gcount"] = 1                       # each nflverse row is one game
    wk["src"] = "nv"
    return wk


# ---------------------------------------------------------------------------
# 3. Games before 1999
#    Main source: Kaggle "NFL football player stats" (game logs scraped from Pro Football
#    Reference, incl. playoffs and the AFL 1960-69) -> data/kaggle_games_1950_1998.parquet
#    (scripts/make_kaggle_snapshot.py).
#    Cross-check: fantasydatapros regular-season totals -> data/fdp_regular_season_1970_2000.csv,
#    also used to fill three games missing from nflverse in 1999-2000.
# ---------------------------------------------------------------------------
DATA = Path(__file__).with_name("data")
KAGGLE = DATA / "kaggle_games_1950_1998.parquet"
HIST = DATA / "fdp_regular_season_1970_2000.csv"
STAT_MAP = {  # fantasydatapros column -> nflverse column
    "pass_yds": "passing_yards", "pass_td": "passing_tds", "rush_att": "carries",
    "rush_yds": "rushing_yards", "rush_td": "rushing_tds", "rec": "receptions",
    "rec_yds": "receiving_yards", "rec_td": "receiving_tds"}

# Regular-season games missing from nflverse (found by checking every team has 16 weeks)
MISSING_GAMES = [(1999, 1, {"LA", "BAL"}),     # Rams vs Ravens
                 (2000, 3, {"KC", "LAC"}),     # Chiefs vs Chargers
                 (2000, 6, {"BUF", "MIA"})]    # Bills vs Dolphins


def norm(name):
    """Name key for matching across sources: lowercase, no punctuation or Jr./Sr./II/III."""
    n = re.sub(r"[^a-z ]", "", str(name).lower())
    return " ".join(re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n).split())


def load_hist():
    h = pd.read_csv(HIST, dtype={"pos": str})
    h["key"] = h.player.map(norm)
    return h


def patch_missing_games(wk, hist):
    """Add each missing game as one extra row per player: fdp season total minus nflverse total."""
    add = []
    for season, week, teams in MISSING_GAMES:
        f = hist[hist.season == season].groupby("key")[list(STAT_MAP)].sum()
        reg = wk[(wk.season == season) & (wk.season_type == "REG")]
        on_team = reg[reg.team.isin(teams)].player_id.unique()
        tot = reg[reg.player_id.isin(on_team)].groupby(["player_id", "player_display_name", "position"])[
            list(STAT_MAP.values())].sum().reset_index()
        for r in tot.itertuples():
            k = norm(r.player_display_name)
            if k not in f.index:
                continue
            diff = {nv: int(f.loc[k, fd]) - int(getattr(r, nv)) for fd, nv in STAT_MAP.items()}
            # A real missing game shows up as extra attempts/catches or clearly more yards;
            # keep the raw differences (a lost-yardage run can make one of them negative)
            plays = diff["carries"] + diff["receptions"]
            yards = diff["passing_yards"] + diff["rushing_yards"] + diff["receiving_yards"]
            if plays > 0 or yards >= 10 or diff["passing_tds"] > 0:
                add.append({"player_id": r.player_id, "player_display_name": r.player_display_name,
                            "position": r.position, "team": "", "season": season, "week": week,
                            "season_type": "REG", "passing_tds": diff["passing_tds"], "rushing_tds": diff["rushing_tds"],
                            "receiving_tds": diff["receiving_tds"], "passing_yards": diff["passing_yards"],
                            "rushing_yards": diff["rushing_yards"], "receiving_yards": diff["receiving_yards"],
                            "carries": diff["carries"], "receptions": diff["receptions"], "gcount": 1, "src": "nv"})
    print(f"  patched {len(add)} player-games missing from nflverse (1999-2000)")
    return pd.concat([wk, pd.DataFrame(add)], ignore_index=True)


def load_kaggle():
    """Kaggle 1970-1998 game rows + one career record per Kaggle player id."""
    k = pd.read_parquet(KAGGLE)
    k["key"] = k.name.map(norm)
    # Positions: old labels HB / B mean running back
    k["pos"] = k.pos.replace({"HB": "RB", "B": "RB"})
    car = k.groupby("pid").agg(name=("name", "first"), key=("key", "first"), pos=("pos", "first"),
                               first=("first_season", "first"), last=("season", "max"))
    # The data starts in 1950, so careers already underway then would be incomplete:
    # drop players drafted before 1950, or already 25+ at their first 1950 game
    pre = k.groupby("pid").agg(draft=("draft_year", "first"), age0=("first_age", "first"))
    started_before = (pre.draft < 1950) | ((car["first"] == 1950) & (pre.age0 >= 25))
    car = car[~started_before]
    return k[k.pid.isin(car.index)], car


def corroborate(k, hist):
    """Check every Kaggle regular season against fantasydatapros' season totals."""
    reg = k[k.pt == 0].groupby(["key", "season"])[list(STAT_MAP)].sum()
    f = hist[hist.season <= 1998].groupby(["key", "season"])[list(STAT_MAP)].sum()
    both = reg.index.intersection(f.index)
    exact = (reg.loc[both] == f.loc[both]).all(axis=1)
    print(f"  cross-check vs fantasydatapros: {exact.mean():.2%} of {len(both):,} player-seasons match exactly")
    if (~exact).any():
        print("  mismatches (first 5):", list(exact[~exact].index[:5]))


# ---------------------------------------------------------------------------
# 4. Build per-player, per-season rows
# ---------------------------------------------------------------------------
def build():
    cur = current_season()
    print(f"Current season: {cur}")
    wk = load_weekly(cur)
    hist = load_hist()
    wk = patch_missing_games(wk, hist)

    # --- nflverse career start = earliest of rookie season, draft year, first game in data ---
    pl = fetch_parquet(f"{BASE}/players/players.parquet")[["gsis_id", "rookie_season", "draft_year"]]
    first_seen = wk.groupby("player_id").season.min().rename("first_seen")
    wk = wk.merge(pl, left_on="player_id", right_on="gsis_id", how="left").merge(first_seen, on="player_id")
    wk["start"] = wk[["rookie_season", "draft_year", "first_seen"]].min(axis=1)

    # --- Games before 1999 (Kaggle), checked against fantasydatapros ---
    k, car = load_kaggle()
    corroborate(k, hist)

    # --- Link pre-1999 careers to the same player's 1999+ nflverse seasons ---
    nv_car = wk.groupby("player_id").agg(name=("player_display_name", "first"), start=("start", "min"),
                                         first_seen=("first_seen", "first"))
    nv_car["key"] = nv_car.name.map(norm)
    # Candidates: nflverse players already playing in 1999-2000 whose careers began before 1999
    cand = nv_car[(nv_car.first_seen <= 2000) & (nv_car.start < 1999)]
    link = {}                                   # Kaggle id -> nflverse player_id
    for pid, c in car[car["last"] >= 1997].iterrows():
        m = cand[cand.key == c.key]
        if len(m) == 1:                         # only an unambiguous name match
            link[pid] = m.index[0]
    print(f"  pre-1999 careers: {len(car):,}; linked to their 1999+ seasons: {len(link):,}")

    # Linked players: their nflverse career starts at their first Kaggle season
    start_of = {link[pid]: int(car.loc[pid, "first"]) for pid in link}
    wk["start"] = wk.player_id.map(start_of).fillna(wk.start)
    # Other pre-1999 nflverse careers can't be completed, so leave them out
    wk = wk[(wk.start >= 1999) | wk.player_id.isin(start_of.keys())]

    # --- Kaggle games in the same layout as nflverse rows (one row = one game) ---
    kr = k.copy()
    kr["player_id"] = kr.pid.map(link).fillna("K:" + kr.pid.astype(str))
    names = wk.groupby("player_id").player_display_name.first()
    kr["player_display_name"] = kr.player_id.map(names).fillna(kr.name)
    kr["position"] = kr.pid.map(car.pos)
    kr["start"] = kr.pid.map(car["first"])
    kr = kr.rename(columns=STAT_MAP).assign(week=kr.gnum, gcount=1, src="kag",
                                            season_type=kr.pt.map(lambda t: "REG" if t == 0 else "POST"))
    kr = kr[[c for c in list(wk.columns) + ["pt"] if c in kr.columns]]

    rows = pd.concat([wk, kr], ignore_index=True)
    rows["start"] = rows.groupby("player_id").start.transform("min")
    for c in ["passing_tds", "rushing_tds", "receiving_tds", "passing_yards", "rushing_yards",
              "receiving_yards", "carries", "receptions"]:
        rows[c] = rows[c].fillna(0)

    # --- Season index within the career (1 = rookie year) ---
    rows["k"] = (rows.season - rows.start + 1).astype(int)
    rows = rows[rows.k <= MAXK]

    # --- Game type: 0 = regular season, 1 = playoff, 2 = wild-card playoff ---
    wc_week = rows[(rows.season_type == "POST") & (rows.src == "nv")].groupby("season").week.min()
    nvpost = (rows.season_type == "POST") & (rows.src == "nv")
    rows.loc[(rows.src == "nv") & (rows.season_type == "REG"), "pt"] = 0
    rows.loc[nvpost, "pt"] = 1
    rows.loc[nvpost & (rows.week == rows.season.map(wc_week)), "pt"] = 2
    rows["pt"] = rows.pt.astype(int)

    # --- Totals ---
    rows["ttd"] = rows.passing_tds + rows.rushing_tds + rows.receiving_tds
    rows["tyd"] = rows.passing_yards + rows.rushing_yards + rows.receiving_yards
    rows["syd"] = rows.rushing_yards + rows.receiving_yards

    # Primary position: Kaggle's for players with pre-1999 games, else nflverse's most listed
    has_k = set(rows[rows.src == "kag"].player_id)
    kpos = rows[rows.src == "kag"].groupby("player_id").position.first()
    npos = rows[(rows.src == "nv") & (rows.position.fillna("") != "")].groupby("player_id").position \
        .agg(lambda x: x.mode().iat[0])
    pos = pd.concat([npos[~npos.index.isin(has_k)], kpos])
    ctx = {}
    by_pos = lambda p: rows.player_id.map(pos).eq(p)
    qb, qb_games = build_group(rows, rows.index == rows.index,
        [("ptd", "passing_tds"), ("ttd", "ttd"), ("pyd", "passing_yards"), ("tyd", "tyd")],
        keep_col="tyd", total_min=5000, season_min=1500, **ctx)
    rb, rb_games = build_group(rows, by_pos("RB"),
        [("att", "carries"), ("ryd", "rushing_yards"), ("rtd", "rushing_tds"),
         ("recyd", "receiving_yards"), ("rectd", "receiving_tds")],
        keep_col="syd", total_min=3000, season_min=800, **ctx)
    wr, wr_games = build_group(rows, by_pos("WR"),
        [("rec", "receptions"), ("recyd", "receiving_yards"), ("rectd", "receiving_tds")],
        keep_col="receiving_yards", total_min=3000, season_min=800, **ctx)
    # Tight ends: same receiving stats as WRs, lower size filter (TEs catch fewer yards)
    te, te_games = build_group(rows, by_pos("TE"),
        [("rec", "receptions"), ("recyd", "receiving_yards"), ("rectd", "receiving_tds")],
        keep_col="receiving_yards", total_min=2000, season_min=500, **ctx)

    # --- Label for "data through": last week with games in the current season ---
    cur_rows = wk[wk.season == cur]
    if len(cur_rows):
        last = cur_rows.sort_values(["week"]).iloc[-1]
        through = f"Week {int(last.week)}" if last.season_type == "REG" else "the playoffs"
    else:
        through = "the start of the season"

    out = {"meta": {"season": cur, "through": through}, "players": qb, "rbs": rb, "wrs": wr, "tes": te}
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    games = {"players": qb_games, "rbs": rb_games, "wrs": wr_games, "tes": te_games}
    OUT_GAMES.write_text(json.dumps(games, separators=(",", ":")))
    print(f"Wrote {OUT.name}: {len(qb)} QB-list, {len(rb)} RBs, {len(wr)} WRs, {len(te)} TEs; "
          f"{OUT_GAMES.name}; {cur} data through {through}")


def build_group(rows, mask, stats, keep_col, total_min, season_min):
    """Per-season rows and per-game rows for one group of players.

    Season row: [season#, gameType, G, stat1, stat2, ...]
    Game row:   [season#, gameType, stat1, stat2, ...]   (chronological)
    gameType: 0 = regular season, 1 = playoff, 2 = wild-card playoff.
    """
    w = rows[mask]
    # Keep players who could plausibly chart: enough total yards, or one big season
    tot = w.groupby("player_id")[keep_col].sum()
    best = w.groupby(["player_id", "k"])[keep_col].sum().groupby("player_id").max()
    keep = set(tot[tot >= total_min].index) | set(best[best >= season_min].index)
    w = w[w.player_id.isin(keep)]
    names = [n for n, _ in stats]

    # Season totals (every row is one game)
    agg = w.groupby(["player_id", "k", "pt"]).agg(
        g=("gcount", "sum"), **{n: (src, "sum") for n, src in stats}).reset_index()
    info = w.groupby("player_id").agg(name=("player_display_name", "first"), start=("start", "min"))
    seasons = []
    for pid, grp in agg.groupby("player_id"):
        rec = {"id": pid, "n": info.loc[pid, "name"], "s": int(info.loc[pid, "start"]),
               "r": [[int(r.k), int(r.pt), int(r.g)] + [int(getattr(r, n)) for n in names]
                     for r in grp.itertuples()]}
        seasons.append(rec)

    # Per-game rows, oldest first (every row is one game: nflverse or Kaggle)
    games = {}
    for pid, grp in w.sort_values(["season", "week"]).groupby("player_id", sort=False):
        games[pid] = [[int(r.k), int(r.pt)] + [int(getattr(r, src)) for _, src in stats] for r in grp.itertuples()]

    dedupe_names(seasons)
    game_list = []
    for p in seasons:
        game_list.append({"n": p["n"], "g": games.pop(p["id"], [])})
        del p["id"]
    return seasons, game_list


def dedupe_names(lst):
    """Two players with one name (e.g. Adrian Peterson) -> add rookie year: 'Adrian Peterson (2007)'."""
    from collections import Counter
    counts = Counter(p["n"] for p in lst)
    for p in lst:
        if counts[p["n"]] > 1:
            p["n"] = f'{p["n"]} ({p["s"]})'


if __name__ == "__main__":
    build()
