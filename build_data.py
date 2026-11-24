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
# 3. Seasons before 1999: fantasydatapros regular-season totals (1970-2000)
#    Snapshot in data/; verified against nflverse (1999-2021, ~99% exact) and NFL.com.
# ---------------------------------------------------------------------------
HIST = Path(__file__).with_name("data") / "fdp_regular_season_1970_2000.csv"
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


def hist_players(hist):
    """One career per (name key, unbroken run of seasons), 1970-1998, with first season and position."""
    h = hist[hist.season <= 1998].sort_values("season").copy()
    # A gap of 3+ seasons under one name = a different player with the same name
    h["run"] = h.groupby("key").season.transform(lambda s: (s.diff() > 3).cumsum())
    h["hid"] = h.key + "#" + h.run.astype(str)
    # Position: most-listed real position (some seasons have it blank / "0")
    real_pos = lambda s: (s[~s.isin(["0", ""]) & s.notna()].mode().iat[0]
                          if (~s.isin(["0", ""]) & s.notna()).any() else "")
    car = h.groupby("hid").agg(name=("player", "first"), key=("key", "first"), first=("season", "min"),
                               last=("season", "max"), age0=("age", "first"), pos=("pos", real_pos))
    # Careers already underway in 1970 would be incomplete: drop 1970 players aged 25+
    car = car[~((car["first"] == 1970) & (car.age0 >= 25))]
    return h[h.hid.isin(car.index)], car


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

    # --- Link pre-1999 careers to the same player's 1999+ nflverse seasons ---
    h, car = hist_players(hist)
    nv_car = wk.groupby("player_id").agg(name=("player_display_name", "first"), start=("start", "min"),
                                         first_seen=("first_seen", "first"))
    nv_car["key"] = nv_car.name.map(norm)
    # Candidates: nflverse players already playing in 1999-2000 whose careers began before 1999
    cand = nv_car[(nv_car.first_seen <= 2000) & (nv_car.start < 1999)]
    link = {}                                   # hist id -> nflverse player_id
    for hid, c in car[car["last"] >= 1997].iterrows():
        m = cand[cand.key == c.key]
        if len(m) == 1:
            link[hid] = m.index[0]
    print(f"  pre-1999 careers: {len(car):,}; linked to their 1999+ seasons: {len(link):,}")

    # Linked players: their nflverse career starts at the first fdp season
    start_of = {link[hid]: int(car.loc[hid, "first"]) for hid in link}
    wk["start"] = wk.player_id.map(start_of).fillna(wk.start)
    # Other pre-1999 nflverse careers can't be completed, so leave them out
    wk = wk[(wk.start >= 1999) | wk.player_id.isin(start_of.keys())]

    # --- fdp seasons as rows in the same layout (one row = one season, gcount = games) ---
    hr = h.copy()
    hr["player_id"] = hr.hid.map(link).fillna("H:" + hr.hid)
    names = wk.groupby("player_id").player_display_name.first()
    hr["player_display_name"] = hr.player_id.map(names).fillna(hr.player)
    hr["position"] = hr.hid.map(car["pos"])           # career position (blank seasons ignored)
    hr["start"] = hr.hid.map(car["first"])
    hr = hr.rename(columns={k: v for k, v in STAT_MAP.items()}).assign(
        week=0, season_type="REG", gcount=hr.g, src="hist", team=hr.team)
    hr = hr[[c for c in wk.columns if c in hr.columns]]

    # --- Hand-entered playoff games for pre-1999 seasons (fdp has no playoffs) ---
    hand = hand_playoffs()
    # Attach each hand row to that player's id (Kelly's comes from his fdp seasons)
    ids = pd.concat([hr[["player_display_name", "player_id"]], wk[["player_display_name", "player_id"]]]) \
        .drop_duplicates("player_display_name").set_index("player_display_name").player_id
    hand["player_id"] = hand.player_display_name.map(ids)
    rows = pd.concat([wk, hr, hand], ignore_index=True)
    # Players whose pre-1999 seasons have complete playoff data (or no playoff games then)
    COMPLETE = {"Dan Marino", "Brett Favre", "Jim Kelly", "Peyton Manning", "Kurt Warner"}
    hand_ids = {pid for pid, nm in rows.groupby("player_id").player_display_name.first().items() if nm in COMPLETE}
    rows["start"] = rows.groupby("player_id").start.transform("min")
    for c in ["passing_tds", "rushing_tds", "receiving_tds", "passing_yards", "rushing_yards",
              "receiving_yards", "carries", "receptions"]:
        rows[c] = rows[c].fillna(0)

    # --- Season index within the career (1 = rookie year) ---
    rows["k"] = (rows.season - rows.start + 1).astype(int)
    rows = rows[rows.k <= MAXK]

    # --- Game type: 0 = regular season, 1 = playoff, 2 = wild-card playoff ---
    wc_week = rows[(rows.season_type == "POST") & (rows.src == "nv")].groupby("season").week.min()
    if "pt" not in rows: rows["pt"] = None
    nvpost = (rows.season_type == "POST") & (rows.src == "nv")
    rows.loc[rows.season_type == "REG", "pt"] = 0
    rows.loc[nvpost, "pt"] = 1
    rows.loc[nvpost & (rows.week == rows.season.map(wc_week)), "pt"] = 2
    rows["pt"] = rows.pt.astype(int)

    # --- Totals ---
    rows["ttd"] = rows.passing_tds + rows.rushing_tds + rows.receiving_tds
    rows["tyd"] = rows.passing_yards + rows.rushing_yards + rows.receiving_yards
    rows["syd"] = rows.rushing_yards + rows.receiving_yards

    # Regular-season-only players: have pre-1999 seasons without playoff data
    ro_ids = set(rows[rows.src == "hist"].player_id) - hand_ids
    # Primary position: fdp's for players with pre-1999 seasons, else nflverse's most listed
    pos = rows[rows.position.fillna("") != ""].assign(h=rows.src.eq("hist")).sort_values("h", ascending=False) \
        .groupby("player_id").apply(lambda g: g[g.h].position.mode().iat[0] if g.h.any() else g.position.mode().iat[0])

    ctx = dict(ro_ids=ro_ids, approx_ids={pid for pid in hand_ids
              if rows[rows.player_id == pid].player_display_name.iat[0] in {"Dan Marino", "Brett Favre", "Jim Kelly"}})
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
    n_ro = sum(1 for lst in (qb, rb, wr, te) for p in lst if p.get("ro"))
    print(f"Wrote {OUT.name}: {len(qb)} QB-list, {len(rb)} RBs, {len(wr)} WRs, {len(te)} TEs "
          f"({n_ro} list entries regular-season only); {OUT_GAMES.name}; {cur} data through {through}")


def build_group(rows, mask, stats, keep_col, total_min, season_min, ro_ids, approx_ids):
    """Per-season rows and per-game rows for one group of players.

    Season row: [season#, gameType, G, stat1, stat2, ...]
    Game row:   [season#, gameType, stat1, stat2, ...]   (chronological)
    gameType: 0 = regular season, 1 = playoff, 2 = wild-card playoff.
    ro = only regular-season data before 1999 (the page shows these players only with
    Playoffs: Excluded and the slider in Seasons).
    """
    w = rows[mask]
    # Keep players who could plausibly chart: enough total yards, or one big season
    tot = w.groupby("player_id")[keep_col].sum()
    best = w.groupby(["player_id", "k"])[keep_col].sum().groupby("player_id").max()
    keep = set(tot[tot >= total_min].index) | set(best[best >= season_min].index)
    w = w[w.player_id.isin(keep)]
    names = [n for n, _ in stats]

    # Season totals (games = sum of gcount: 1 per nflverse game, G per fdp/hand season row)
    agg = w.groupby(["player_id", "k", "pt"]).agg(
        g=("gcount", "sum"), **{n: (src, "sum") for n, src in stats}).reset_index()
    info = w.groupby("player_id").agg(name=("player_display_name", "first"), start=("start", "min"))
    seasons = []
    for pid, grp in agg.groupby("player_id"):
        rec = {"id": pid, "n": info.loc[pid, "name"], "s": int(info.loc[pid, "start"]),
               "r": [[int(r.k), int(r.pt), int(r.g)] + [int(getattr(r, n)) for n in names]
                     for r in grp.itertuples()]}
        if pid in ro_ids: rec["ro"] = 1
        if pid in approx_ids: rec["approx"] = 1
        seasons.append(rec)

    # Per-game rows, oldest first. nflverse rows are single games; season rows for the
    # complete pre-1999 players are spread evenly across their games (approximate).
    games = {}
    for pid, grp in w.sort_values(["season", "week"]).groupby("player_id", sort=False):
        if pid in ro_ids:
            games[pid] = []                     # no per-game data: hidden in Games mode
            continue
        out = []
        for r in grp.itertuples():
            vals = [int(getattr(r, src)) for _, src in stats]
            if r.src == "nv":
                out.append([int(r.k), int(r.pt)] + vals + [int(r.week)])
            else:
                order0 = {0: 0, 2: 100, 1: 200}[int(r.pt)]
                g = int(r.gcount)
                # Whole numbers that add back exactly: v // g each, remainder one per game
                split = [[v // g + (1 if i < v - (v // g) * g else 0) for i in range(g)] for v in vals]
                out += [[int(r.k), int(r.pt)] + [col[i] for col in split] + [order0 + i] for i in range(g)]
        games[pid] = [x[:-1] for x in sorted(out, key=lambda x: (x[0], x[-1]))]

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


# ---------------------------------------------------------------------------
# 5. Hand-entered playoff games before 1999 (Pro Football Reference, StatMuse)
#    fdp has no playoffs. A few per-round splits are approximate.
# ---------------------------------------------------------------------------
def hand_playoffs():
    """Rows: (player, calendar season, gameType, games, passTD, passYds). 1 = playoff, 2 = wild card."""
    P = [  # Marino
         ("Dan Marino", 1983, 1, 1, 2, 193), ("Dan Marino", 1984, 1, 3, 8, 1001), ("Dan Marino", 1985, 1, 2, 2, 486),
         ("Dan Marino", 1990, 2, 1, 2, 221), ("Dan Marino", 1990, 1, 1, 3, 323), ("Dan Marino", 1992, 1, 2, 4, 435),
         ("Dan Marino", 1994, 2, 1, 2, 257), ("Dan Marino", 1994, 1, 1, 4, 262), ("Dan Marino", 1995, 2, 1, 2, 422),
         ("Dan Marino", 1997, 2, 1, 0, 141), ("Dan Marino", 1998, 2, 1, 1, 235), ("Dan Marino", 1998, 1, 1, 0, 243),
         # Favre
         ("Brett Favre", 1993, 2, 1, 3, 204), ("Brett Favre", 1993, 1, 1, 2, 331), ("Brett Favre", 1994, 2, 1, 0, 262),
         ("Brett Favre", 1994, 1, 1, 0, 211), ("Brett Favre", 1995, 2, 1, 3, 199), ("Brett Favre", 1995, 1, 2, 5, 606),
         ("Brett Favre", 1996, 1, 3, 5, 617), ("Brett Favre", 1997, 1, 3, 5, 668), ("Brett Favre", 1998, 2, 1, 2, 292),
         # Kelly
         ("Jim Kelly", 1988, 1, 2, 1, 407), ("Jim Kelly", 1989, 1, 1, 4, 405), ("Jim Kelly", 1990, 1, 3, 5, 851),
         ("Jim Kelly", 1991, 1, 3, 5, 665), ("Jim Kelly", 1992, 1, 2, 1, 259), ("Jim Kelly", 1993, 1, 3, 2, 707),
         ("Jim Kelly", 1995, 2, 1, 1, 195), ("Jim Kelly", 1995, 1, 1, 1, 135), ("Jim Kelly", 1996, 2, 1, 1, 239)]
    df = pd.DataFrame(P, columns=["player", "season", "pt", "gcount", "passing_tds", "passing_yards"])
    df = df.assign(rushing_tds=0, receiving_tds=0, rushing_yards=0, receiving_yards=0, carries=0, receptions=0,
                   week=0, season_type="POST", src="hand", position="QB")
    df["player_display_name"] = df.player
    return df.drop(columns="player")


if __name__ == "__main__":
    build()
