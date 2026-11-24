"""One-time: turn the Kaggle 'NFL football player stats' download (zynicide/nfl-football-player-stats,
scraped from Pro Football Reference in Dec 2017) into data/kaggle_games_1950_1998.parquet.

Usage:  python scripts/make_kaggle_snapshot.py path/to/archive.zip
Keeps offensive game rows for 1950-1998 (NFL, and AFL 1960-69), plus each player's draft year and age
at his first game (to drop careers that began before 1950). Playoff games are those numbered past that
team's regular-season length; the 1960s Playoff Bowl exhibitions aren't in the data.
"""
import io, sys, zipfile
from pathlib import Path
import ijson
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "kaggle_games_1950_1998.parquet"
FIRST, LAST = 1950, 1998
AFL_1960 = {"NYT", "DEN", "DTX", "BUF", "BOS", "OAK", "HOU", "LAC"}   # AFL teams in its first season
STATS = {"pass_yds": "passing_yards", "pass_td": "passing_touchdowns", "rush_att": "rushing_attempts",
         "rush_yds": "rushing_yards", "rush_td": "rushing_touchdowns", "rec": "receiving_receptions",
         "rec_yds": "receiving_yards", "rec_td": "receiving_touchdowns"}


def reg_len(y, team):
    """Regular-season games for a team that year: NFL 12 through 1960, AFL 14 from 1960,
    both 14 from 1961, 16 from 1978; strike years 1982 (9) and 1987 (15)."""
    if y < 1960: return 12
    if y == 1960: return 14 if team in AFL_1960 else 12
    return {1982: 9, 1987: 15}.get(y, 14 if y <= 1977 else 16)


with zipfile.ZipFile(sys.argv[1]) as z:
    games_name = next(n for n in z.namelist() if n.startswith("games"))
    prof_name = next(n for n in z.namelist() if n.startswith("profiles"))
    prof = pd.read_json(io.BytesIO(z.read(prof_name)))
    rows, first, first_age = [], {}, {}
    with z.open(games_name) as f:
        for g in ijson.items(f, "item", use_float=True):
            y, pid = int(g["year"]), g["player_id"]
            if y < first.get(pid, 9999):                       # first season with a game, and age then
                first[pid] = y
                first_age[pid] = int(str(g.get("age") or "0-0").split("-")[0] or 0)
            if FIRST <= y <= LAST:
                rows.append([pid, y, g["date"], int(g["game_number"]), g["team"]] +
                            [int(g.get(src) or 0) for src in STATS.values()])

df = pd.DataFrame(rows, columns=["pid", "season", "date", "gnum", "team"] + list(STATS))
# Names: some contain non-breaking spaces
prof["name"] = prof["name"].astype(str).str.replace("\xa0", " ").str.strip()
df = df.merge(prof[["player_id", "name", "position", "draft_year"]].rename(
    columns={"player_id": "pid", "position": "pos"}), on="pid")
df["draft_year"] = pd.to_numeric(df.draft_year, errors="coerce")
df["first_season"] = df.pid.map(first)
df["first_age"] = df.pid.map(first_age)

# Game type: 0 regular season, 1 playoff, 2 wild-card playoff (first playoff weekend, 1978 on)
post = df.gnum > [reg_len(y, t) for y, t in zip(df.season, df.team)]
first_post = df[post].groupby("season").date.min()
days = (pd.to_datetime(df.date) - pd.to_datetime(df.season.map(first_post))).dt.days
df["pt"] = 0
df.loc[post, "pt"] = 1
df.loc[post & (df.season >= 1978) & (days < 6), "pt"] = 2

# Keep only players with some offense in 1950-1998 (drops linemen / defense to keep the file small)
off = df.groupby("pid")[["pass_yds", "rush_att", "rec"]].apply(lambda t: (t.abs().sum(axis=1) > 0).any())
df = df[df.pid.isin(off[off].index)].sort_values(["pid", "season", "gnum"])
df.to_parquet(OUT, index=False)
print(f"{len(df):,} game rows, {df.pid.nunique():,} players -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
