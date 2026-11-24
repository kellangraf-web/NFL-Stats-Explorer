"""One-time: turn the Kaggle 'NFL football player stats' download (zynicide/nfl-football-player-stats,
scraped from Pro Football Reference in Dec 2017) into data/kaggle_games_1970_1998.parquet.

Usage:  python scripts/make_kaggle_snapshot.py path/to/archive.zip
Keeps offensive game rows for 1970-1998 and each player's first season in the data (to detect
careers that began before 1970). Playoff games are those numbered past that year's regular season.
"""
import io, sys, zipfile
from pathlib import Path
import ijson
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "kaggle_games_1970_1998.parquet"
STATS = {"pass_yds": "passing_yards", "pass_td": "passing_touchdowns", "rush_att": "rushing_attempts",
         "rush_yds": "rushing_yards", "rush_td": "rushing_touchdowns", "rec": "receiving_receptions",
         "rec_yds": "receiving_yards", "rec_td": "receiving_touchdowns"}


def reg_len(y):
    """Regular-season games per team that year (strike years 1982, 1987)."""
    return {1982: 9, 1987: 15}.get(y, 14 if y <= 1977 else 16)


with zipfile.ZipFile(sys.argv[1]) as z:
    games_name = next(n for n in z.namelist() if n.startswith("games"))
    prof_name = next(n for n in z.namelist() if n.startswith("profiles"))
    prof = pd.read_json(io.BytesIO(z.read(prof_name)))
    rows, first = [], {}
    with z.open(games_name) as f:
        for g in ijson.items(f, "item", use_float=True):
            y, pid = int(g["year"]), g["player_id"]
            first[pid] = min(first.get(pid, 9999), y)          # first season with a game (any year)
            if 1970 <= y <= 1998:
                rows.append([pid, y, g["date"], int(g["game_number"]), g["team"]] +
                            [int(g.get(src) or 0) for src in STATS.values()])

df = pd.DataFrame(rows, columns=["pid", "season", "date", "gnum", "team"] + list(STATS))
# Names: some contain non-breaking spaces
prof["name"] = prof["name"].astype(str).str.replace("\xa0", " ").str.strip()
df = df.merge(prof[["player_id", "name", "position"]].rename(columns={"player_id": "pid", "position": "pos"}), on="pid")
df["first_season"] = df.pid.map(first)

# Game type: 0 regular season, 1 playoff, 2 wild-card playoff (first playoff weekend, 1978 on)
post = df.gnum > df.season.map(reg_len)
first_post = df[post].groupby("season").date.min()
days = (pd.to_datetime(df.date) - pd.to_datetime(df.season.map(first_post))).dt.days
df["pt"] = 0
df.loc[post, "pt"] = 1
df.loc[post & (df.season >= 1978) & (days < 6), "pt"] = 2

# Keep only players with some offense in 1970-1998 (drops linemen / defense to keep the file small)
off = df.groupby("pid")[["pass_yds", "rush_att", "rec"]].apply(lambda t: (t.abs().sum(axis=1) > 0).any())
df = df[df.pid.isin(off[off].index)].sort_values(["pid", "season", "gnum"])
df.to_parquet(OUT, index=False)
print(f"{len(df):,} game rows, {df.pid.nunique():,} players -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
