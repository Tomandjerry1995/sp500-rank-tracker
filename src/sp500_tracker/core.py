# file: projects/sp500-rank-tracker/src/sp500_tracker/core.py
"""
Core logic: scrape, store (top-500, dropped=501), query, plot (static & interactive).

Why:
- Keep daily top 500 for signal density; mark known-but-missing as 501 to keep series continuous.
- Interactive HTML (Plotly) helps explore many series (hover names, toggle legend).
"""
import datetime as dt
import os
import re
import sqlite3
from typing import Iterable, List, Optional, Sequence, Tuple, Set, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Response
from requests.adapters import HTTPAdapter, Retry

SLICKCHARTS_URL = "https://www.slickcharts.com/sp500"


def ensure_dirs(base_dir: str) -> Tuple[str, str]:
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return data_dir, plots_dir


def utc_date() -> str:
    return dt.datetime.utcnow().date().isoformat()


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    ad = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=8)
    s.mount("http://", ad)
    s.mount("https://", ad)
    s.headers.update({"User-Agent": "sp500-rank-tracker/0.2"})
    return s


def _normalize_columns(cols: Sequence[str]) -> List[str]:
    return [re.sub(r"[^a-z0-9]+", "", str(c).strip().lower()) for c in cols]


def _resolve_columns(df: pd.DataFrame) -> Tuple[int, int, int]:
    norm = _normalize_columns(df.columns.astype(str).tolist())

    def find_one(cands: Iterable[str]) -> int:
        for i, n in enumerate(norm):
            if n in cands:
                return i
        for i, n in enumerate(norm):
            if any(c in n for c in cands):
                return i
        raise KeyError(f"Missing columns like {cands}, got {norm}")

    rank_idx = find_one({"", "rank"})  # "#" often becomes ""
    symbol_idx = find_one({"symbol", "ticker"})
    company_idx = find_one({"company", "name"})
    return rank_idx, symbol_idx, company_idx


def fetch_today_table() -> pd.DataFrame:
    """Scrape the table and return cleaned DataFrame with columns rank/symbol/company."""
    sess = make_session()
    resp: Response = sess.get(SLICKCHARTS_URL, timeout=20)
    if resp.status_code != 200 or not resp.text:
        raise RuntimeError(f"HTTP {resp.status_code} fetching slickcharts")

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if table is None:
        tables = pd.read_html(resp.text)
        if not tables:
            raise RuntimeError("No table found")
        df = tables[0]
    else:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr"):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if tds:
                rows.append(tds)
        max_len = max(len(r) for r in rows) if rows else 0
        rows = [r + [""] * (max_len - len(r)) for r in rows]
        df = pd.DataFrame(rows, columns=headers[:max_len])

    r_idx, s_idx, c_idx = _resolve_columns(df)
    cleaned = pd.DataFrame(
        {
            "rank": pd.to_numeric(df.iloc[:, r_idx], errors="coerce"),
            "symbol": df.iloc[:, s_idx].astype(str).str.strip().str.upper(),
            "company": df.iloc[:, c_idx].astype(str).str.strip(),
        }
    ).dropna(subset=["rank", "symbol", "company"])
    cleaned["rank"] = cleaned["rank"].astype(int)
    cleaned = cleaned.sort_values("rank").reset_index(drop=True)
    cleaned = cleaned[cleaned["symbol"].str.fullmatch(r"[A-Z\.]{1,8}") == True]
    return cleaned


# --- DB ---
def connect_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rank_history (
          date    TEXT NOT NULL,
          symbol  TEXT NOT NULL,
          company TEXT NOT NULL,
          rank    INTEGER NOT NULL,
          PRIMARY KEY (date, symbol)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON rank_history(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON rank_history(date)")
    return conn


def latest_company_name(conn: sqlite3.Connection, symbol: str) -> Optional[str]:
    cur = conn.execute(
        "SELECT company FROM rank_history WHERE symbol=? ORDER BY date DESC LIMIT 1",
        (symbol,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def known_symbols(conn: sqlite3.Connection) -> Set[str]:
    cur = conn.execute("SELECT DISTINCT symbol FROM rank_history")
    return {r[0] for r in cur.fetchall()}


def insert_rows_present(conn: sqlite3.Connection, date_str: str, df: pd.DataFrame) -> int:
    cur = conn.cursor()
    inserted = 0
    for _, r in df.iterrows():
        cur.execute(
            "INSERT OR IGNORE INTO rank_history(date, symbol, company, rank) VALUES (?, ?, ?, ?)",
            (date_str, r["symbol"], r["company"], int(r["rank"])),
        )
        if cur.rowcount > 0:
            inserted += 1
    conn.commit()
    return inserted


def insert_rows_dropped(
    conn: sqlite3.Connection, date_str: str, drops: Set[str], dropped_rank: int
) -> int:
    cur = conn.cursor()
    inserted = 0
    for sym in sorted(drops):
        name = latest_company_name(conn, sym) or sym  # why: human-readable even if new
        cur.execute(
            "INSERT OR IGNORE INTO rank_history(date, symbol, company, rank) VALUES (?, ?, ?, ?)",
            (date_str, sym, name, dropped_rank),
        )
        if cur.rowcount > 0:
            inserted += 1
    conn.commit()
    return inserted


def query_history(
    conn: sqlite3.Connection,
    symbols: Optional[List[str]] = None,
    company_substr: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    top: Optional[int] = None,
) -> pd.DataFrame:
    params: List[object] = []
    where: List[str] = []

    if since:
        where.append("date >= ?"); params.append(since)
    if until:
        where.append("date <= ?"); params.append(until)

    # '=' for single, IN(...) for multiple
    if symbols:
        syms = [s.upper().strip() for s in symbols if s.strip()]
        if len(syms) == 1:
            where.append("symbol = ?")
            params.append(syms[0])
        else:
            qs = ",".join("?" for _ in syms)
            where.append(f"symbol IN ({qs})")
            params.extend(syms)

    if company_substr:
        where.append("LOWER(company) LIKE ?"); params.append(f"%{company_substr.lower()}%")

    base_sql = "SELECT * FROM rank_history"
    if top:
        sub_where = " AND ".join(where) if where else "1=1"
        sql = f"{base_sql} WHERE {sub_where} AND rank <= ?"
        bind = tuple(params + [top])
    else:
        sql = base_sql + (" WHERE " + " AND ".join(where) if where else "")
        bind = tuple(params)

    df = pd.read_sql_query(sql, conn, params=bind)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["rank"] = df["rank"].astype(int)
        df["symbol"] = df["symbol"].astype(str)
        df["company"] = df["company"].astype(str)
    return df


# --- Plotting (matplotlib static) ---
def plot_rank(df: pd.DataFrame, outfile: str, title: str, dropped_rank: int) -> str:
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("No data to plot")

    df = df.copy()
    df.loc[df["rank"] > dropped_rank, "rank"] = dropped_rank

    pivot = (
        df.pivot_table(index="date", columns="symbol", values="rank", aggfunc="first")
        .sort_index()
    )

    ax = pivot.plot(figsize=(10, 6), marker="o")
    ax.set_title(title)
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Rank (lower is better)")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.4)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    import matplotlib
    # why: ensure non-GUI save works on servers
    # (matplotlib will pick Agg backend automatically in most CLIs)
    import matplotlib.pyplot as plt  # re-import safe
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    return outfile


# --- Plotting (plotly interactive) ---
def _select_symbols_for_plot(df: pd.DataFrame, limit: int) -> List[str]:
    """Pick up to `limit` symbols by best (min) rank across the period; stable ordering."""
    agg = df.groupby("symbol")["rank"].min().sort_values().reset_index()
    return agg.head(limit)["symbol"].tolist()


def plot_interactive(
    df: pd.DataFrame,
    outfile_html: str,
    title: str,
    dropped_rank: int,
    max_lines: int,
) -> str:
    """Interactive HTML with hover (symbol/company/date/rank)."""
    import plotly.graph_objects as go

    if df.empty:
        raise ValueError("No data to plot")

    df = df.copy()
    df.loc[df["rank"] > dropped_rank, "rank"] = dropped_rank

    # choose symbols to draw
    symbols = _select_symbols_for_plot(df, max_lines)
    df = df[df["symbol"].isin(symbols)]

    # latest company name per symbol for hover
    last_names: Dict[str, str] = (
        df.sort_values("date").groupby("symbol")["company"].last().to_dict()
    )

    fig = go.Figure()
    for sym in symbols:
        sub = df[df["symbol"] == sym].sort_values("date")
        company = last_names.get(sym, "")
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["rank"],
                mode="lines+markers",
                name=f"{sym}",
                hovertemplate=(
                    "<b>%{customdata[0]} (%{legendgroup})</b><br>"
                    "Date: %{x}<br>"
                    "Rank: %{y}<extra></extra>"
                ),
                legendgroup=sym,
                customdata=[[company] for _ in range(len(sub))],
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date (UTC)",
        yaxis_title="Rank (lower is better; 501 = dropped)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        width=1200,
        height=700,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    fig.update_yaxes(autorange="reversed")  # lower rank at top

    os.makedirs(os.path.dirname(outfile_html), exist_ok=True)
    fig.write_html(outfile_html, include_plotlyjs="cdn")
    return outfile_html
