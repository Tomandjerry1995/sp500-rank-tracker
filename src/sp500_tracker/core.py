# file: projects/sp500-rank-tracker/src/sp500_tracker/core.py
"""
Core: scrape, store (top-500; dropped=501), query, plot (PNG & interactive HTML via Jinja2).
Why:
- 统一规则：当天榜单只保留前500；历史出现但当天不在前500 → 记 501（持续不进前500则每天记501）。
- 交互图：支持“仅看昨日有变化”与“今日名次区间”过滤；默认包含所有（≈500）曲线，靠控件筛选可见性。
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
    s.mount("http://", ad); s.mount("https://", ad)
    s.headers.update({"User-Agent": "sp500-rank-tracker/0.4"})
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

    rank_idx = find_one({"", "rank"})  # "#" often parsed empty
    symbol_idx = find_one({"symbol", "ticker"})
    company_idx = find_one({"company", "name"})
    return rank_idx, symbol_idx, company_idx


def fetch_today_table() -> pd.DataFrame:
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


# ---------- DB ----------
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
    conn.commit(); return inserted


def insert_rows_dropped(
    conn: sqlite3.Connection, date_str: str, drops: Set[str], dropped_rank: int
) -> int:
    cur = conn.cursor(); inserted = 0
    for sym in sorted(drops):
        name = latest_company_name(conn, sym) or sym  # why: keep human-readable
        cur.execute(
            "INSERT OR IGNORE INTO rank_history(date, symbol, company, rank) VALUES (?, ?, ?, ?)",
            (date_str, sym, name, dropped_rank),
        )
        if cur.rowcount > 0:
            inserted += 1
    conn.commit(); return inserted


def query_history(
    conn: sqlite3.Connection,
    symbols: Optional[List[str]] = None,
    company_substr: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    top: Optional[int] = None,
) -> pd.DataFrame:
    params: List[object] = []; where: List[str] = []
    if since: where.append("date >= ?"); params.append(since)
    if until: where.append("date <= ?"); params.append(until)

    if symbols:
        syms = [s.upper().strip() for s in symbols if s.strip()]
        if len(syms) == 1: where.append("symbol = ?"); params.append(syms[0])
        else:
            where.append(f"symbol IN ({','.join('?' for _ in syms)})"); params.extend(syms)

    if company_substr: where.append("LOWER(company) LIKE ?"); params.append(f"%{company_substr.lower()}%")

    base_sql = "SELECT * FROM rank_history"
    if top:
        sub_where = " AND ".join(where) if where else "1=1"
        sql = f"{base_sql} WHERE {sub_where} AND rank <= ?"; bind = tuple(params + [top])
    else:
        sql = base_sql + (" WHERE " + " AND ".join(where) if where else ""); bind = tuple(params)

    df = pd.read_sql_query(sql, conn, params=bind)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["rank"] = df["rank"].astype(int)
        df["symbol"] = df["symbol"].astype(str)
        df["company"] = df["company"].astype(str)
    return df


# ---------- Static plot ----------
def plot_rank(df: pd.DataFrame, outfile: str, title: str, dropped_rank: int) -> str:
    import matplotlib.pyplot as plt
    if df.empty: raise ValueError("No data to plot")
    df = df.copy(); df.loc[df["rank"] > dropped_rank, "rank"] = dropped_rank
    pivot = df.pivot_table(index="date", columns="symbol", values="rank", aggfunc="first").sort_index()
    ax = pivot.plot(figsize=(10, 6), marker="o")
    ax.set_title(title); ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(f"Rank (lower is better; {dropped_rank} = dropped)")
    ax.invert_yaxis(); ax.grid(True, linestyle="--", alpha=0.4)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    import matplotlib.pyplot as plt; plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()
    return outfile


# ---------- Interactive plot (Plotly + Jinja2) ----------
from jinja2 import Environment, FileSystemLoader, select_autoescape
import plotly.graph_objects as go

def _last_two_dates(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if df.empty: return None, None
    dates = sorted(pd.to_datetime(df["date"]).unique())
    if len(dates) == 1: return dates[-1], None
    return dates[-1], dates[-2]


def plot_interactive(
    df: pd.DataFrame,
    outfile_html: str,
    title: str,
    dropped_rank: int,
    max_lines: int,
    include_all: bool = True,
    default_lo: int = 1,
    default_hi: Optional[int] = None,
) -> str:
    """Render interactive HTML:
    - include_all=True: 按“今天名次”把所有(<=dropped_rank)都画出（≈500）；否则仅取前 max_lines。
    - 顶部控件：仅看昨日有变化 + 今日名次区间。
    """
    if df.empty: raise ValueError("No data to plot")
    df = df.copy(); df.loc[df["rank"] > dropped_rank, "rank"] = dropped_rank

    latest, prev = _last_two_dates(df)
    if latest is None: raise ValueError("No latest date found")

    today_map: Dict[str, int] = (
        df[pd.to_datetime(df["date"]) == latest]
        .groupby("symbol")["rank"].first().astype(int).to_dict()
    )
    # 仅保留今天有名次的
    today_sorted_syms = [s for s, r in sorted(today_map.items(), key=lambda kv: kv[1]) if r <= dropped_rank]
    symbols = today_sorted_syms if include_all else today_sorted_syms[:max_lines]
    df = df[df["symbol"].isin(symbols)]

    changed_set: Set[str] = set()
    if prev is not None:
        prev_map = (
            df[pd.to_datetime(df["date"]) == prev]
            .groupby("symbol")["rank"].first().astype(int).to_dict()
        )
        for s in set(today_map) | set(prev_map):
            if today_map.get(s, dropped_rank) != prev_map.get(s, dropped_rank):
                changed_set.add(s)

    last_names: Dict[str, str] = df.sort_values("date").groupby("symbol")["company"].last().to_dict()

    fig = go.Figure()
    for sym in symbols:
        sub = df[df["symbol"] == sym].sort_values("date")
        company = last_names.get(sym, "")
        today_rank = int(today_map.get(sym, dropped_rank))
        changed = bool(sym in changed_set)
        fig.add_trace(
            go.Scatter(
                x=sub["date"], y=sub["rank"],
                mode="lines+markers", name=f"{sym}", legendgroup=sym,
                hovertemplate="<b>%{customdata[0]} (%{legendgroup})</b><br>Date: %{x}<br>Rank: %{y}<extra></extra>",
                customdata=[[company] for _ in range(len(sub))],
                meta=dict(symbol=sym, company=company, today_rank=today_rank, changed=changed),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date (UTC)",
        yaxis_title=f"Rank (lower is better; {dropped_rank} = dropped)",
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=70, b=140),
        width=1200, height=720,
    )
    fig.update_yaxes(autorange="reversed")

    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=select_autoescape())
    tmpl = env.get_template("interactive.html.j2")
    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="rankplot")
    if default_hi is None: default_hi = max_lines
    html = tmpl.render(plot_div=plot_div, dropped=dropped_rank, default_lo=int(default_lo), default_hi=int(default_hi))

    os.makedirs(os.path.dirname(outfile_html), exist_ok=True)
    with open(outfile_html, "w", encoding="utf-8") as f: f.write(html)
    return outfile_html
