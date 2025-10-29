# file: projects/sp500-rank-tracker/src/sp500_tracker/cli.py
"""
CLI:
- fetch: 只写当天前 500；历史里出现过但今天不在前 500 的，记 501。
- list: 列表/搜索。
- plot: 静态 PNG；可用 --interactive 输出 HTML（默认上限更高）。
"""
import argparse
import datetime as dt
import os
from typing import List, Optional

from .core import (
    ensure_dirs,
    utc_date,
    fetch_today_table,
    connect_db,
    insert_rows_present,
    insert_rows_dropped,
    known_symbols,
    query_history,
    plot_rank,
    plot_interactive,
)

DEFAULT_TOP = 500
DEFAULT_DROPPED_RANK = 501


def parse_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return dt.date.fromisoformat(s).isoformat()


def cmd_fetch(args: argparse.Namespace) -> None:
    base = os.getcwd()
    data_dir, _plots = ensure_dirs(base)
    db_path = os.path.join(data_dir, "sp500_ranks.sqlite3")
    conn = connect_db(db_path)

    today_all = fetch_today_table()
    today_df = today_all[today_all["rank"] <= args.top].copy()

    today = utc_date()
    inserted_present = insert_rows_present(conn, today, today_df)

    today_syms = set(today_df["symbol"].tolist())
    already_known = known_symbols(conn)
    drops = already_known - today_syms
    inserted_drops = insert_rows_dropped(conn, today, drops, args.dropped_rank)

    print(
        f"[ok] date={today} top={args.top} present={len(today_syms)} "
        f"inserted_present={inserted_present} dropped={len(drops)} inserted_dropped={inserted_drops}"
    )


def cmd_list(args: argparse.Namespace) -> None:
    base = os.getcwd()
    data_dir, _plots = ensure_dirs(base)
    db_path = os.path.join(data_dir, "sp500_ranks.sqlite3")
    conn = connect_db(db_path)

    import pandas as pd

    if args.q:
        df = pd.read_sql_query(
            "SELECT DISTINCT symbol, company FROM rank_history WHERE LOWER(company) LIKE ? ORDER BY symbol",
            conn,
            params=[f"%{args.q.lower()}%"],
        )
    else:
        df = pd.read_sql_query(
            "SELECT DISTINCT symbol, company FROM rank_history ORDER BY symbol", conn
        )

    if df.empty:
        print("No matches.")
    else:
        print(df.to_string(index=False))


def cmd_plot(args: argparse.Namespace) -> None:
    base = os.getcwd()
    data_dir, plots_dir = ensure_dirs(base)
    db_path = os.path.join(data_dir, "sp500_ranks.sqlite3")
    conn = connect_db(db_path)

    symbols: Optional[List[str]] = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    since = parse_date(args.since)
    until = parse_date(args.until)

    df = query_history(
        conn,
        symbols=symbols,
        company_substr=args.company,
        since=since,
        until=until,
        top=args.top,
    )
    if df.empty:
        print("No data to plot.")
        return

    uniq = sorted(df["symbol"].unique().tolist())
    # default max lines: interactive 100, static 20 (can override)
    max_lines = args.max_lines or (100 if args.interactive else 20)
    if len(uniq) > max_lines and not args.interactive:
        keep = set(uniq[:max_lines])
        df = df[df["symbol"].isin(keep)]
        uniq = sorted(keep)

    title_bits = []
    if symbols:
        title_bits.append(",".join(uniq[:5]) + ("..." if len(uniq) > 5 else ""))
    if args.company:
        title_bits.append(f'company~"{args.company}"')
    if args.top:
        title_bits.append(f"top={args.top}")
    if since or until:
        title_bits.append(f"{since or '...'}..{until or '...'}")
    title = (
        "S&P 500 Rank History — " + " | ".join(title_bits)
        if title_bits
        else "S&P 500 Rank History"
    )

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    if args.interactive:
        # interactive HTML
        short = "interactive"
        if symbols:
            short = "_".join(uniq[:5]) or "interactive"
        outfile = os.path.join(plots_dir, f"rank_plot_{short}_{ts}.html")
        path = plot_interactive(
            df,
            outfile,
            title,
            args.dropped_rank,
            max_lines=args.max_lines or 100,
        )
        print(f"[ok] HTML saved -> {path}")
    else:
        # static PNG
        short = "_".join(uniq[:5]) or "all"
        if len(uniq) > 5:
            short += f"_plus{len(uniq)-5}"
        outfile = os.path.join(plots_dir, f"rank_plot_{short}_{ts}.png")
        path = plot_rank(df, outfile, title, args.dropped_rank)
        print(f"[ok] PNG saved -> {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Track S&P 500 ranks daily; only top-N kept (default 500); dropped marked as N+1 (default 501)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser(
        "fetch", help="Fetch today's table, keep top-N, mark known-but-missing as N+1."
    )
    p_fetch.add_argument(
        "--top", type=int, default=DEFAULT_TOP, help="Keep daily top-N (default 500)."
    )
    p_fetch.add_argument(
        "--dropped-rank",
        type=int,
        default=DEFAULT_DROPPED_RANK,
        help="Rank used for dropped constituents (default 501).",
    )
    p_fetch.set_defaults(func=cmd_fetch)

    p_list = sub.add_parser(
        "list", help="List known symbols or fuzzy search companies."
    )
    p_list.add_argument("--q", type=str, help="Fuzzy query on company name.")
    p_list.set_defaults(func=cmd_list)

    p_plot = sub.add_parser("plot", help="Plot rank history (PNG by default).")
    p_plot.add_argument("--symbols", type=str, help="AAPL,MSFT,GOOGL")
    p_plot.add_argument("--company", type=str, help="Fuzzy match by company name.")
    p_plot.add_argument("--top", type=int, help="Filter to daily top-N when querying.")
    p_plot.add_argument("--since", type=str, help="YYYY-MM-DD")
    p_plot.add_argument("--until", type=str, help="YYYY-MM-DD")
    p_plot.add_argument("--max-lines", type=int, help="Limit number of lines on chart.")
    p_plot.add_argument(
        "--dropped-rank", type=int, default=DEFAULT_DROPPED_RANK
    )
    p_plot.add_argument(
        "--interactive",
        action="store_true",
        help="Output interactive HTML (hover, toggle legend).",
    )
    p_plot.set_defaults(func=cmd_plot)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        import sys

        sys.stderr.write(f"[error] {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
