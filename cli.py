"""
particle/cli.py
===============
Interactive menus + agentic argparse for the Particle system.

Interactive:  python -m particle.main
Agentic:      python -m particle.main --action swing --symbol SBIN --timeframe 1min

Adding future modules:
  1. Add engine in particle/engines/yourmodule.py with run()
  2. Add cmd_yourmodule() here
  3. Add to MENU and build_parser()
  4. Handle in run_agentic()
"""

import argparse, json, logging, sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from . import config as cfg, storage, downloader, instruments

log = logging.getLogger(__name__)

SEP  = "─" * 58
SEP2 = "═" * 58

def _header(t):  print(f"\n{SEP2}\n  {t}\n{SEP2}")
def _section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def _ok(m):      print(f"  OK  {m}")
def _warn(m):    print(f"  WW  {m}")
def _err(m):     print(f"  EE  {m}")
def _info(m):    print(f"  >>  {m}")

def _dl(r):
    sym = r.get("symbol","")
    if r.get("status") == "error": _err(f"[{sym}] {r.get('message')}")
    else:
        _ok(f"[{sym}] {r.get('message')}")
        if r.get("total_fetched"): _info(f"Fetched {r['total_fetched']:,} candles")

def _ask(prompt, default=""):
    sfx = f" [{default}]" if default else ""
    v   = input(f"\n  {prompt}{sfx}: ").strip()
    return v if v else default

def _choose(options, prompt="Choose"):
    print()
    for i, o in enumerate(options, 1): print(f"    {i}. {o}")
    print(f"    0. Back")
    while True:
        raw = input(f"\n  {prompt} [0-{len(options)}]: ").strip()
        if raw == "0": return None
        if raw.isdigit() and 1 <= int(raw) <= len(options): return int(raw)-1
        print("  Invalid.")

def _ask_symbols():
    raw = _ask("Symbol(s) — e.g. SBIN  or  SBIN,ICICIBANK")
    if not raw: return []
    return [s.strip().upper() for s in raw.replace(","," ").split() if s.strip()]

def _ask_date(prompt):
    raw = _ask(f"{prompt} YYYY-MM-DD")
    if not raw: return None
    try: return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError: _err("Invalid date."); return None

def _ask_tf():
    c = _choose(["1-Minute (detailed)", "Daily (broad view)"], "Timeframe")
    return "daily" if c == 1 else "1min"

def _ask_outdir():
    return Path(_ask("Output directory", str(cfg.REPORTS_DIR)))

def _load(symbol):
    df = storage.load(symbol)
    if df is None: _err(f"No data for {symbol}. Run update first.")
    return df


# ── DATA COMMANDS ─────────────────────────────────────────────────────────────

def cmd_update(symbols, interactive=True):
    if not symbols and interactive: symbols = _ask_symbols()
    if not symbols: _warn("No symbols."); return []
    _section(f"Smart Update — {', '.join(symbols)}")
    results = downloader.update_many(symbols)
    for r in results: _dl(r)
    return results

def cmd_backfill(symbols, years=cfg.MAX_YEARS, interactive=True):
    if not symbols and interactive: symbols = _ask_symbols()
    if not symbols: _warn("No symbols."); return []
    if interactive:
        try: years = int(_ask(f"Years (max {cfg.MAX_YEARS})", str(cfg.MAX_YEARS)))
        except ValueError: years = cfg.MAX_YEARS
    _section(f"Backfill — {', '.join(symbols)} — {years}yr")
    results = []
    for sym in symbols:
        _info(f"Backfilling {sym} ...")
        r = downloader.backfill(sym, years)
        _dl(r); results.append(r)
    return results

def cmd_custom(symbol=None, interactive=True):
    if not symbol and interactive:
        syms = _ask_symbols(); symbol = syms[0] if syms else None
    if not symbol: _warn("No symbol."); return {}
    _section(f"Custom Range — {symbol}")
    start = _ask_date("Start date")
    end   = _ask_date("End date") or date.today()
    if not start: _warn("Cancelled."); return {}
    if start > end: _err("Start must be before end."); return {}
    r = downloader.fetch_custom(symbol, start, end); _dl(r); return r

def cmd_status(symbols=None):
    _section("Data Status")
    df = storage.summary()
    if df.empty: _warn("No data downloaded yet."); return {"symbols":[]}
    if symbols: df = df[df["symbol"].isin([s.upper() for s in symbols])]
    print(f"\n  {'SYMBOL':<16} {'START':<12} {'END':<12} {'ROWS':>10} {'UPDATED':<12}")
    print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")
    for _, row in df.iterrows():
        rf = f"{int(row['rows']):,}" if str(row['rows']).isdigit() else str(row['rows'])
        print(f"  {row['symbol']:<16} {row['start']:<12} {row['end']:<12} {rf:>10} {row['updated']:<12}")
    print(f"\n  Total: {len(df)} symbol(s)")
    return {"symbols": df.to_dict(orient="records")}

def cmd_validate(symbols=None, interactive=True):
    if not symbols and interactive: symbols = _ask_symbols()
    if not symbols: symbols = list(storage.load_metadata().keys())
    if not symbols: _warn("No data to validate."); return []
    _section(f"Validation — {', '.join(symbols)}")
    results = []
    for sym in symbols:
        r = storage.validate(sym)
        if r.get("ok"): _ok(f"[{sym}] {r['rows']:,} rows  {r['start']} to {r['end']}  Clean")
        else:
            _warn(f"[{sym}] Issues:")
            for iss in r.get("issues",[]): _err(f"    {iss}")
        results.append(r)
    return results

def cmd_search(query=None, interactive=True):
    if not query and interactive: query = _ask("Search query")
    if not query: return pd.DataFrame()
    _section(f"Instrument Search — {query}")
    df = instruments.search(query)
    if df.empty: _warn(f"No instruments matching '{query}'"); return df
    for i, row in df.head(30).iterrows():
        print(f"  {i+1:>3}. {row['tradingsymbol']:<22}  token: {row['instrument_token']}")
    if len(df) > 30: _info(f"...and {len(df)-30} more.")
    return df


# ── ANALYSIS COMMANDS ─────────────────────────────────────────────────────────

def cmd_swing(symbols, timeframe="1min", outdir=None, interactive=True):
    """Swing pressure engine — 1-min or daily."""
    from .engines import swing as eng
    if not symbols and interactive: symbols = _ask_symbols()
    if not symbols: _warn("No symbols."); return []
    if interactive and not timeframe: timeframe = _ask_tf()
    outdir = outdir or cfg.REPORTS_DIR
    _section(f"Swing — {', '.join(symbols)} — {timeframe.upper()}")
    results = []
    for sym in symbols:
        df = _load(sym)
        if df is None: results.append({"ticker":sym,"status":"error","message":"No data"}); continue
        _info(f"Running swing on {sym} ...")
        r = eng.run(df, ticker=sym, timeframe=timeframe, outdir=outdir, silent=False)
        if r.get("status") == "ok":
            _ok(f"[{sym}] Score:{r['pressure_score']:+.4f}  {r['signal']}  {r['action']}")
        else:
            _err(f"[{sym}] {r.get('message')}")
        results.append(r)
    return results

def cmd_overnight(symbols, outdir=None, interactive=True):
    """Overnight pressure engine — 1-min only."""
    from .engines import overnight as eng
    if not symbols and interactive: symbols = _ask_symbols()
    if not symbols: _warn("No symbols."); return []
    outdir = outdir or cfg.REPORTS_DIR
    _section(f"Overnight — {', '.join(symbols)}")
    results = []
    for sym in symbols:
        df = _load(sym)
        if df is None: results.append({"ticker":sym,"status":"error","message":"No data"}); continue
        _info(f"Running overnight on {sym} ...")
        r = eng.run(df, ticker=sym, outdir=outdir, silent=False)
        if r.get("status") == "ok":
            _ok(f"[{sym}] Score:{r['pressure_score']:+.4f}  {r.get('overnight_action','')}")
        else:
            _err(f"[{sym}] {r.get('message')}")
        results.append(r)
    return results

def cmd_correction(symbols, timeframe="daily", pressure_score=None,
                   outdir=None, interactive=True):
    """Correction profiler + phase detector."""
    from .engines import correction as eng
    if not symbols and interactive: symbols = _ask_symbols()
    if not symbols: _warn("No symbols."); return []
    if interactive and not timeframe: timeframe = _ask_tf()
    outdir = outdir or cfg.REPORTS_DIR
    _section(f"Correction — {', '.join(symbols)} — {timeframe.upper()}")
    results = []
    for sym in symbols:
        df = _load(sym)
        if df is None: results.append({"ticker":sym,"status":"error","message":"No data"}); continue
        _info(f"Running correction on {sym} ...")
        r = eng.run(df, ticker=sym, timeframe=timeframe,
                    pressure_score=pressure_score, outdir=outdir, silent=False)
        if r.get("status") == "ok":
            _ok(f"[{sym}] Phase:{r.get('phase')}  "
                f"Drawdown:{r.get('current_drawdown',0):+.2f}%  "
                f"Confidence:{r.get('confidence')}")
        else:
            _err(f"[{sym}] {r.get('message')}")
        results.append(r)
    return results

def cmd_full(symbols, timeframe="1min", outdir=None, interactive=True):
    """Run swing + overnight + correction for one or more symbols."""
    from .engines import swing as sw, overnight as ov, correction as cor
    if not symbols and interactive: symbols = _ask_symbols()
    if not symbols: _warn("No symbols."); return []
    if interactive and not timeframe: timeframe = _ask_tf()
    outdir = outdir or cfg.REPORTS_DIR
    all_results = []

    for sym in symbols:
        _header(f"Full Analysis — {sym}")
        df = _load(sym)
        if df is None: all_results.append({"ticker":sym,"status":"error"}); continue

        swing_r = sw.run(df, ticker=sym, timeframe=timeframe, outdir=outdir, silent=False)
        on_r    = ov.run(df, ticker=sym, outdir=outdir, silent=False) if timeframe=="1min" else None
        ps      = swing_r.get("pressure_score") if swing_r.get("status")=="ok" else None
        cor_r   = cor.run(df, ticker=sym, timeframe="daily",
                          pressure_score=ps, outdir=outdir, silent=False)

        _section(f"COMBINED SUMMARY — {sym}")
        if swing_r.get("status") == "ok":
            print(f"  Swing  : {swing_r['pressure_score']:+.4f}  {swing_r['signal']}")
            print(f"  Action : {swing_r['action']}")
            if swing_r.get("stop_loss"):
                print(f"  Stop/Target : {swing_r['stop_loss']:,.4f} / {swing_r.get('target'):,.4f}  "
                      f"R:R 1:{swing_r.get('risk_reward',0):.1f}")
        if on_r and on_r.get("status") == "ok":
            print(f"  Overnight : {on_r.get('overnight_action')}")
        if cor_r.get("status") == "ok":
            print(f"  Phase : {cor_r.get('phase')}  {cor_r.get('current_drawdown',0):+.2f}%")
        print(f"  Reports : {outdir}")

        all_results.append({"ticker":sym,"swing":swing_r,"overnight":on_r,"correction":cor_r})
    return all_results


# ── INTERACTIVE MENU ──────────────────────────────────────────────────────────

MENU = [
    "Smart Update            — Download only missing data",
    "Full Backfill           — Download full history",
    "Custom Range            — Download specific dates",
    "Data Status             — Show downloaded data",
    "Validate Data           — Integrity check",
    "Search Instruments      — Find NSE symbols",
    "Swing Analysis          — Pressure score (swing)",
    "Overnight Analysis      — Pressure score (overnight)",
    "Correction Analysis     — Phase + forecast",
    "Full Analysis           — All three engines",
]

def run_interactive():
    _header("PARTICLE PRESSURE MODEL")
    print(f"  Base dir : {cfg.BASE_DIR}")
    print(f"  Timeframe: {cfg.TIMEFRAME}")

    while True:
        _section("Main Menu")
        choice = _choose(MENU + ["Exit"], "Select action")

        if choice is None or choice == len(MENU):
            print("\n  Goodbye.\n"); break
        elif choice == 0:
            cmd_update(_ask_symbols(), interactive=False)
        elif choice == 1:
            cmd_backfill(_ask_symbols(), interactive=True)
        elif choice == 2:
            cmd_custom(interactive=True)
        elif choice == 3:
            cmd_status()
        elif choice == 4:
            cmd_validate(_ask_symbols() or None, interactive=False)
        elif choice == 5:
            cmd_search(interactive=True)
        elif choice == 6:
            cmd_swing(_ask_symbols(), _ask_tf(), _ask_outdir(), interactive=False)
        elif choice == 7:
            cmd_overnight(_ask_symbols(), _ask_outdir(), interactive=False)
        elif choice == 8:
            cmd_correction(_ask_symbols(), _ask_tf(), outdir=_ask_outdir(), interactive=False)
        elif choice == 9:
            cmd_full(_ask_symbols(), _ask_tf(), _ask_outdir(), interactive=False)

        input("\n  Press Enter to continue ...")


# ── AGENTIC ARGPARSE ──────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="particle",
        description="Particle Pressure Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
actions (data):
  update backfill custom status validate search

actions (analysis):
  swing overnight correction full

examples:
  python -m particle.main --action update --symbol SBIN ICICIBANK
  python -m particle.main --action swing --symbol SBIN --timeframe 1min
  python -m particle.main --action overnight --symbol SBIN
  python -m particle.main --action correction --symbol SBIN --timeframe daily
  python -m particle.main --action full --symbol SBIN
  python -m particle.main --action swing --symbol SBIN --json
        """,
    )
    p.add_argument("--action","-a",
                   choices=[
        "update", "backfill", "custom", "status",
        "validate", "search", "swing", "overnight",
        "correction", "full",
        "backtest"   # 🔥 ADD THIS
    ])
    p.add_argument("--symbol","-s", nargs="+", metavar="SYMBOL")
    p.add_argument("--timeframe","-tf", choices=["1min","daily"], default="1min")
    p.add_argument("--start", metavar="YYYY-MM-DD")
    p.add_argument("--end",   metavar="YYYY-MM-DD")
    p.add_argument("--years", type=int, default=cfg.MAX_YEARS)
    p.add_argument("--query","-q")
    p.add_argument("--outdir","-o", default=str(cfg.REPORTS_DIR))
    p.add_argument("--pressure", type=float, default=None,
                   help="Pass pressure score into correction engine")
    p.add_argument("--json", action="store_true",
                   help="Output result as JSON for agentic use")
    p.add_argument("--verbose","-v", action="store_true")
    p.add_argument(
    "--engine",
    choices=["swing", "overnight", "correction"],
    default="swing",
    help="Engine to use for backtesting"
)
    return p


def run_agentic(args) -> int:
    result = None; ok = True
    outdir = Path(args.outdir)

    if args.action == "update":
        if not args.symbol: _err("--symbol required"); return 1
        result = cmd_update(args.symbol, interactive=False)
        ok = all(r.get("status")=="ok" for r in result)

    elif args.action == "backfill":
        if not args.symbol: _err("--symbol required"); return 1
        result = cmd_backfill(args.symbol, args.years, interactive=False)
        ok = all(r.get("status")=="ok" for r in result)

    elif args.action == "custom":
        if not args.symbol or not args.start: _err("--symbol --start required"); return 1
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end   = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
        result = downloader.fetch_custom(args.symbol[0], start, end)
        _dl(result); ok = result.get("status")=="ok"

    elif args.action == "status":
        result = cmd_status(args.symbol)

    elif args.action == "validate":
        result = cmd_validate(args.symbol, interactive=False)
        ok = all(r.get("ok") for r in result)

    elif args.action == "search":
        q = args.query or (args.symbol[0] if args.symbol else "")
        if not q: _err("--query required"); return 1
        result = cmd_search(q, interactive=False)

    elif args.action == "swing":
        if not args.symbol: _err("--symbol required"); return 1
        result = cmd_swing(args.symbol, args.timeframe, outdir, interactive=False)
        ok = all(r.get("status")=="ok" for r in result)

    elif args.action == "overnight":
        if not args.symbol: _err("--symbol required"); return 1
        result = cmd_overnight(args.symbol, outdir, interactive=False)
        ok = all(r.get("status")=="ok" for r in result)

    elif args.action == "correction":
        if not args.symbol: _err("--symbol required"); return 1
        result = cmd_correction(args.symbol, args.timeframe,
                                pressure_score=args.pressure,
                                outdir=outdir, interactive=False)
        ok = all(r.get("status")=="ok" for r in result)

    elif args.action == "full":
        if not args.symbol: _err("--symbol required"); return 1
        result = cmd_full(args.symbol, args.timeframe, outdir, interactive=False)
        ok = all(r.get("swing",{}).get("status")=="ok" for r in result)
    
    elif args.action == "backtest":
        return cmd_backtest(args)

    if args.json and result is not None:
        def _s(o):
            if hasattr(o,"__dict__"): return o.__dict__
            if isinstance(o, set): return list(o)
            return str(o)
        out = result.to_dict(orient="records") if hasattr(result,"to_dict") else result
        print(json.dumps(out, default=_s))

    return 0 if ok else 1

def cmd_backtest(args):
    from particle.storage import load
    from particle.backtest.runner import run_backtest
    from particle.backtest.evaluator import evaluate
    from particle.backtest.report import (
        generate_report,
        generate_simple_conclusion,
        generate_actionable_summary
    )
    from particle.backtest.tradesim_signal import (
        simulate_all_signals,
        evaluate_signal_quality
    )

    symbol = args.symbol[0] if isinstance(args.symbol, list) else args.symbol
    df = load(symbol)

    bt = run_backtest(df, engine=args.engine, timeframe=args.timeframe)

    bt.to_csv(f"{args.symbol}_{args.engine}_backtest.csv", index=False)

    stats = evaluate(bt)
    report = generate_report(bt)

    trades = simulate_all_signals(bt)
    signal_stats = evaluate_signal_quality(trades)

    latest = bt.iloc[-1].to_dict()

    print("\n=== BASIC METRICS ===")
    print(stats)

    print("\n=== REPORT ===")
    print(report)

    print("\n=== SIMPLE ===")
    print(generate_simple_conclusion(report))

    print("\n=== ACTION ===")
    print(generate_actionable_summary(latest))

    print("\n=== SIGNAL VALIDATION ===")
    print(signal_stats)
    
    return 0