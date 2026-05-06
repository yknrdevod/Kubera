"""
particle/main.py
================
Entry point for the Particle system.

Interactive:
    python -m particle.main

Agentic:
    python -m particle.main --action update    --symbol SBIN
    python -m particle.main --action swing     --symbol SBIN --timeframe 1min
    python -m particle.main --action overnight --symbol SBIN
    python -m particle.main --action correction --symbol SBIN --timeframe daily
    python -m particle.main --action full      --symbol SBIN
    python -m particle.main --action status
    python -m particle.main --action swing     --symbol SBIN --json
"""

import logging
import sys
from datetime import datetime

from particle import __version__
from particle import config as cfg
from particle.cli import build_parser, run_agentic, run_interactive


def setup_console_encoding():
    """Make Windows consoles tolerate the CLI's Unicode separators."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def setup_logging(verbose: bool = False):
    level   = logging.DEBUG if verbose else logging.INFO
    fmt     = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%H:%M:%S"
    cfg.init_dirs()
    handlers = [logging.StreamHandler(sys.stdout)]
    log_path = cfg.LOGS_DIR / "particle.log"
    try:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except OSError:
        fallback = cfg.LOGS_DIR / f"particle_{datetime.now():%Y%m%d_%H%M%S}.log"
        try:
            handlers.append(logging.FileHandler(fallback, encoding="utf-8"))
            print(f"Log file locked, writing this run to {fallback}")
        except OSError:
            print("Log file unavailable, continuing with console logging only.")
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def main():
    setup_console_encoding()
    parser = build_parser()
    args   = parser.parse_args()
    cfg.init_dirs()
    setup_logging(verbose=getattr(args, "verbose", False))
    log = logging.getLogger(__name__)
    log.info(f"Particle Pressure Model  v{__version__}")

    if args.action:
        sys.exit(run_agentic(args))
    else:
        run_interactive()


if __name__ == "__main__":
    main()
