"""
Logger
"""

import sys
import queue
import threading
import atexit
from typing import Optional

# ANSI color codes
class _Colors:
    GREEN = "\033[92m"
    GREEN_BOLD = "\033[1;92m"
    RED = "\033[91m"
    RED_BOLD = "\033[1;91m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

_log_queue = queue.Queue()

def _logger_worker():
    """Background thread that safely prints to the terminal without blocking."""
    while True:
        item = _log_queue.get()
        if item is None:
            break
        msg, file = item
        print(msg, file=file)
        _log_queue.task_done()

_worker_thread = threading.Thread(target = _logger_worker, daemon = True)
_worker_thread.start()

def _cleanup_logger():
    """Ensure all pending logs are printed before the script exits."""
    _log_queue.join()
    _log_queue.put(None)
    _worker_thread.join()

atexit.register(_cleanup_logger)

class Logger:
    """
    Lightweight logger
    green  = info / success
    red    = warning / error
    gray   = debug
    """

    def __init__(self, name: str = "", verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self._prefix = f"[{name}] " if name else ""

    def _log(self, color: str, label: str, msg: str, file=sys.stdout):
        prefix = f"{_Colors.CYAN}{self._prefix}{_Colors.RESET}" if self.name else ""
        formatted_msg = f"{prefix}{color}{label}{_Colors.RESET} {msg}"
        
        _log_queue.put((formatted_msg, file))

    def info(self, msg: str):
        """Green — general progress info"""
        self._log(_Colors.GREEN, "INFO", msg)

    def success(self, msg: str):
        """Green bold — task completed"""
        self._log(_Colors.GREEN_BOLD, " OK ", msg)

    def warning(self, msg: str):
        """Red — something to watch out for"""
        self._log(_Colors.RED, "WARN", msg, file=sys.stderr)

    def error(self, msg: str):
        """Red bold — something failed"""
        self._log(_Colors.RED_BOLD, " ERR", msg, file=sys.stderr)

    def debug(self, msg: str):
        """Gray — only shown when verbose=True"""
        if self.verbose:
            self._log(_Colors.GRAY, " DBG", msg)

    def step(self, msg: str):
        """Yellow — sub-step within a process"""
        self._log(_Colors.YELLOW, "STEP", msg)


# Global registry so same name returns same logger
_loggers: dict = {}

def get_logger(name: str = "", verbose: bool = False) -> Logger:
    """
    Get or create a named logger
    Same name returns the same logger instance
    """
    if name not in _loggers:
        _loggers[name] = Logger(name=name, verbose=verbose)
    return _loggers[name]
