"""Logging estructurado minimo: cada linea lleva job_id o question_id cuando aplica."""
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )


def log_with_id(logger: logging.Logger, level: int, msg: str, **ids: str) -> None:
    suffix = " ".join(f"{k}={v}" for k, v in ids.items() if v)
    logger.log(level, "%s %s", msg, suffix)
