import logging

_LOG = logging.getLogger("siliconvton")


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "siliconvton")
