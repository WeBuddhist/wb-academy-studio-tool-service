"""Exceptions shared by the worker and the tools."""


class TryLater(Exception):
    """Something the job needs is unavailable for now (e.g. Studio is restarting): run the job again later."""
