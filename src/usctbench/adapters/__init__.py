"""Optional external-tool adapters."""

from .matlab import MatlabAdapter, MatlabUnavailable, find_matlab

__all__ = ["MatlabAdapter", "MatlabUnavailable", "find_matlab"]

