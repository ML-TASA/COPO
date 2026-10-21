"""Stand-ins for the ``minigpt4.common`` helpers used by the vendored model modules."""

from __future__ import annotations

import os

import torch


def is_url(url_or_filename) -> bool:
    return isinstance(url_or_filename, str) and url_or_filename.startswith(
        ("http://", "https://")
    )


def download_cached_file(url, check_hash=False, progress=True, cache_dir=None):
    """Download ``url`` to a local cache and return the cached path."""
    from transformers.utils import download_url

    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "minigpt4")
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.basename(url.split("?")[0]) or "download"
    local = os.path.join(cache_dir, fname)
    if os.path.isfile(local):
        return local
    download_url(url, local)
    return local


def is_dist_avail_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class Registry:
    """Minimal registry replacing ``minigpt4.common.registry``."""

    def __init__(self, name: str):
        self._name = name
        self._objs = {}

    def _register(self, key, obj):
        key = key or obj.__name__
        self._objs[key] = obj
        return obj

    def register(self, key=None):
        def wrap(obj):
            return self._register(key, obj)

        return wrap

    def register_model(self, key):
        return self.register(key)

    def register_processor(self, key):
        return self.register(key)

    def get(self, key):
        if key not in self._objs:
            raise KeyError(f"{self._name}: unknown entry {key!r}")
        return self._objs[key]


registry = Registry("minigpt4")
