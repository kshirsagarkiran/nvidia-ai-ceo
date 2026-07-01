"""Abstract collector interface.

Adding a new data source means subclassing BaseCollector and implementing
collect(). Because every collector returns the same Document type, the rest
of the pipeline (dedup, storage, indexing) never needs to know or care which
source the data came from. This keeps the system open for extension.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.schema import Document


class BaseCollector(ABC):
    source: str = "base"
    source_type: str = "generic"

    @abstractmethod
    def collect(self) -> List[Document]:
        """Fetch from the source and return a list of normalized Documents."""
        raise NotImplementedError
