"""Parallel processing utilities for PCMFG.

Provides concurrent API call processing with rate limiting and progress tracking.
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ProcessingResult(Generic[R]):
    """Result of processing a single item."""

    index: int
    result: R | None
    error: Exception | None
    success: bool


class ParallelProcessor(Generic[T, R]):
    """Process items in parallel with concurrency control.

    Uses ThreadPoolExecutor for I/O-bound operations (API calls).
    """

    def __init__(
        self,
        process_fn: Callable[[T], R],
        max_concurrency: int = 5,
        on_progress: Callable[[int, int], None] | None = None,
        on_error: Callable[[int, Exception], None] | None = None,
    ) -> None:
        """Initialize parallel processor.

        Args:
            process_fn: Function to process each item.
            max_concurrency: Maximum number of concurrent operations.
            on_progress: Callback for progress updates (completed, total).
            on_error: Callback for error handling (index, error).
        """
        self.process_fn = process_fn
        self.max_concurrency = max_concurrency
        self.on_progress = on_progress
        self.on_error = on_error

    def process(self, items: list[T]) -> list[ProcessingResult[R]]:
        """Process all items in parallel.

        Args:
            items: List of items to process.

        Returns:
            List of processing results in original order.
        """
        results: list[ProcessingResult[R]] = [None] * len(items)  # type: ignore
        completed = 0

        def process_with_index(index: int, item: T) -> ProcessingResult[R]:
            nonlocal completed
            try:
                result = self.process_fn(item)
                processing_result = ProcessingResult(
                    index=index,
                    result=result,
                    error=None,
                    success=True,
                )
            except Exception as e:
                logger.error(f"Error processing item {index}: {e}")
                processing_result = ProcessingResult(
                    index=index,
                    result=None,
                    error=e,
                    success=False,
                )
                if self.on_error:
                    self.on_error(index, e)

            completed += 1
            if self.on_progress:
                self.on_progress(completed, len(items))

            return processing_result

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = [
                executor.submit(process_with_index, i, item)
                for i, item in enumerate(items)
            ]

            for i, future in enumerate(futures):
                results[i] = future.result()

        return results


class AsyncParallelProcessor(Generic[T, R]):
    """Async version of parallel processor.

    Uses asyncio for better control over concurrent operations.
    """

    def __init__(
        self,
        process_fn: Callable[[T], Any],  # Returns coroutine
        max_concurrency: int = 5,
        on_progress: Callable[[int, int], None] | None = None,
        on_error: Callable[[int, Exception], None] | None = None,
    ) -> None:
        """Initialize async parallel processor.

        Args:
            process_fn: Async function to process each item.
            max_concurrency: Maximum number of concurrent operations.
            on_progress: Callback for progress updates (completed, total).
            on_error: Callback for error handling (index, error).
        """
        self.process_fn = process_fn
        self.max_concurrency = max_concurrency
        self.on_progress = on_progress
        self.on_error = on_error

    async def process(self, items: list[T]) -> list[ProcessingResult[R]]:
        """Process all items asynchronously.

        Args:
            items: List of items to process.

        Returns:
            List of processing results in original order.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0
        results: list[ProcessingResult[R]] = [None] * len(items)  # type: ignore

        async def process_with_semaphore(index: int, item: T) -> ProcessingResult[R]:
            nonlocal completed
            async with semaphore:
                try:
                    result = await self.process_fn(item)
                    processing_result = ProcessingResult(
                        index=index,
                        result=result,
                        error=None,
                        success=True,
                    )
                except Exception as e:
                    logger.error(f"Error processing item {index}: {e}")
                    processing_result = ProcessingResult(
                        index=index,
                        result=None,
                        error=e,
                        success=False,
                    )
                    if self.on_error:
                        self.on_error(index, e)

                completed += 1
                if self.on_progress:
                    self.on_progress(completed, len(items))

                return processing_result

        tasks = [process_with_semaphore(i, item) for i, item in enumerate(items)]

        completed_results = await asyncio.gather(*tasks)

        for i, result in enumerate(completed_results):
            results[i] = result

        return results


def process_in_batches(
    items: list[T],
    process_fn: Callable[[T], R],
    batch_size: int = 10,
    on_batch_complete: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Process items in batches (sequential, for rate limiting).

    Args:
        items: List of items to process.
        process_fn: Function to process each item.
        batch_size: Number of items per batch.
        on_batch_complete: Callback after each batch (batch_num, total_batches).

    Returns:
        List of results in original order.
    """
    results: list[R] = []
    total_batches = (len(items) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(items), batch_size)):
        batch = items[i : i + batch_size]
        batch_results = [process_fn(item) for item in batch]
        results.extend(batch_results)

        if on_batch_complete:
            on_batch_complete(batch_num + 1, total_batches)

    return results


def parallel_process_chunks(
    chunks: list[T],
    process_fn: Callable[[T], R],
    max_concurrency: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[R | None]:
    """Process chunks in parallel with concurrency control.

    Convenience function that returns only successful results in order.

    Args:
        chunks: List of chunks to process.
        process_fn: Function to process each chunk.
        max_concurrency: Maximum concurrent operations.
        on_progress: Progress callback.

    Returns:
        List of results (None for failed items).
    """
    processor = ParallelProcessor(
        process_fn=process_fn,
        max_concurrency=max_concurrency,
        on_progress=on_progress,
    )

    results = processor.process(chunks)
    return [r.result if r.success else None for r in results]
