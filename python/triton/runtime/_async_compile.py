from __future__ import annotations
from typing import Callable, Optional
from concurrent.futures import Executor, as_completed, Future
from contextvars import ContextVar, copy_context

active_mode: ContextVar[Optional[AsyncCompileMode]] = ContextVar("async_compile_active_mode", default=None)


class FutureKernel:

    def __init__(self, finalize_compile: Callable, future: Future):
        self.finalize_compile = finalize_compile
        self.kernel = None
        self.future = future
        self._resolved = False
        self._exception_observed = False

    def result(self):
        if self._resolved:
            return self.kernel

        try:
            kernel = self.future.result()
            self.finalize_compile(kernel)
        except BaseException:
            self._exception_observed = True
            raise
        else:
            self.kernel = kernel
            self._resolved = True
            return kernel


class AsyncCompileMode:

    def __init__(self, executor: Executor):
        self.executor = executor
        self.raw_futures = []
        self.future_kernels = {}

    def submit(self, key, compile_fn, finalize_fn):
        future = self.future_kernels.get(key)
        if future is not None:
            return future

        # Context objects cannot be entered concurrently, so every submission
        # needs its own snapshot rather than sharing one context per mode.
        context = copy_context()

        def compile_in_worker():
            # Propagate caller-owned telemetry and other context variables, but
            # do not make nested JIT calls in the worker resubmit themselves to
            # the outer mode.
            token = active_mode.set(None)
            try:
                return compile_fn()
            finally:
                active_mode.reset(token)

        future = self.executor.submit(context.run, compile_in_worker)
        future._key = key
        self.raw_futures.append(future)
        future_kernel = FutureKernel(finalize_fn, future)
        self.future_kernels[key] = future_kernel
        return future_kernel

    def __enter__(self):
        if active_mode.get() is not None:
            raise RuntimeError("Another AsyncCompileMode is already active")
        self._active_mode_token = active_mode.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            # Finalize any outstanding compiles. An exception already observed
            # by the caller must not be raised for a second time on mode exit.
            for future in as_completed(self.raw_futures):
                future_kernel = self.future_kernels[future._key]
                if not future_kernel._exception_observed:
                    future_kernel.result()
        finally:
            active_mode.reset(self._active_mode_token)
