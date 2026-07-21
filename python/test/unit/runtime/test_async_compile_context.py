from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Barrier

from triton.runtime._async_compile import AsyncCompileMode, active_mode


def test_async_compile_propagates_contextvars():
    marker = ContextVar("marker", default="missing")
    marker.set("present")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with AsyncCompileMode(pool) as mode:
            future = mode.submit("key", marker.get, lambda value: None)
            assert future.result() == "present"


def test_async_compile_copies_context_for_each_concurrent_submission():
    marker = ContextVar("marker", default="missing")
    both_workers_started = Barrier(2, timeout=2)

    def read_marker():
        both_workers_started.wait()
        return marker.get()

    with ThreadPoolExecutor(max_workers=2) as pool:
        with AsyncCompileMode(pool) as mode:
            marker.set("first")
            first = mode.submit("first", read_marker, lambda value: None)
            marker.set("second")
            second = mode.submit("second", read_marker, lambda value: None)

            assert first.result() == "first"
            assert second.result() == "second"


def test_async_compile_does_not_propagate_active_mode_to_worker():
    marker = ContextVar("marker", default="missing")
    marker.set("present")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with AsyncCompileMode(pool) as mode:
            future = mode.submit(
                "worker-context",
                lambda: (marker.get(), active_mode.get()),
                lambda value: None,
            )
            assert future.result() == ("present", None)


def test_async_compile_does_not_refinalize_resolved_none_result():
    finalized = []

    with ThreadPoolExecutor(max_workers=1) as pool:
        with AsyncCompileMode(pool) as mode:
            future = mode.submit("none", lambda: None, finalized.append)
            assert future.result() is None

    assert finalized == [None]
