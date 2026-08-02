"""OCR pooling for parallel blob reads."""

import threading
import time

from automaton_attack.ocr import OcrPool


class SlowFake:
    name = "fake"
    made = 0

    def __init__(self):
        SlowFake.made += 1
        self.busy = False

    def read(self, image):
        assert not self.busy, "engine used from two threads at once"
        self.busy = True
        time.sleep(0.05)
        self.busy = False
        return "WORD"


def test_pool_reads_concurrently_without_sharing_engines():
    SlowFake.made = 0
    pool = OcrPool(SlowFake, size=3)
    assert SlowFake.made == 3
    results = []
    started = time.monotonic()
    threads = [threading.Thread(target=lambda: results.append(pool.read(None)))
               for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - started
    assert results == ["WORD"] * 6
    # 6 x 50ms serially = 300ms; three engines should roughly halve it.
    assert elapsed < 0.25
