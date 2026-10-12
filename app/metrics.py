# SPDX-License-Identifier: AGPL-3.0-only
class FpsMeter:
    """Compute completions/actual elapsed, resetting after epoch or pause."""
    def __init__(self):
        self.reset()

    def reset(self, count=0, elapsed=0.):
        self.count, self.elapsed, self.fps = count, elapsed, None

    def update(self, count, elapsed):
        if count < self.count or elapsed < self.elapsed:
            self.reset()
        if elapsed - self.elapsed >= .3:
            self.fps = (count - self.count) / (elapsed - self.elapsed)
            self.count, self.elapsed = count, elapsed
        return self.fps
