import random
import time
import unittest

from genealogy import GenealogyError, GenealogyStore

HORIZON = 120  # brute force evaluates instants [0, HORIZON)


class BruteForce:
    """Per-instant reference model: re-evaluates every time tick."""

    def __init__(self, horizon=HORIZON):
        self.horizon = horizon
        self.lots = {}
        self.sources = {}
        self.consumes = []
        self.installs = []  # [child, parent, start, end]

    # -- writes ---------------------------------------------------------
    def register_lot(self, lot, t, sources=()):
        if lot in self.lots:
            return False
        if any(s not in self.lots or self.lots[s] > t for s in sources):
            return False
        self.lots[lot] = t
        self.sources[lot] = set(sources)
        return True

    def consume(self, lot, serial, t):
        if lot not in self.lots or t < self.lots[lot]:
            return False
        self.consumes.append((lot, serial, t))
        return True

    def _active(self, rec, t):
        _, _, s, e = rec
        return s <= t and (e is None or t < e)

    def can_install(self, child, parent, s, e):
        if e is not None and e <= s:
            return False
        end = e if e is not None else self.horizon
        for t in range(s, end):
            if any(r[0] == child and self._active(r, t) for r in self.installs):
                return False
            # cycle: can we walk parent ->* child along edges active at t?
            seen, stack = set(), [parent]
            while stack:
                node = stack.pop()
                if node == child:
                    return False
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(
                    r[1]
                    for r in self.installs
                    if r[0] == node and self._active(r, t)
                )
        return True

    def install(self, child, parent, s, e):
        if not self.can_install(child, parent, s, e):
            return False
        self.installs.append([child, parent, s, e])
        return True

    def remove(self, child, parent, t):
        for rec in self.installs:
            if rec[0] == child and rec[1] == parent and self._active(rec, t):
                rec[3] = t
                return True
        return False

    # -- queries --------------------------------------------------------
    def _downstream(self, lot):
        seen, stack = {lot}, [lot]
        while stack:
            cur = stack.pop()
            for cand, srcs in self.sources.items():
                if cur in srcs and cand not in seen:
                    seen.add(cand)
                    stack.append(cand)
        return seen

    def forward_trace(self, lot):
        """serial -> set of instants containing material from lot."""
        downstream = self._downstream(lot)
        result = {}
        for t in range(self.horizon):
            contains = {
                serial
                for (l, serial, tc) in self.consumes
                if l in downstream and tc <= t
            }
            changed = True
            while changed:
                changed = False
                for rec in self.installs:
                    c, p = rec[0], rec[1]
                    if self._active(rec, t) and c in contains and p not in contains:
                        contains.add(p)
                        changed = True
            for serial in contains:
                result.setdefault(serial, set()).add(t)
        return result


def store_coverage(store_result, serial, t):
    return any(
        a <= t and (b is None or t < b)
        for a, b in store_result.get(serial, [])
    )


def assert_well_formed(test, store_result):
    for serial, ivs in store_result.items():
        test.assertTrue(ivs, "empty interval list for %r" % serial)
        for i, (a, b) in enumerate(ivs):
            test.assertLess(a, b if b is not None else float("inf"))
            if i:
                prev_b = ivs[i - 1][1]
                test.assertIsNotNone(prev_b)
                test.assertLess(prev_b, a, "intervals must not touch/overlap")


class RandomizedDifferentialTest(unittest.TestCase):
    def test_random_histories_match_brute_force(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                self._run_one(seed)

    def _run_one(self, seed):
        rng = random.Random(seed)
        serials = ["S%d" % i for i in range(12)]
        store = GenealogyStore(":memory:")
        brute = BruteForce()
        registered = []

        def rand_time(lo=0, hi=80):
            return rng.randint(lo, hi)

        for step in range(400):
            op = rng.randrange(10)
            if op < 2 and len(registered) < 8:
                lot = "L%d" % len(registered)
                t = rand_time(0, 40)
                srcs = tuple(
                    rng.sample(registered, min(len(registered), rng.randrange(3)))
                )
                ok = brute.register_lot(lot, t, srcs)
                try:
                    store.register_lot(lot, t, srcs)
                    store_ok = True
                except GenealogyError:
                    store_ok = False
                self.assertEqual(ok, store_ok, "register_lot %r" % lot)
                if ok:
                    registered.append(lot)
            elif op < 4:
                lot = rng.choice(registered + ["GHOST"])
                serial, t = rng.choice(serials), rand_time(0, 60)
                ok = brute.consume(lot, serial, t)
                try:
                    store.consume(lot, serial, t)
                    store_ok = True
                except GenealogyError:
                    store_ok = False
                self.assertEqual(ok, store_ok, "consume %r" % lot)
            elif op < 8:
                child = rng.choice(serials)
                parent = rng.choice(serials + [child])  # sometimes self
                start = rand_time(0, 70)
                end = rng.choice([None, rand_time(start - 2, start + 10)])
                if end is not None and end < 0:
                    end = 0
                ok = brute.install(child, parent, start, end)
                try:
                    store.install(child, parent, start, end)
                    store_ok = True
                except GenealogyError:
                    store_ok = False
                self.assertEqual(
                    ok, store_ok,
                    "install %r->%r [%s,%s)" % (child, parent, start, end),
                )
            else:
                if brute.installs and rng.random() < 0.7:
                    rec = rng.choice(brute.installs)
                    child, parent = rec[0], rec[1]
                else:
                    child, parent = rng.choice(serials), rng.choice(serials)
                t = rand_time(0, 80)
                ok = brute.remove(child, parent, t)
                try:
                    store.remove(child, parent, t)
                    store_ok = True
                except GenealogyError:
                    store_ok = False
                self.assertEqual(
                    ok, store_ok, "remove %r<-%r @%s" % (child, parent, t)
                )
            if step % 25 == 24 and registered:
                lot = rng.choice(registered)
                got = store.forward_trace(lot)
                want = brute.forward_trace(lot)
                assert_well_formed(self, got)
                for serial in set(got) | set(want):
                    for t in range(HORIZON):
                        self.assertEqual(
                            serial in want and t in want[serial],
                            store_coverage(got, serial, t),
                            "lot %r serial %r t=%d" % (lot, serial, t),
                        )
        store.close()


class DeepHierarchyTest(unittest.TestCase):
    """20 layers x 2 serials; every child sits under both parents of the
    layer above in two consecutive intervals -> 2**20 static paths."""

    LAYERS = 20

    def _build(self, store, brute):
        for i in range(self.LAYERS - 1, 0, -1):
            for j in range(2):
                child = "L%d_%d" % (i, j)
                for parent, s, e in (
                    ("L%d_0" % (i - 1), 0, 10),
                    ("L%d_1" % (i - 1), 10, 20),
                ):
                    self.assertTrue(brute.install(child, parent, s, e))
                    store.install(child, parent, s, e)
        store.register_lot("LOT", 0)
        brute.register_lot("LOT", 0)
        for j in range(2):
            store.consume("LOT", "L%d_%d" % (self.LAYERS - 1, j), 0)
            brute.consume("LOT", "L%d_%d" % (self.LAYERS - 1, j), 0)

    def test_static_paths_exceed_one_million(self):
        paths = 2 * 2 ** (self.LAYERS - 1)
        self.assertGreater(paths, 1_000_000)

    def test_forward_trace_deep_hierarchy(self):
        store = GenealogyStore(":memory:")
        brute = BruteForce(horizon=30)
        self._build(store, brute)
        started = time.monotonic()
        got = store.forward_trace("LOT")
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 5.0, "forward_trace took %.2fs" % elapsed)
        want = brute.forward_trace("LOT")
        assert_well_formed(self, got)
        self.assertEqual(set(got), set(want))
        for serial in want:
            for t in range(30):
                self.assertEqual(
                    t in want[serial],
                    store_coverage(got, serial, t),
                    "serial %r t=%d" % (serial, t),
                )
        store.close()


class BackwardTraceTest(unittest.TestCase):
    def test_tree_lots_and_upstream(self):
        store = GenealogyStore(":memory:")
        store.register_lot("RAW", 0)
        store.register_lot("MIX", 2, sources=("RAW",))
        store.consume("MIX", "C1", 3)
        store.consume("RAW", "TOP", 1)
        store.install("C1", "MID", 4, 9)
        store.install("MID", "TOP", 5, 8)
        tree = store.backward_trace("TOP", 6)
        self.assertEqual(tree["serial"], "TOP")
        self.assertEqual(tree["lots"], ["RAW"])
        self.assertEqual(tree["upstream"], [])
        mid, = tree["children"]
        self.assertEqual((mid["serial"], mid["lots"]), ("MID", []))
        c1, = mid["children"]
        self.assertEqual(c1["lots"], ["MIX"])
        self.assertEqual(c1["upstream"], ["RAW"])
        self.assertEqual(store.backward_trace("TOP", 4)["children"], [])
        self.assertEqual(store.backward_trace("TOP", 9)["children"], [])
        store.close()

    def test_persists_to_file(self):
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        try:
            store = GenealogyStore(path)
            store.register_lot("L", 0)
            store.consume("L", "SN", 1)
            store.close()
            reopened = GenealogyStore(path)
            self.assertEqual(reopened.forward_trace("L"), {"SN": [(1, None)]})
            reopened.close()
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
