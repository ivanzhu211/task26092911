import random
import os
import tempfile
import time
import unittest

from genealogy import GenealogyStore, GenealogyError


class BruteForce:
    """Reference implementation: re-evaluates every integer time point."""

    def __init__(self, max_time):
        self.max_time = max_time
        self.lots = {}
        self.consumption = []
        self.installs = []

    def register_lot(self, lot_id, t, sources=()):
        if lot_id in self.lots:
            raise GenealogyError("dup")
        for s in sources:
            if s not in self.lots or self.lots[s][0] > t:
                raise GenealogyError("bad source")
        self.lots[lot_id] = (t, tuple(sources))

    def consume(self, lot_id, serial_id, t):
        if lot_id not in self.lots or t < self.lots[lot_id][0]:
            raise GenealogyError("bad consume")
        self.consumption.append((lot_id, serial_id, t))

    def _parents_at(self, tau):
        m = {}
        for c, p, s, e in self.installs:
            if s <= tau and (e is None or e > tau):
                m[c] = p
        return m

    def install(self, child, parent, start, end=None):
        if child == parent or (end is not None and end <= start):
            raise GenealogyError("bad interval")
        stop = end if end is not None else self.max_time + 1
        for tau in range(start, stop):
            m = self._parents_at(tau)
            if child in m:
                raise GenealogyError("double install")
            node = parent
            while node is not None:
                if node == child:
                    raise GenealogyError("cycle")
                node = m.get(node)
        self.installs.append((child, parent, start, end))

    def remove(self, child, parent, t):
        for i, (c, p, s, e) in enumerate(self.installs):
            if c == child and p == parent and e is None:
                if t < s:
                    raise GenealogyError("remove before start")
                self.installs[i] = (c, p, s, t)
                return
        raise GenealogyError("no open install")

    def _downstream_lots(self, lot_id):
        lots = {lot_id}
        frontier = [lot_id]
        while frontier:
            cur = frontier.pop()
            for lot, (_, srcs) in self.lots.items():
                if cur in srcs and lot not in lots:
                    lots.add(lot)
                    frontier.append(lot)
        return lots

    def forward_trace(self, lot_id):
        lots = self._downstream_lots(lot_id)
        hits = {}
        # max_time + 1 stands in for the unbounded tail.
        for tau in range(0, self.max_time + 2):
            m = self._parents_at(tau)
            containing = set()
            for lot, serial, t in self.consumption:
                if lot in lots and t <= tau:
                    node = serial
                    while node is not None and node not in containing:
                        containing.add(node)
                        node = m.get(node)
            for serial in containing:
                hits.setdefault(serial, []).append(tau)
        out = {}
        for serial, taus in hits.items():
            ivs = []
            for tau in taus:
                if ivs and tau == ivs[-1][1]:
                    ivs[-1][1] += 1
                else:
                    ivs.append([tau, tau + 1])
            if ivs and ivs[-1][1] == self.max_time + 2:
                ivs[-1][1] = None
            out[serial] = [tuple(iv) for iv in ivs]
        return out

    def backward_trace(self, serial_id, at):
        children = sorted(
            c
            for c, p, s, e in self.installs
            if p == serial_id and s <= at and (e is None or e > at)
        )
        lots = {
            lot
            for lot, serial, t in self.consumption
            if serial == serial_id and t <= at
        }
        frontier = list(lots)
        while frontier:
            cur = frontier.pop()
            for src in self.lots[cur][1]:
                if src not in lots:
                    lots.add(src)
                    frontier.append(src)
        return {
            "serial": serial_id,
            "lots": sorted(lots),
            "children": [self.backward_trace(c, at) for c in children],
        }


def apply_both(store, brute, op, *args):
    """Run a write on both implementations; assert same accept/reject."""
    err_store = err_brute = None
    try:
        getattr(brute, op)(*args)
    except GenealogyError as exc:
        err_brute = exc
    try:
        getattr(store, op)(*args)
    except GenealogyError as exc:
        err_store = exc
    if (err_store is None) != (err_brute is None):
        raise AssertionError(
            f"{op}{args}: store {'rejected' if err_store else 'accepted'}"
            f" but brute {'rejected' if err_brute else 'accepted'}"
        )
    return err_store is None


class RandomizedTest(unittest.TestCase):
    def test_random_histories(self):
        MAX_T = 40
        for trial in range(6):
            rng = random.Random(1000 + trial)
            store = GenealogyStore(":memory:")
            # Install ends can reach MAX_T - 1 + 7; evaluate past that.
            brute = BruteForce(MAX_T + 10)
            serials = [f"S{i}" for i in range(12)]
            lots = []
            for _ in range(300):
                kind = rng.randrange(6)
                t = rng.randrange(0, MAX_T)
                if kind == 0 or not lots:
                    lot = f"L{trial}_{len(lots)}"
                    eligible = [
                        l for l in lots if brute.lots[l][0] <= t
                    ]
                    sources = tuple(
                        rng.sample(
                            eligible, min(len(eligible), rng.randrange(0, 3))
                        )
                    ) if eligible else ()
                    if apply_both(
                        store, brute, "register_lot", lot, t, sources
                    ):
                        lots.append(lot)
                elif kind == 1:
                    apply_both(
                        store, brute, "consume",
                        rng.choice(lots), rng.choice(serials), t,
                    )
                elif kind in (2, 3):
                    child, parent = rng.sample(serials, 2)
                    end = (
                        None
                        if rng.random() < 0.4
                        else t + rng.randrange(1, 8)
                    )
                    apply_both(
                        store, brute, "install", child, parent, t, end
                    )
                elif kind == 4:
                    ongoing = [
                        (c, p) for c, p, s, e in brute.installs if e is None
                    ]
                    if ongoing and rng.random() < 0.8:
                        child, parent = rng.choice(ongoing)
                    else:
                        child, parent = rng.sample(serials, 2)
                    apply_both(store, brute, "remove", child, parent, t)
                else:
                    serial = rng.choice(serials)
                    self.assertEqual(
                        store.backward_trace(serial, t),
                        brute.backward_trace(serial, t),
                        f"trial {trial}",
                    )
            for lot in lots:
                self.assertEqual(
                    store.forward_trace(lot),
                    brute.forward_trace(lot),
                    f"trial {trial} lot {lot}",
                )
            store.close()


class DeepLayerTest(unittest.TestCase):
    def test_twenty_layers(self):
        LAYERS = 20
        MAX_T = 45
        store = GenealogyStore(":memory:")
        brute = BruteForce(MAX_T)
        for impl in (store, brute):
            impl.register_lot("ROOT", 0)
            impl.consume("ROOT", "L19s0", 1)
            impl.consume("ROOT", "L19s1", 2)
            for i in range(LAYERS - 1):
                for j in range(2):
                    child = f"L{i + 1}s{j}"
                    impl.install(child, f"L{i}s0", 10, 20)
                    impl.install(child, f"L{i}s1", 30, 40)
        start = time.perf_counter()
        got = store.forward_trace("ROOT")
        elapsed = time.perf_counter() - start
        self.assertEqual(got, brute.forward_trace("ROOT"))
        self.assertLess(elapsed, 5.0, f"forward_trace took {elapsed:.2f}s")
        store.close()


class ExplicitCasesTest(unittest.TestCase):
    def test_mutual_install_disjoint_times(self):
        g = GenealogyStore(":memory:")
        g.install("A", "B", 1, 5)
        g.install("B", "A", 6, 9)
        g.close()

    def test_same_time_move_between_parents(self):
        g = GenealogyStore(":memory:")
        g.install("A", "B", 1)
        g.remove("A", "B", 7)
        g.install("A", "C", 7)
        g.close()

    def test_backfill_out_of_order(self):
        g = GenealogyStore(":memory:")
        g.install("A", "B", 10, 20)
        g.install("A", "C", 1, 5)  # earlier business time, written later
        g.register_lot("L1", 3)
        g.register_lot("L2", 2)  # registered earlier, recorded later
        g.consume("L2", "A", 4)
        g.consume("L1", "C", 8)
        self.assertEqual(
            g.forward_trace("L2"),
            {"A": [(4, None)], "B": [(10, 20)], "C": [(4, 5)]},
        )
        g.close()

    def test_rejected_write_leaves_no_partial_state(self):
        g = GenealogyStore(":memory:")
        g.register_lot("L1", 1)
        with self.assertRaises(GenealogyError):
            g.register_lot("L2", 2, sources=("L1", "MISSING"))
        g.register_lot("L2", 2, sources=("L1",))  # retry must succeed
        g.install("A", "B", 1, 5)
        with self.assertRaises(GenealogyError):
            g.install("B", "A", 2, 3)
        g.install("B", "A", 5, 8)  # still allowed afterwards
        g.close()

    def test_file_persistence(self):
        fd, path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        try:
            g = GenealogyStore(path)
            g.register_lot("L", 0)
            g.consume("L", "A", 1)
            g.install("A", "B", 2, 6)
            g.close()
            g2 = GenealogyStore(path)
            self.assertEqual(
                g2.forward_trace("L"),
                {"A": [(1, None)], "B": [(2, 6)]},
            )
            g2.close()
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
