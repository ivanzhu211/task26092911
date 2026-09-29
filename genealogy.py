"""Assembly genealogy store for a manufacturing execution system.

Persists lots, lot derivation, material consumption and time-bounded
install relations in SQLite, and answers forward / backward traceability
queries. Standard library only.

Conventions:
- Time is an integer; install intervals are half-open [start, end).
- end=None means "still installed" (open interval).
- Serial numbers are not registered; any hashable string id may appear
  in consume/install records.
"""

import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lots (
    lot_id TEXT PRIMARY KEY,
    t INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS lot_sources (
    lot_id TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (lot_id, source)
);
CREATE INDEX IF NOT EXISTS idx_lot_sources_source ON lot_sources(source);
CREATE TABLE IF NOT EXISTS consumes (
    lot_id TEXT NOT NULL,
    serial_id TEXT NOT NULL,
    t INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consumes_lot ON consumes(lot_id);
CREATE INDEX IF NOT EXISTS idx_consumes_serial ON consumes(serial_id);
CREATE TABLE IF NOT EXISTS installs (
    child TEXT NOT NULL,
    parent TEXT NOT NULL,
    start INTEGER NOT NULL,
    "end" INTEGER
);
CREATE INDEX IF NOT EXISTS idx_installs_child ON installs(child);
CREATE INDEX IF NOT EXISTS idx_installs_parent ON installs(parent);
"""


class GenealogyError(Exception):
    """Raised when a write violates genealogy invariants."""


class GenealogyStore:
    def __init__(self, path):
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def register_lot(self, lot_id, t, sources=()):
        with self._conn:
            row = self._conn.execute(
                "SELECT t FROM lots WHERE lot_id = ?", (lot_id,)
            ).fetchone()
            if row is not None:
                raise GenealogyError("lot %r already registered" % (lot_id,))
            for source in sources:
                srow = self._conn.execute(
                    "SELECT t FROM lots WHERE lot_id = ?", (source,)
                ).fetchone()
                if srow is None:
                    raise GenealogyError("unknown source lot %r" % (source,))
                if srow[0] > t:
                    raise GenealogyError(
                        "source lot %r registered at %s, later than %s"
                        % (source, srow[0], t)
                    )
            self._conn.execute(
                "INSERT INTO lots (lot_id, t) VALUES (?, ?)", (lot_id, t)
            )
            for source in dict.fromkeys(sources):
                self._conn.execute(
                    "INSERT INTO lot_sources (lot_id, source) VALUES (?, ?)",
                    (lot_id, source),
                )

    def consume(self, lot_id, serial_id, t):
        with self._conn:
            row = self._conn.execute(
                "SELECT t FROM lots WHERE lot_id = ?", (lot_id,)
            ).fetchone()
            if row is None:
                raise GenealogyError("unknown lot %r" % (lot_id,))
            if t < row[0]:
                raise GenealogyError(
                    "consume at %s predates lot registration at %s"
                    % (t, row[0])
                )
            self._conn.execute(
                "INSERT INTO consumes (lot_id, serial_id, t) VALUES (?, ?, ?)",
                (lot_id, serial_id, t),
            )

    def install(self, child, parent, start, end=None):
        if end is not None and end <= start:
            raise GenealogyError(
                "empty or inverted interval [%s, %s)" % (start, end)
            )
        with self._conn:
            clash = self._conn.execute(
                "SELECT 1 FROM installs WHERE child = ? "
                "AND (? IS NULL OR start < ?) "
                "AND (end IS NULL OR end > ?) LIMIT 1",
                (child, end, end, start),
            ).fetchone()
            if clash is not None:
                raise GenealogyError(
                    "%r already installed elsewhere during [%s, %s)"
                    % (child, start, end)
                )
            contained = self._containment_intervals(child, parent)
            if _clip(contained, start, end):
                raise GenealogyError(
                    "installing %r under %r would create a cycle"
                    % (child, parent)
                )
            self._conn.execute(
                "INSERT INTO installs (child, parent, start, end) "
                "VALUES (?, ?, ?, ?)",
                (child, parent, start, end),
            )

    def remove(self, child, parent, t):
        with self._conn:
            row = self._conn.execute(
                "SELECT rowid, start FROM installs "
                "WHERE child = ? AND parent = ? AND start <= ? "
                "AND (\"end\" IS NULL OR \"end\" > ?) LIMIT 1",
                (child, parent, t, t),
            ).fetchone()
            if row is None:
                raise GenealogyError(
                    "no open install of %r under %r at %s"
                    % (child, parent, t)
                )
            if row[1] == t:
                # ending at the very start: the interval is empty, drop it
                self._conn.execute(
                    "DELETE FROM installs WHERE rowid = ?", (row[0],)
                )
            else:
                self._conn.execute(
                    "UPDATE installs SET \"end\" = ? WHERE rowid = ?",
                    (t, row[0]),
                )

    def _containment_intervals(self, ancestor, descendant):
        """Times during which *descendant* is (transitively) inside *ancestor*.

        Least-fixpoint over the static install graph; safe on statically
        cyclic graphs (e.g. A in B, later B in A) because the accepted
        history keeps every single instant acyclic.  Memoised per node, so
        cost is linear in touched nodes/edges, not in path count.
        """
        state = {ancestor: [(float("-inf"), None)]}
        queue = [ancestor]
        while queue:
            node = queue.pop()
            ivs = state[node]
            rows = self._conn.execute(
                "SELECT child, start, \"end\" FROM installs WHERE parent = ?",
                (node,),
            )
            for child, s, e in rows:
                old = state.get(child)
                new = _merge((old or []) + _clip(ivs, s, e))
                if new != old:
                    state[child] = new
                    queue.append(child)
        return state.get(descendant, [])

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def forward_trace(self, lot_id):
        """Map serial_id -> sorted, merged [(start, end)] intervals (end=None
        means "still contains") during which the serial held material that
        originated from *lot_id* (directly, via derived lots, or via
        installed children at any depth)."""
        downstream = self._downstream_lots(lot_id)
        state = {}
        queue = []

        def add(serial, ivs):
            old = state.get(serial)
            new = _merge((old or []) + ivs)
            if new != old:
                state[serial] = new
                queue.append(serial)

        for lot in downstream:
            rows = self._conn.execute(
                "SELECT serial_id, t FROM consumes WHERE lot_id = ?", (lot,)
            )
            for serial, t in rows:
                add(serial, [(t, None)])
        while queue:
            node = queue.pop()
            ivs = state[node]
            rows = self._conn.execute(
                "SELECT parent, start, \"end\" FROM installs WHERE child = ?",
                (node,),
            )
            for parent, s, e in rows:
                add(parent, _clip(ivs, s, e))
        return {serial: ivs for serial, ivs in state.items() if ivs}

    def _downstream_lots(self, lot_id):
        seen = {lot_id}
        queue = [lot_id]
        while queue:
            nxt = []
            for lot in queue:
                rows = self._conn.execute(
                    "SELECT lot_id FROM lot_sources WHERE source = ?", (lot,)
                )
                for (derived,) in rows:
                    if derived not in seen:
                        seen.add(derived)
                        nxt.append(derived)
            queue = nxt
        return seen

    def backward_trace(self, serial_id, at):
        """Assembly tree of *serial_id* at instant *at*.

        Returns a nested dict; each node has:
          serial   -- this serial number
          lots     -- sorted lots consumed into this serial at t <= at
          upstream -- sorted strict ancestor lots of `lots`
          children -- subtrees installed in this serial at *at*
        """
        root = {"serial": serial_id, "children": []}
        nodes = [root]
        index = 0
        while index < len(nodes):  # iterative: safe for deep trees
            node = nodes[index]
            index += 1
            rows = self._conn.execute(
                "SELECT child FROM installs WHERE parent = ? "
                "AND start <= ? AND (\"end\" IS NULL OR \"end\" > ?) "
                "ORDER BY child",
                (node["serial"], at, at),
            )
            for (child,) in rows:
                sub = {"serial": child, "children": []}
                node["children"].append(sub)
                nodes.append(sub)
        for node in nodes:
            lots = sorted(
                row[0]
                for row in self._conn.execute(
                    "SELECT DISTINCT lot_id FROM consumes "
                    "WHERE serial_id = ? AND t <= ?",
                    (node["serial"], at),
                )
            )
            node["lots"] = lots
            node["upstream"] = self._upstream_lots(lots)
        return root

    def _upstream_lots(self, lot_ids):
        seen = set()
        queue = list(lot_ids)
        while queue:
            nxt = []
            for lot in queue:
                rows = self._conn.execute(
                    "SELECT source FROM lot_sources WHERE lot_id = ?", (lot,)
                )
                for (source,) in rows:
                    if source not in seen and source not in lot_ids:
                        seen.add(source)
                        nxt.append(source)
            queue = nxt
        return sorted(seen)


# ----------------------------------------------------------------------
# interval-set helpers
#
# An interval set is a sorted list of (start, end) tuples, pairwise
# disjoint and non-adjacent.  end=None means +infinity ("still open").
# starts may be float("-inf") in internal computations.
# ----------------------------------------------------------------------
def _merge(intervals):
    items = sorted(
        intervals, key=lambda iv: (iv[0], iv[1] is None, iv[1] or 0)
    )
    if not items:
        return []
    merged = []
    cur_s, cur_e = items[0]
    for s, e in items[1:]:
        if cur_e is None:
            break  # current interval already runs to +inf
        if s <= cur_e:  # overlapping or touching -> merge
            if e is None:
                cur_e = None
            elif e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _clip(intervals, start, end):
    """Intersect an interval set with the half-open [start, end)."""
    out = []
    for a, b in intervals:
        lo = a if a > start else start
        hi = b
        if end is not None and (hi is None or hi > end):
            hi = end
        if hi is None or lo < hi:
            out.append((lo, hi))
    return out
