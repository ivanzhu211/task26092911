"""Assembly genealogy store for a manufacturing execution system.

Persistent, SQLite-backed genealogy of lots and serial-numbered assemblies.
All times are integers; intervals are left-closed, right-open; ``None`` as
an interval end means "still installed / still contained".
"""

import sqlite3


def _covered(ivs, lo, hi):
    """True if [lo, hi) is fully covered by the union of `ivs`."""
    cur = lo
    for s, e in sorted(ivs, key=lambda x: x[0]):
        if s > cur:
            return False
        if e is None:
            return True
        if e > cur:
            cur = e
        if hi is not None and cur >= hi:
            return True
    return hi is not None and cur >= hi


def _merge(ivs):
    """Sort and merge overlapping or touching [start, end) intervals."""
    out = []
    for s, e in sorted(ivs, key=lambda x: x[0]):
        if out and (out[-1][1] is None or s <= out[-1][1]):
            if out[-1][1] is not None and (e is None or e > out[-1][1]):
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _clip(ivs, lo, hi):
    """Intersect every interval with [lo, hi)."""
    out = []
    for s, e in ivs:
        a = s if s > lo else lo
        b = e if hi is None else (hi if e is None else min(e, hi))
        if b is None or a < b:
            out.append((a, b))
    return out


class GenealogyError(ValueError):
    """Raised when a write violates genealogy invariants."""


class GenealogyStore:
    def __init__(self, path):
        self._conn = sqlite3.connect(path)
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lots (
                lot_id PRIMARY KEY,
                t INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lot_sources (
                lot_id,
                source,
                PRIMARY KEY (lot_id, source)
            );
            CREATE INDEX IF NOT EXISTS idx_sources_src
                ON lot_sources(source);
            CREATE TABLE IF NOT EXISTS consumption (
                lot_id,
                serial_id,
                t INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cons_lot
                ON consumption(lot_id);
            CREATE INDEX IF NOT EXISTS idx_cons_serial
                ON consumption(serial_id);
            CREATE TABLE IF NOT EXISTS installs (
                child,
                parent,
                start INTEGER NOT NULL,
                end INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_inst_child
                ON installs(child);
            CREATE INDEX IF NOT EXISTS idx_inst_parent
                ON installs(parent);
            """
        )
        self._conn.commit()

    def close(self):
        self._conn.close()

    def register_lot(self, lot_id, t, sources=()):
        sources = tuple(dict.fromkeys(sources))
        with self._conn:
            row = self._conn.execute(
                "SELECT t FROM lots WHERE lot_id = ?", (lot_id,)
            ).fetchone()
            if row is not None:
                raise GenealogyError(f"lot {lot_id!r} already registered")
            for src in sources:
                if src == lot_id:
                    raise GenealogyError(f"lot {lot_id!r} is its own source")
                srow = self._conn.execute(
                    "SELECT t FROM lots WHERE lot_id = ?", (src,)
                ).fetchone()
                if srow is None:
                    raise GenealogyError(f"source lot {src!r} not registered")
                if srow[0] > t:
                    raise GenealogyError(
                        f"source lot {src!r} registered at {srow[0]} > {t}"
                    )
            self._conn.execute(
                "INSERT INTO lots (lot_id, t) VALUES (?, ?)", (lot_id, t)
            )
            for src in sources:
                self._conn.execute(
                    "INSERT INTO lot_sources (lot_id, source) VALUES (?, ?)",
                    (lot_id, src),
                )

    def consume(self, lot_id, serial_id, t):
        with self._conn:
            row = self._conn.execute(
                "SELECT t FROM lots WHERE lot_id = ?", (lot_id,)
            ).fetchone()
            if row is None:
                raise GenealogyError(f"lot {lot_id!r} not registered")
            if t < row[0]:
                raise GenealogyError(
                    f"consumed at {t} before lot registered at {row[0]}"
                )
            self._conn.execute(
                "INSERT INTO consumption (lot_id, serial_id, t)"
                " VALUES (?, ?, ?)",
                (lot_id, serial_id, t),
            )

    def install(self, child, parent, start, end=None):
        if child == parent:
            raise GenealogyError("child and parent must differ")
        if end is not None and end <= start:
            raise GenealogyError(f"empty install interval [{start}, {end})")
        with self._conn:
            clash = self._conn.execute(
                "SELECT 1 FROM installs WHERE child = ?"
                " AND (? IS NULL OR start < ?)"
                " AND (end IS NULL OR end > ?) LIMIT 1",
                (child, end, end, start),
            ).fetchone()
            if clash is not None:
                raise GenealogyError(
                    f"{child!r} already installed during [{start}, {end})"
                )
            if self._reachable(child, parent, start, end):
                raise GenealogyError(
                    f"installing {child!r} under {parent!r} creates a cycle"
                )
            self._conn.execute(
                "INSERT INTO installs (child, parent, start, end)"
                " VALUES (?, ?, ?, ?)",
                (child, parent, start, end),
            )

    def _reachable(self, target, origin, start, end):
        """True if `target` is an ancestor of `origin` at some time in
        [start, end) via already-accepted installs (i.e. adding
        target->origin over that interval would close a cycle)."""
        reach = {origin: [(start, end)]}
        stack = [origin]
        while stack:
            node = stack.pop()
            rows = self._conn.execute(
                "SELECT parent, start, end FROM installs WHERE child = ?",
                (node,),
            ).fetchall()
            for p2, s2, e2 in rows:
                for rs, re in reach[node]:
                    lo = s2 if s2 > rs else rs
                    hi = e2 if re is None else (
                        re if e2 is None else min(e2, re)
                    )
                    if hi is not None and lo >= hi:
                        continue
                    if p2 == target:
                        return True
                    known = reach.get(p2)
                    if known is None:
                        reach[p2] = [(lo, hi)]
                        stack.append(p2)
                    elif not _covered(known, lo, hi):
                        known.append((lo, hi))
                        stack.append(p2)
        return False

    def remove(self, child, parent, t):
        with self._conn:
            row = self._conn.execute(
                "SELECT rowid, start FROM installs"
                " WHERE child = ? AND parent = ? AND end IS NULL",
                (child, parent),
            ).fetchone()
            if row is None:
                raise GenealogyError(
                    f"no open install of {child!r} under {parent!r}"
                )
            if t < row[1]:
                raise GenealogyError(
                    f"removal at {t} before install start {row[1]}"
                )
            if t == row[1]:
                # Zero-length install: keep no record at all so it can
                # never conflict with anything.
                self._conn.execute(
                    "DELETE FROM installs WHERE rowid = ?", (row[0],)
                )
            else:
                self._conn.execute(
                    "UPDATE installs SET end = ? WHERE rowid = ?",
                    (t, row[0]),
                )

    def forward_trace(self, lot_id):
        row = self._conn.execute(
            "SELECT 1 FROM lots WHERE lot_id = ?", (lot_id,)
        ).fetchone()
        if row is None:
            raise GenealogyError(f"lot {lot_id!r} not registered")

        # All lots whose material includes this lot (itself + downstream
        # derived lots), via the derivation DAG.
        lots = {lot_id}
        frontier = [lot_id]
        while frontier:
            cur = frontier.pop()
            for (derived,) in self._conn.execute(
                "SELECT lot_id FROM lot_sources WHERE source = ?", (cur,)
            ):
                if derived not in lots:
                    lots.add(derived)
                    frontier.append(derived)

        # Seed: a serial contains the material from consumption time on.
        held = {}
        for lot in lots:
            for serial, t in self._conn.execute(
                "SELECT serial_id, t FROM consumption WHERE lot_id = ?",
                (lot,),
            ):
                held.setdefault(serial, []).append((t, None))

        # Propagate containment up through installs until fixpoint. Each
        # serial's interval set only grows, endpoints come from a finite
        # set of record boundaries, so this terminates.
        queue = list(held)
        enqueued = set(queue)
        while queue:
            serial = queue.pop()
            enqueued.discard(serial)
            mine = held[serial]
            rows = self._conn.execute(
                "SELECT parent, start, end FROM installs WHERE child = ?",
                (serial,),
            ).fetchall()
            for parent, s, e in rows:
                add = _clip(mine, s, e)
                if not add:
                    continue
                before = held.get(parent)
                merged = _merge((before or []) + add)
                if before is None or merged != _merge(before):
                    held[parent] = merged
                    if parent not in enqueued:
                        queue.append(parent)
                        enqueued.add(parent)

        return {
            serial: [tuple(iv) for iv in _merge(ivs)]
            for serial, ivs in held.items()
            if ivs
        }

    def backward_trace(self, serial_id, at):
        def build(serial):
            children = [
                row[0]
                for row in self._conn.execute(
                    "SELECT child FROM installs"
                    " WHERE parent = ? AND start <= ?"
                    " AND (end IS NULL OR end > ?)",
                    (serial, at, at),
                )
            ]
            direct = {
                row[0]
                for row in self._conn.execute(
                    "SELECT lot_id FROM consumption"
                    " WHERE serial_id = ? AND t <= ?",
                    (serial, at),
                )
            }
            # Direct lots plus every upstream lot in the derivation DAG.
            lots = set(direct)
            frontier = list(direct)
            while frontier:
                cur = frontier.pop()
                for (src,) in self._conn.execute(
                    "SELECT source FROM lot_sources WHERE lot_id = ?", (cur,)
                ):
                    if src not in lots:
                        lots.add(src)
                        frontier.append(src)
            return {
                "serial": serial,
                "lots": sorted(lots, key=repr),
                "children": [build(c) for c in sorted(children, key=repr)],
            }

        return build(serial_id)
