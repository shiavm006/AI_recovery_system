from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import NamedTuple

from models import LedgerEntry

GENESIS_PREV_HASH = "0" * 64

# Rows are missing: the chain is shorter than the tip says it should be, or it
# does not end where the tip says it ends.
TRUNCATED = "truncated"
# The rows present contradict each other: an edited payload, a broken link, or
# a row spliced in.
TAMPERED = "tampered"
# Entries exist but nothing anchors them. Either the tip was deleted, or the
# file predates the tip table, and neither can be vouched for.
MISSING_TIP = "missing_tip"


class VerifyResult(NamedTuple):
    """``failure`` is None when ok. ``seq`` is the bad row for TAMPERED, or the
    expected tip seq for TRUNCATED.
    """

    ok: bool
    failure: str | None = None
    seq: int | None = None


def _compute_entry_hash(
    seq: int,
    timestamp: datetime,
    event_id: str,
    entry_type: str,
    payload: dict,
    prev_hash: str,
) -> str:
    material = {
        "seq": seq,
        "timestamp": timestamp,
        "event_id": event_id,
        "entry_type": entry_type,
        "payload": payload,
        "prev_hash": prev_hash,
    }
    encoded = json.dumps(material, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


class Ledger:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        # Autocommit mode, so append can drive its own BEGIN IMMEDIATE rather
        # than letting sqlite3 open an implicit transaction around part of it.
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
              seq INTEGER PRIMARY KEY,
              timestamp TEXT NOT NULL,
              event_id TEXT NOT NULL,
              entry_type TEXT NOT NULL,
              payload TEXT NOT NULL,
              prev_hash TEXT NOT NULL,
              entry_hash TEXT NOT NULL
            )
            """
        )
        # id is pinned to 1 by the CHECK, so the table holds one row at most
        # and "the tip" is never ambiguous.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chain_tip (
              id INTEGER PRIMARY KEY CHECK (id = 1),
              seq INTEGER NOT NULL,
              entry_hash TEXT NOT NULL
            )
            """
        )

    def _read_tip(self) -> tuple[int, str] | None:
        row = self._conn.execute("SELECT seq, entry_hash FROM chain_tip").fetchone()
        return (int(row[0]), row[1]) if row is not None else None

    def append(self, event_id: str, entry_type: str, payload: dict) -> LedgerEntry:
        """Append one entry and advance the tip, atomically.

        The predecessor comes from the tip, never from the entries table. If it
        came from the entries table, wiping the table would silently restart
        the chain at seq 1 with a genesis link and the result would verify
        clean. Anchoring to the tip means a wipe leaves a chain whose length
        cannot match, which is exactly what :meth:`verify` looks for.
        """
        # IMMEDIATE takes the write lock up front, so the entry and the tip
        # move together or not at all; a reader can never see one without the
        # other, and two appenders cannot interleave onto the same seq.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            tip = self._read_tip()
            prev_hash = tip[1] if tip is not None else GENESIS_PREV_HASH
            seq = (tip[0] if tip is not None else 0) + 1

            timestamp = datetime.now(timezone.utc)
            entry_hash = _compute_entry_hash(
                seq=seq,
                timestamp=timestamp,
                event_id=event_id,
                entry_type=entry_type,
                payload=payload,
                prev_hash=prev_hash,
            )

            self._conn.execute(
                """
                INSERT INTO entries (seq, timestamp, event_id, entry_type, payload, prev_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seq,
                    timestamp.isoformat(),
                    event_id,
                    entry_type,
                    json.dumps(payload, sort_keys=True, default=str),
                    prev_hash,
                    entry_hash,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO chain_tip (id, seq, entry_hash) VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET seq = excluded.seq, entry_hash = excluded.entry_hash
                """,
                (seq, entry_hash),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

        return LedgerEntry(
            seq=seq,
            timestamp=timestamp,
            event_id=event_id,
            entry_type=entry_type,
            payload=payload,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )

    def verify(self) -> VerifyResult:
        """Check that the chain is intact and that all of it is still here.

        Structure is checked before content: the tip says how long the chain is
        and where it ends, so a missing row is caught by the anchor even though
        the rows that remain would link up perfectly on their own. Only then is
        each link recomputed.
        """
        rows = self._conn.execute(
            """
            SELECT seq, timestamp, event_id, entry_type, payload, prev_hash, entry_hash
            FROM entries
            ORDER BY seq ASC
            """
        ).fetchall()
        tip = self._read_tip()

        if tip is None:
            # Genesis is the only state without a tip, and it has no entries.
            if rows:
                return VerifyResult(False, MISSING_TIP, int(rows[-1][0]))
            return VerifyResult(True)

        tip_seq, tip_hash = tip
        # seq is assigned from the tip and starts at 1, so an intact chain of
        # length n ends at seq n and has exactly n rows. Either mismatch means
        # rows were removed, including the case where the table was emptied and
        # then appended to.
        if len(rows) != tip_seq:
            return VerifyResult(False, TRUNCATED, tip_seq)
        if int(rows[-1][0]) != tip_seq or rows[-1][6] != tip_hash:
            return VerifyResult(False, TRUNCATED, tip_seq)

        expected_prev_hash = GENESIS_PREV_HASH
        for (
            seq,
            timestamp_raw,
            event_id,
            entry_type,
            payload_raw,
            prev_hash,
            entry_hash,
        ) in rows:
            if prev_hash != expected_prev_hash:
                return VerifyResult(False, TAMPERED, int(seq))

            timestamp = _parse_timestamp(timestamp_raw)
            payload = json.loads(payload_raw)
            recomputed_hash = _compute_entry_hash(
                seq=int(seq),
                timestamp=timestamp,
                event_id=event_id,
                entry_type=entry_type,
                payload=payload,
                prev_hash=prev_hash,
            )
            if recomputed_hash != entry_hash:
                return VerifyResult(False, TAMPERED, int(seq))

            expected_prev_hash = entry_hash

        return VerifyResult(True)

    def read_all(self) -> list[LedgerEntry]:
        rows = self._conn.execute(
            """
            SELECT seq, timestamp, event_id, entry_type, payload, prev_hash, entry_hash
            FROM entries
            ORDER BY seq ASC
            """
        ).fetchall()

        return [
            LedgerEntry(
                seq=int(seq),
                timestamp=_parse_timestamp(timestamp_raw),
                event_id=event_id,
                entry_type=entry_type,
                payload=json.loads(payload_raw),
                prev_hash=prev_hash,
                entry_hash=entry_hash,
            )
            for seq, timestamp_raw, event_id, entry_type, payload_raw, prev_hash, entry_hash in rows
        ]


if __name__ == "__main__":
    import os

    if os.path.exists("test.db"):
        os.remove("test.db")
    ledger = Ledger("test.db")
    entries = [
        ledger.append("evt-1", "failure", {"payment_id": "pay_1", "amount_paise": 49900}),
        ledger.append(
            "evt-1",
            "diagnosis",
            {"cause": "INSUFFICIENT_FUNDS", "confidence": 0.92},
        ),
        ledger.append(
            "evt-1",
            "action",
            {"action": "RETRY", "scheduled_in_hours": 24},
        ),
    ]

    for entry in entries:
        print(entry)

    print(ledger.verify())
