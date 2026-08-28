"""Append-only hash-chained ledger backed by SQLite."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from models import LedgerEntry

GENESIS_PREV_HASH = "0" * 64


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
        self._conn = sqlite3.connect(db_path)
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
        self._conn.commit()

    def append(self, event_id: str, entry_type: str, payload: dict) -> LedgerEntry:
        row = self._conn.execute(
            "SELECT entry_hash FROM entries ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        prev_hash = row[0] if row is not None else GENESIS_PREV_HASH

        seq_row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM entries"
        ).fetchone()
        seq = int(seq_row[0]) + 1

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
        self._conn.commit()

        return LedgerEntry(
            seq=seq,
            timestamp=timestamp,
            event_id=event_id,
            entry_type=entry_type,
            payload=payload,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )

    def verify(self) -> tuple[bool, int | None]:
        rows = self._conn.execute(
            """
            SELECT seq, timestamp, event_id, entry_type, payload, prev_hash, entry_hash
            FROM entries
            ORDER BY seq ASC
            """
        ).fetchall()

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
                return False, int(seq)

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
                return False, int(seq)

            expected_prev_hash = entry_hash

        return True, None

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
