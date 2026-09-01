from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger import MISSING_TIP, TAMPERED, TRUNCATED, Ledger

CHAIN_LENGTH = 10


@pytest.fixture
def chain(tmp_path) -> tuple[Ledger, str]:
    path = str(tmp_path / "chain.db")
    ledger = Ledger(path)
    for index in range(CHAIN_LENGTH):
        ledger.append(f"evt-{index}", "failure", {"amount_paise": 100 * index})
    return ledger, path


def _damage(path: str, sql: str, params: tuple = ()) -> None:
    """Edit the file behind the Ledger's back, the way an attacker would."""
    connection = sqlite3.connect(path)
    connection.execute(sql, params)
    connection.commit()
    connection.close()


def test_an_untouched_chain_verifies(chain):
    ledger, _ = chain
    result = ledger.verify()
    assert result.ok
    assert result.failure is None
    assert result.seq is None


def test_an_empty_ledger_verifies(tmp_path):
    # Genesis: no tip and no entries is the one consistent empty state.
    assert Ledger(str(tmp_path / "empty.db")).verify().ok


def test_a_mid_chain_payload_edit_is_caught_at_that_row(chain):
    ledger, path = chain
    _damage(
        path,
        "UPDATE entries SET payload = ? WHERE seq = 4",
        (json.dumps({"amount_paise": 999_999}),),
    )

    result = Ledger(path).verify()
    assert not result.ok
    assert result.failure == TAMPERED
    assert result.seq == 4


def test_a_relinked_row_is_caught_as_tampering(chain):
    # Rewriting a link rather than a payload is the subtler edit, and it has to
    # be caught by the walk rather than by the length check.
    ledger, path = chain
    _damage(path, "UPDATE entries SET prev_hash = ? WHERE seq = 6", ("0" * 64,))

    result = Ledger(path).verify()
    assert not result.ok
    assert result.failure == TAMPERED
    assert result.seq == 6


def test_tail_deletion_is_caught(chain):
    # The surviving rows still link perfectly to each other. Only the tip knows
    # the chain was longer than this.
    ledger, path = chain
    _damage(path, "DELETE FROM entries WHERE seq > 7")

    assert len(Ledger(path).read_all()) == 7
    result = Ledger(path).verify()
    assert not result.ok
    assert result.failure == TRUNCATED
    assert result.seq == CHAIN_LENGTH


def test_deleting_only_the_final_row_is_caught(chain):
    ledger, path = chain
    _damage(path, "DELETE FROM entries WHERE seq = ?", (CHAIN_LENGTH,))

    result = Ledger(path).verify()
    assert not result.ok
    assert result.failure == TRUNCATED


def test_a_full_wipe_is_caught(chain):
    # An empty entries table is a vacuously valid chain, which is exactly why
    # the tip has to outlive it.
    ledger, path = chain
    _damage(path, "DELETE FROM entries")

    assert Ledger(path).read_all() == []
    result = Ledger(path).verify()
    assert not result.ok
    assert result.failure == TRUNCATED
    assert result.seq == CHAIN_LENGTH


def test_appending_after_a_wipe_is_caught(chain):
    # The cover-up: empty the table, then append so the file looks alive again.
    # append() takes its predecessor from the tip, so the new row lands at
    # seq 11 in a table holding one row, and the count gives it away.
    ledger, path = chain
    _damage(path, "DELETE FROM entries")

    reopened = Ledger(path)
    reopened.append("evt-forged", "failure", {"amount_paise": 1})

    result = reopened.verify()
    assert not result.ok
    assert result.failure == TRUNCATED
    assert len(reopened.read_all()) == 1


def test_entries_without_a_tip_are_not_vouched_for(chain):
    ledger, path = chain
    _damage(path, "DELETE FROM chain_tip")

    result = Ledger(path).verify()
    assert not result.ok
    assert result.failure == MISSING_TIP


def test_a_stale_tip_is_caught(chain):
    # Rolling the tip backwards to match a truncated chain still fails, because
    # the row count no longer matches the seq the tip claims.
    ledger, path = chain
    _damage(path, "DELETE FROM entries WHERE seq > 7")
    _damage(path, "UPDATE chain_tip SET seq = 7")

    result = Ledger(path).verify()
    assert not result.ok
    assert result.failure == TRUNCATED


def test_a_datetime_payload_round_trips_through_verify(tmp_path):
    # run.py writes occurred_at and scheduled_for straight into payloads, so
    # the hash has to survive the datetime being serialised on the way in and
    # read back as a string.
    path = str(tmp_path / "dates.db")
    ledger = Ledger(path)
    occurred = datetime(2026, 9, 2, 14, 30, tzinfo=timezone.utc)
    entry = ledger.append(
        "evt-1",
        "ingested",
        {"occurred_at": occurred, "scheduled_for": None, "amount_paise": 49_900},
    )

    assert entry.payload["occurred_at"] is occurred
    assert Ledger(path).verify().ok

    stored = Ledger(path).read_all()[0]
    assert stored.payload["occurred_at"] == str(occurred)
    assert Ledger(path).verify().ok


def test_the_tip_tracks_every_append(tmp_path):
    path = str(tmp_path / "tip.db")
    ledger = Ledger(path)
    for index in range(1, 6):
        entry = ledger.append(f"evt-{index}", "failure", {"n": index})
        tip = sqlite3.connect(path).execute(
            "SELECT seq, entry_hash FROM chain_tip"
        ).fetchone()
        assert tip == (entry.seq, entry.entry_hash)
        assert entry.seq == index
        assert ledger.verify().ok


def test_the_tip_table_cannot_hold_a_second_row(chain):
    _, path = chain
    with pytest.raises(sqlite3.IntegrityError):
        _damage(path, "INSERT INTO chain_tip (id, seq, entry_hash) VALUES (2, 99, 'x')")
