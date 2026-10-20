"""
tests/test_features.py
======================
Tests for the as-of rule in src/features.py. Run from the project folder with either:

    python tests/test_features.py
    pytest tests

Part A builds a tiny world by hand, so every expected number can be checked with your eyes.
Part B is the strong test: take the real generated data, CUT IT OFF at a scoring date as the
system would have seen it that day, rebuild the features, and require they are identical to
the ones built from the full data. If any feature peeks at the future, this fails.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as F  # noqa: E402

T = pd.Timestamp
DATA_START = T("2024-01-01")
PULL = T("2025-06-01")


# --------------------------------------------------------------------------- tiny world
def tiny_world():
    """One org, one account (acc_1) that renews on 2025-03-31. Scored on 2025-02-01 and 2025-03-01."""
    ev = pd.DataFrame([
        # id,   account,  event_time,        received_at
        ("e1", "acc_1", "2025-01-20 10:00", "2025-01-20 10:05"),   # normal
        ("e2", "acc_1", "2025-01-25 10:00", "2025-02-10 09:00"),   # LATE: happened before 1 Feb, arrived after
        ("e3", "acc_1", "2025-01-20 10:00", "2025-01-21 08:00"),   # duplicate of e1 (same id below)
        ("e4", "acc_1", "2025-02-05 10:00", "2025-02-05 10:01"),   # after the 1 Feb scoring date
        ("e5", None,    "2025-01-22 10:00", "2025-01-22 10:01"),   # orphan: no account
        ("e6", "acc_X", "2025-01-22 10:00", "2025-01-22 10:01"),   # ghost: unknown account
    ], columns=["event_id", "account_id", "event_time", "received_at"])
    ev.loc[2, "event_id"] = "e1"                                    # make e3 a re-delivery of e1
    ev["org_id"], ev["user_id"], ev["event_type"] = "org_1", "u1", "login"
    for c in ("event_time", "received_at"):
        ev[c] = pd.to_datetime(ev[c])

    tk = pd.DataFrame([("t1", "acc_1", "2025-01-15 09:00", "2025-02-15 09:00", "2025-01-15 09:05", True)],
                      columns=["ticket_id", "account_id", "opened_at", "closed_at", "received_at", "escalated"])
    tk["org_id"], tk["priority"] = "org_1", "high"
    for c in ("opened_at", "closed_at", "received_at"):
        tk[c] = pd.to_datetime(tk[c])

    inv = pd.DataFrame([("i1", "acc_1", "2025-01-28", "2025-01-29", None, "2025-01-28 03:00", "failed", 3),
                        ("i2", "acc_1", "2025-02-20", "2025-02-21", "2025-02-27", "2025-02-20 03:00", "paid", 2)],
                       columns=["invoice_id", "account_id", "invoice_date", "failed_at", "paid_at",
                                "received_at", "status", "attempts"])
    inv["org_id"], inv["amount"] = "org_1", 1000.0
    for c in ("invoice_date", "failed_at", "paid_at", "received_at"):
        inv[c] = pd.to_datetime(inv[c])

    gm = pd.DataFrame([("m1", "acc_1", "outbound", "2025-01-10 10:00", "2025-01-10 10:01"),
                       ("m2", "acc_1", "inbound", "2025-01-11 10:00", "2025-01-11 10:01")],
                      columns=["message_id", "account_id", "direction", "sent_at", "received_at"])
    gm["org_id"] = "org_1"
    for c in ("sent_at", "received_at"):
        gm[c] = pd.to_datetime(gm[c])

    acc = pd.DataFrame([
        ("acc_1", "org_1", "Cobalt Forge Inc", " Growth", "monthly", 20, 20000.0, 20000.0, "2024-03-31", "csm_1"),
        # same company entered twice: upper case, trailing dot, almost empty. Must be removed.
        ("acc_d1", "org_1", "COBALT FORGE INC.", None, None, 20, np.nan, np.nan, "2024-03-31", None),
    ], columns=["account_id", "org_id", "account_name", "plan", "billing_cycle", "seats", "initial_arr", "arr",
                "customer_since", "owner_id"])
    acc["customer_since"] = pd.to_datetime(acc["customer_since"])
    acc["contract_start"] = acc["contract_end"] = acc["customer_since"]

    own = pd.DataFrame([("acc_1", "org_1", "csm_1", "2024-03-31", None)],
                       columns=["account_id", "org_id", "owner_id", "valid_from", "valid_to"])
    own["valid_from"] = pd.to_datetime(own["valid_from"])
    own["valid_to"] = pd.to_datetime(own["valid_to"])

    ren = pd.DataFrame([("acc_1", "org_1", "2025-03-31", "churned", "lapsed", "2025-03-31", "2025-04-05", 20000.0, np.nan)],
                       columns=["account_id", "org_id", "renewal_date", "outcome", "churn_kind", "outcome_date",
                                "outcome_recorded_at", "arr_before", "arr_after"])
    for c in ("renewal_date", "outcome_date", "outcome_recorded_at"):
        ren[c] = pd.to_datetime(ren[c])

    t = dict(events=ev, tickets=tk, mails=gm, invoices=inv, accounts=acc, owners=own, renewals=ren)
    return F.clean(t)


def row(t, asof):
    df = F.build_features(t, [T(asof)], DATA_START, PULL)
    assert len(df) == 1, f"expected one row for acc_1 at {asof}, got {len(df)}"
    return df.iloc[0]


def test_cleaning():
    t, rep = tiny_world()
    assert rep["crm_duplicate_accounts_removed"] == 1
    assert list(t["accounts"]["account_id"]) == ["acc_1"]
    assert rep["events_duplicate_rows_removed"] == 1
    assert rep["events_orphan_rows_removed"] == 1
    assert rep["events_unknown_account_rows_removed"] == 1
    assert t["accounts"]["plan"].iloc[0] == "growth"            # " Growth" normalised


def test_late_event_is_not_known_yet():
    t, _ = tiny_world()
    r = row(t, "2025-02-01")
    # known on 1 Feb: e1 (once, duplicate ignored). NOT e2 (arrived 10 Feb), NOT e4 (happens 5 Feb).
    assert r["events_270d"] == 1, r["events_270d"]
    later = row(t, "2025-03-01")
    assert later["events_270d"] == 3, later["events_270d"]     # e1, e2, e4 all known by 1 Mar


def test_ticket_open_then_closed():
    t, _ = tiny_world()
    assert row(t, "2025-02-01")["open_tix"] == 1               # closes on 15 Feb, still open on 1 Feb
    assert row(t, "2025-03-01")["open_tix"] == 0
    assert row(t, "2025-02-01")["tix_30d"] == 1


def test_payment_fields_respect_their_own_dates():
    t, _ = tiny_world()
    r = row(t, "2025-02-01")
    assert r["pay_fail_120d"] == 1                              # i1 failed 29 Jan; i2 does not exist yet
    assert r["unpaid_invoices"] == 1                            # i1 never paid
    r2 = row(t, "2025-03-01")
    assert r2["pay_fail_120d"] == 2
    assert r2["unpaid_invoices"] == 1                           # i2 was paid on 27 Feb, so it no longer counts


def test_email_and_labels():
    t, _ = tiny_world()
    r = row(t, "2025-02-01")
    assert r["csm_out_120d"] == 1 and r["csm_in_120d"] == 1 and r["reply_rate_120d"] == 1.0
    assert r["days_to_renewal"] == 58
    assert r["label"] == 1.0 and r["label_status"] == "known"
    assert r["label_known_at"] == T("2025-04-05")               # a model trained on 1 Mar must not see this label
    assert r["n_prior_renewals"] == 0


def test_event_time_filter_alone_would_have_leaked():
    """Documents WHY received_at matters: the naive event_time-only count gives a different answer."""
    t, _ = tiny_world()
    ev = t["events"]
    naive = ev[(ev["event_time"] <= T("2025-02-01"))].drop_duplicates("event_id")
    right = ev[(ev["event_time"] <= T("2025-02-01")) & (ev["received_at"] <= T("2025-02-01"))]
    assert len(naive) == 2 and len(right) == 1


# --------------------------------------------------------------------------- real data
def truncate_to(raw, asof):
    """The raw data exactly as it stood on `asof`: later rows absent, later timestamps blanked."""
    asof = T(asof)
    r = {k: v.copy() for k, v in raw.items()}
    for k in ("events", "tickets", "mails", "invoices"):
        r[k] = r[k][r[k]["received_at"] <= asof].copy()
    for col in ("closed_at",):
        r["tickets"].loc[r["tickets"][col] > asof, col] = pd.NaT
    for col in ("failed_at", "paid_at"):
        r["invoices"].loc[r["invoices"][col] > asof, col] = pd.NaT
    ren = r["renewals"]
    late = ren["outcome_recorded_at"] > asof
    for col in ("outcome", "churn_kind", "outcome_date", "outcome_recorded_at"):
        ren.loc[late, col] = None if col in ("outcome", "churn_kind") else pd.NaT
    ren.loc[late, "arr_after"] = np.nan
    return r


def test_asof_invariance_on_generated_data():
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not (raw_dir / "product_events.csv").exists():
        print("  (skipped: run src/generate_data.py first)")
        return
    raw = F.load_raw(raw_dir)
    full, _ = F.clean(raw)
    data_start = full["events"]["event_time"].min().normalize()
    pull = max(full[k]["received_at"].max() for k in ("events", "tickets", "mails", "invoices")).ceil("D")
    for asof in ["2024-09-01", "2025-05-01", "2026-02-01"]:
        a = F.build_features(full, [T(asof)], data_start, pull)
        cut, _ = F.clean(truncate_to(raw, asof))
        b = F.build_features(cut, [T(asof)], data_start, T(asof))
        a, b = a.set_index("snapshot_id").sort_index(), b.set_index("snapshot_id").sort_index()
        assert list(a.index) == list(b.index), f"{asof}: different rows"
        cols = F.FEATURE_COLUMNS + ["data_state", "stale_source", "owner_id_asof"]
        for c in cols:
            x, y = a[c], b[c]
            if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
                ok = np.allclose(x.astype(float), y.astype(float), equal_nan=True)
            else:
                ok = (x.fillna("~") == y.fillna("~")).all()
            assert ok, f"{asof}: feature '{c}' changed when future data was removed (LEAK)"
        print(f"  {asof}: {len(a):,} rows, {len(cols)} columns identical with and without the future")


def test_labels_are_dated_and_pending_is_not_renewed():
    p = Path(__file__).resolve().parents[1] / "data" / "features" / "feature_snapshots.csv"
    if not p.exists():
        print("  (skipped: run src/features.py first)")
        return
    df = pd.read_csv(p, parse_dates=["as_of_date", "renewal_date", "label_known_at"])
    known = df[df["label_status"] == "known"]
    assert known["label"].isin([0.0, 1.0]).all()
    assert known["label_known_at"].notna().all()
    assert (known["label_known_at"] >= known["renewal_date"]).all()      # recorded on/after the renewal date
    assert df.loc[df["label_status"] != "known", "label"].isna().all()   # pending/open are NOT labelled "renewed"
    assert (df["renewal_date"] > df["as_of_date"]).all()
    assert ((df["renewal_date"] - df["as_of_date"]).dt.days <= F.HORIZON_DAYS).all()
    assert df["snapshot_id"].is_unique


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
