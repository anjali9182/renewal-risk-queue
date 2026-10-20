#!/usr/bin/env python
"""
features.py
===========
Turns the raw tables into the training / scoring table: ONE ROW per (account, scoring date,
upcoming renewal), using only what the system could have known on that date.

    python src/features.py                 # reads data/raw, writes data/features/

The one rule that matters
-------------------------
For a scoring date `asof`, a row of data may be used ONLY IF it had already ARRIVED:

        received_at <= asof     (and the thing it describes happened by asof too)

Filtering on event_time alone is the classic leak: an event that happened on 25 Sep but
reached the warehouse on 5 Oct looks "known" on 1 Oct when it was not. The same goes for
fields that are overwritten later (ticket closed_at, invoice paid_at / failed_at, renewal
outcomes): they are used only when their own timestamp is <= asof.

tests/test_features.py proves this with an as-of invariance test: cut the raw data off at
`asof`, rebuild the features, and they must be identical.

Row definition
--------------
A row exists for every renewal that falls in (asof, asof + 90 days]. Scoring dates are the
first of each month. An account renewing in 70 days appears at its last 3 scoring dates,
so it is scored again as new data arrives. The label (did it churn at that renewal?) is
attached, together with `label_known_at`, the date the CRM recorded the outcome. A model
trained "as of" some date may only use rows whose label_known_at is on or before it.

Assumptions (also in the README):
  * ticket `escalated` is treated as known when the ticket is opened (we have no escalation
    timestamp, so this is a small, documented risk);
  * plan, seats and billing cycle are static (the CRM only shows current state);
  * org-level source freshness (days since an org's last received event per source) is a
    feature, so an outage looks like "stale data", not like "customers went quiet".

Columns
-------
keys      snapshot_id, org_id, account_id, as_of_date, renewal_date, owner_id_asof
flags     data_state (ok | sparse | new), stale_source (bool)  -> for the CSM view, not the model
          ("new" can only occur if a renewal falls within 90 days of signup; a 12-month contract's
          first renewal is 275+ days after signup, so new accounts are simply not in the queue yet)
features  see FEATURE_COLUMNS
labels    label (1 churned, 0 renewed, blank = not known yet), label_known_at, label_status
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_VERSION = "v1"
HORIZON_DAYS = 90            # a row exists for renewals in (asof, asof + 90d]
MIN_HISTORY_DAYS = 90        # first scoring date = data start + this
MIN_EXPOSURE = 7             # a usage rate needs at least 7 observed days in its window
EPS = 0.01                   # events/day smoothing for the usage ratio
NEW_DAYS = 60                # tenure below this -> data_state "new"
SPARSE_EVENTS = 5            # fewer product events than this in the account's whole history -> "sparse"
STALE_DAYS = 7               # an org source silent this long -> stale_source flag
PLAN_CODE = {"starter": 0, "growth": 1, "enterprise": 2}

FEATURE_COLUMNS = [
    # contract
    "days_to_renewal", "contract_arr_log", "plan_code", "seats", "billing_monthly",
    "tenure_days", "n_prior_renewals",
    # product usage
    "ev_rate_30d", "ev_rate_31_90d", "ev_rate_91_270d", "usage_ratio_30_vs_90",
    "active_users_90d", "days_since_last_event", "events_270d",
    # support
    "tix_30d", "tix_90d", "esc_90d", "open_tix",
    # billing
    "pay_fail_120d", "unpaid_invoices",
    # CSM relationship
    "csm_out_120d", "csm_in_120d", "reply_rate_120d", "days_since_inbound",
    # ownership
    "has_owner", "owner_changed_180d",
    # source freshness (org level)
    "stale_product_days", "stale_tickets_days", "stale_gmail_days",
]
KEY_COLUMNS = ["snapshot_id", "org_id", "account_id", "as_of_date", "renewal_date", "owner_id_asof"]
FLAG_COLUMNS = ["data_state", "stale_source"]
LABEL_COLUMNS = ["label", "label_known_at", "label_status"]


# =========================================================================== loading + cleaning
def load_raw(raw_dir) -> dict:
    raw = Path(raw_dir)

    def rd(name, dates):
        return pd.read_csv(raw / f"{name}.csv", parse_dates=list(dates))

    return dict(
        events=rd("product_events", ["event_time", "received_at"]),
        tickets=rd("zd_tickets", ["opened_at", "closed_at", "received_at"]),
        mails=rd("gmail_messages", ["sent_at", "received_at"]),
        invoices=rd("stripe_invoices", ["invoice_date", "failed_at", "paid_at", "received_at"]),
        accounts=rd("sf_accounts", ["customer_since", "contract_start", "contract_end"]),
        owners=rd("sf_owner_history", ["valid_from", "valid_to"]),
        renewals=rd("sf_renewals", ["renewal_date", "outcome_date", "outcome_recorded_at"]),
    )


def _name_key(s: pd.Series) -> pd.Series:
    """'COBALT FORGE INC.' and 'Cobalt Forge Inc ' are the same company."""
    return (s.fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
            .str.split().str.join(" "))


def clean(tables: dict):
    """Dedupe and repair the raw tables. Returns (clean_tables, report_of_what_was_removed)."""
    t = {k: v.copy() for k, v in tables.items()}
    rep = {}

    # --- CRM: duplicate company records (same org + same normalised name). Keep the fullest one.
    acc = t["accounts"]
    acc["plan"] = acc["plan"].str.strip().str.lower()
    acc["name_key"] = _name_key(acc["account_name"])
    acc["_filled"] = acc[["plan", "arr", "initial_arr", "owner_id"]].notna().sum(axis=1)
    acc = acc.sort_values(["org_id", "name_key", "_filled", "account_id"], ascending=[True, True, False, True])
    keep = acc.drop_duplicates(["org_id", "name_key"], keep="first")
    dropped = acc[~acc["account_id"].isin(keep["account_id"])]
    key_to_id = keep.set_index(["org_id", "name_key"])["account_id"]
    alias = {r.account_id: key_to_id[(r.org_id, r.name_key)] for r in dropped.itertuples()}
    rep["crm_duplicate_accounts_removed"] = len(dropped)
    t["accounts"] = keep.drop(columns=["name_key", "_filled"]).reset_index(drop=True)
    valid = set(t["accounts"]["account_id"])

    # --- event tables: earliest arrival wins; no orphans; no ghost accounts
    spec = {"events": "event_id", "tickets": "ticket_id", "mails": "message_id", "invoices": "invoice_id"}
    for name, idcol in spec.items():
        df = t[name].sort_values("received_at", kind="stable")
        n0 = len(df)
        df = df.drop_duplicates(idcol, keep="first")
        rep[f"{name}_duplicate_rows_removed"] = n0 - len(df)
        n1 = len(df)
        df = df[df["account_id"].notna()].copy()
        rep[f"{name}_orphan_rows_removed"] = n1 - len(df)
        df["account_id"] = df["account_id"].replace(alias)
        n2 = len(df)
        df = df[df["account_id"].isin(valid)]
        rep[f"{name}_unknown_account_rows_removed"] = n2 - len(df)
        t[name] = df.reset_index(drop=True)

    t["renewals"] = t["renewals"][t["renewals"]["account_id"].isin(valid)].reset_index(drop=True)
    t["owners"] = (t["owners"][t["owners"]["account_id"].isin(valid)]
                   .sort_values(["account_id", "valid_from"]).reset_index(drop=True))
    return t, rep


# =========================================================================== scoring dates
def scoring_dates(data_start, pull_date, min_history_days=MIN_HISTORY_DAYS):
    """First of each month from (data start + 90d) up to the pull date, plus the pull date itself."""
    first = pd.Timestamp(data_start) + pd.Timedelta(days=min_history_days)
    dates = list(pd.date_range(first, pd.Timestamp(pull_date), freq="MS"))
    if not dates or dates[-1] != pd.Timestamp(pull_date):
        dates.append(pd.Timestamp(pull_date))
    return dates


# =========================================================================== one scoring date
def _binned_counts(df, tcol, asof, edges, accounts):
    """Rows per account in [0,e1), [e1,e2), ... days before asof. Returns DataFrame indexed by account."""
    n_bins = len(edges) - 1
    if df.empty:
        return pd.DataFrame(0, index=accounts, columns=range(n_bins))
    days_ago = (asof - df[tcol]).dt.total_seconds() / 86400.0
    b = pd.cut(days_ago, bins=edges, right=False, labels=False)
    out = (pd.DataFrame({"account_id": df["account_id"].to_numpy(), "b": b.to_numpy()})
           .dropna().astype({"b": int}).groupby(["account_id", "b"]).size().unstack(fill_value=0))
    return out.reindex(index=accounts, columns=range(n_bins), fill_value=0).fillna(0)


def _days_since_last(df, tcol, asof, accounts):
    if df.empty:
        return pd.Series(np.nan, index=accounts)
    last = df.groupby("account_id")[tcol].max()
    return ((asof - last).dt.total_seconds() / 86400.0).reindex(accounts)


def build_snapshot(t: dict, asof, data_start, pull_date) -> pd.DataFrame:
    asof = pd.Timestamp(asof)
    ren, acc = t["renewals"], t["accounts"]
    day = pd.Timedelta(days=1)

    snap = ren[(ren["renewal_date"] > asof) & (ren["renewal_date"] <= asof + HORIZON_DAYS * day)].copy()
    if snap.empty:
        return pd.DataFrame()
    snap = snap.merge(acc[["account_id", "plan", "seats", "billing_cycle", "customer_since"]],
                      on="account_id", how="left")
    accounts = pd.Index(snap["account_id"])

    # ---- exposure: how many days of this account have we actually been able to observe?
    obs_start = snap["customer_since"].where(snap["customer_since"] > data_start, pd.Timestamp(data_start))
    obs_days = ((asof - obs_start).dt.days).clip(lower=0).to_numpy()
    tenure = ((asof - snap["customer_since"]).dt.days).clip(lower=0).to_numpy()

    def rate(count, lo, hi):
        exposure = np.clip(np.minimum(hi, obs_days) - lo, 0, None)
        c = np.asarray(count, dtype=float)
        return np.where(exposure >= MIN_EXPOSURE, c / np.maximum(exposure, 1), np.nan)

    # ---- product usage: only rows that had ARRIVED and HAPPENED by asof
    ev = t["events"]
    ev = ev[(ev["received_at"] <= asof) & (ev["event_time"] <= asof) & ev["account_id"].isin(accounts)]
    ev270 = ev[ev["event_time"] > asof - 270 * day]
    ec = _binned_counts(ev270, "event_time", asof, [0, 30, 90, 270], accounts)
    r30, r3190, r91270 = rate(ec[0], 0, 30), rate(ec[1], 30, 90), rate(ec[2], 90, 270)
    ev90 = ev270[ev270["event_time"] > asof - 90 * day]
    lifetime_events = ev.groupby("account_id").size().reindex(accounts).fillna(0).to_numpy()   # all history, as of asof
    users = ev90.dropna(subset=["user_id"]).groupby("account_id")["user_id"].nunique().reindex(accounts).fillna(0)

    # ---- support tickets
    tk = t["tickets"]
    tk = tk[(tk["received_at"] <= asof) & (tk["opened_at"] <= asof) & tk["account_id"].isin(accounts)]
    tk90 = tk[tk["opened_at"] > asof - 90 * day]
    tc = _binned_counts(tk90, "opened_at", asof, [0, 30, 90], accounts)
    esc = _binned_counts(tk90[tk90["escalated"] == True], "opened_at", asof, [0, 90], accounts)[0]  # noqa: E712
    recent_open = tk[(tk["opened_at"] > asof - 180 * day) & (tk["closed_at"].isna() | (tk["closed_at"] > asof))]
    open_tix = recent_open.groupby("account_id").size().reindex(accounts).fillna(0)

    # ---- billing: failed_at / paid_at count only if they had happened by asof
    inv = t["invoices"]
    inv = inv[(inv["received_at"] <= asof) & (inv["invoice_date"] <= asof) & inv["account_id"].isin(accounts)]
    failed = inv[(inv["failed_at"] <= asof) & (inv["failed_at"] > asof - 120 * day)]
    pay_fail = failed.groupby("account_id").size().reindex(accounts).fillna(0)
    unpaid = inv[(inv["failed_at"] <= asof) & (inv["paid_at"].isna() | (inv["paid_at"] > asof))
                 & (inv["invoice_date"] > asof - 365 * day)]
    unpaid_n = unpaid.groupby("account_id").size().reindex(accounts).fillna(0)

    # ---- CSM <-> customer email
    gm = t["mails"]
    gm = gm[(gm["received_at"] <= asof) & (gm["sent_at"] <= asof) & gm["account_id"].isin(accounts)]
    gm120 = gm[gm["sent_at"] > asof - 120 * day]
    out_n = gm120[gm120["direction"] == "outbound"].groupby("account_id").size().reindex(accounts).fillna(0)
    in_n = gm120[gm120["direction"] == "inbound"].groupby("account_id").size().reindex(accounts).fillna(0)
    reply = np.where(out_n.to_numpy() > 0, in_n.to_numpy() / np.maximum(out_n.to_numpy(), 1), np.nan)
    since_in = _days_since_last(gm[gm["direction"] == "inbound"], "sent_at", asof, accounts)

    # ---- previous renewals the CRM had already recorded by asof
    prior = ren[ren["outcome_recorded_at"] <= asof][["account_id", "renewal_date"]]
    m = snap[["account_id", "renewal_date"]].merge(prior, on="account_id", suffixes=("", "_p"))
    n_prior = m[m["renewal_date_p"] < m["renewal_date"]].groupby("account_id").size().reindex(accounts).fillna(0)

    # ---- who owned the account on asof (effective-dated history)
    ow = t["owners"][t["owners"]["account_id"].isin(accounts)]
    cur = ow[(ow["valid_from"] <= asof) & (ow["valid_to"].isna() | (ow["valid_to"] > asof))]
    cur = cur.sort_values("valid_from").drop_duplicates("account_id", keep="last").set_index("account_id")["owner_id"]
    owner_asof = pd.Series(accounts.map(cur), index=accounts)
    later = ow[ow.groupby("account_id").cumcount() > 0]
    changed = later[(later["valid_from"] <= asof) & (later["valid_from"] > asof - 180 * day)]
    changed_n = changed.groupby("account_id").size().reindex(accounts).fillna(0)

    # ---- org-level freshness: days since the org's last RECEIVED row, per source
    def stale(df, tcol):
        d = df[(df["received_at"] <= asof) & (df[tcol] <= asof)]
        last = d.groupby("org_id")[tcol].max()
        return ((asof - last).dt.total_seconds() / 86400.0)
    st_ev = snap["org_id"].map(stale(t["events"], "event_time")).to_numpy()
    st_tk = snap["org_id"].map(stale(t["tickets"], "opened_at")).to_numpy()
    st_gm = snap["org_id"].map(stale(t["mails"], "sent_at")).to_numpy()

    # ---- assemble
    contract = snap["arr_before"].to_numpy(dtype=float)
    days_since_ev = _days_since_last(ev, "event_time", asof, accounts).to_numpy()
    out = pd.DataFrame({
        "snapshot_id": (snap["account_id"] + "|" + asof.strftime("%Y-%m-%d") + "|"
                        + snap["renewal_date"].dt.strftime("%Y-%m-%d")).to_numpy(),
        "org_id": snap["org_id"].to_numpy(), "account_id": snap["account_id"].to_numpy(),
        "as_of_date": asof, "renewal_date": snap["renewal_date"].to_numpy(),
        "owner_id_asof": owner_asof.to_numpy(),
        "days_to_renewal": (snap["renewal_date"] - asof).dt.days.to_numpy(),
        "contract_arr_log": np.where(contract > 0, np.log(np.where(contract > 0, contract, 1)), np.nan),
        "plan_code": snap["plan"].map(PLAN_CODE).to_numpy(dtype=float),
        "seats": snap["seats"].to_numpy(dtype=float),
        "billing_monthly": np.where(snap["billing_cycle"].isna(), np.nan,
                                    (snap["billing_cycle"] == "monthly").astype(float)),
        "tenure_days": tenure, "n_prior_renewals": n_prior.to_numpy(),
        "ev_rate_30d": r30, "ev_rate_31_90d": r3190, "ev_rate_91_270d": r91270,
        "usage_ratio_30_vs_90": (r30 + EPS) / (r3190 + EPS),
        "active_users_90d": users.to_numpy(), "days_since_last_event": days_since_ev,
        "events_270d": ec.sum(axis=1).to_numpy(),
        "tix_30d": tc[0].to_numpy(), "tix_90d": (tc[0] + tc[1]).to_numpy(),
        "esc_90d": esc.to_numpy(), "open_tix": open_tix.to_numpy(),
        "pay_fail_120d": pay_fail.to_numpy(), "unpaid_invoices": unpaid_n.to_numpy(),
        "csm_out_120d": out_n.to_numpy(), "csm_in_120d": in_n.to_numpy(),
        "reply_rate_120d": reply, "days_since_inbound": since_in.to_numpy(),
        "has_owner": owner_asof.notna().astype(float).to_numpy(),
        "owner_changed_180d": (changed_n.to_numpy() > 0).astype(float),
        "stale_product_days": st_ev, "stale_tickets_days": st_tk, "stale_gmail_days": st_gm,
    })
    out["data_state"] = np.where(tenure < NEW_DAYS, "new",
                                 np.where(lifetime_events < SPARSE_EVENTS, "sparse", "ok"))
    out["stale_source"] = (np.nan_to_num(np.stack([st_ev, st_tk, st_gm]), nan=0.0).max(axis=0) >= STALE_DAYS)

    # ---- label (never a feature): the outcome and WHEN the CRM learned it
    out["label"] = snap["outcome"].map({"churned": 1.0, "renewed": 0.0}).to_numpy()
    out["label_known_at"] = snap["outcome_recorded_at"].to_numpy()
    # known = outcome recorded by the pull; pending = renewal date has passed but no outcome yet
    # (do NOT treat as "renewed"); open = the renewal is still in the future at pull time.
    out["label_status"] = np.where(out["label"].notna(), "known",
                                   np.where(snap["renewal_date"].to_numpy() <= np.datetime64(pd.Timestamp(pull_date)),
                                            "pending", "open"))
    return out


def build_features(t: dict, asofs, data_start, pull_date) -> pd.DataFrame:
    parts = [build_snapshot(t, a, data_start, pull_date) for a in asofs]
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True)
    df["feature_version"] = FEATURE_VERSION
    return df[KEY_COLUMNS + FLAG_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS + ["feature_version"]]


# =========================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--pull-date", default=None, help="default: day after the last received row")
    args = ap.parse_args()

    tables = load_raw(args.raw)
    t, rep = clean(tables)
    data_start = t["events"]["event_time"].min().normalize()
    last_recv = max(t[k]["received_at"].max() for k in ("events", "tickets", "mails", "invoices"))
    pull = pd.Timestamp(args.pull_date) if args.pull_date else last_recv.ceil("D")
    asofs = scoring_dates(data_start, pull)

    print(f"data starts {data_start.date()} | pulled {pull.date()} | {len(asofs)} scoring dates "
          f"({asofs[0].date()} .. {asofs[-1].date()})")
    print("cleaning removed:", {k: v for k, v in rep.items() if v})

    df = build_features(t, asofs, data_start, pull)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "feature_snapshots.csv", index=False)
    (out / "feature_meta.json").write_text(json.dumps(dict(
        feature_version=FEATURE_VERSION, features=FEATURE_COLUMNS, horizon_days=HORIZON_DAYS,
        data_start=str(data_start.date()), pull_date=str(pull.date()),
        scoring_dates=[str(a.date()) for a in asofs], rows=len(df), cleaning=rep), indent=2))

    known = df[df["label_status"] == "known"]
    print(f"\n{len(df):,} snapshot rows | {len(known):,} with a known label "
          f"({int(known['label'].sum())} churn, {known['label'].mean():.1%}) | "
          f"{(df['label_status'] == 'open').sum():,} open | {(df['label_status'] == 'pending').sum():,} pending")
    print("data_state:", df["data_state"].value_counts(normalize=True).round(3).to_dict())
    print(f"stale_source rows: {df['stale_source'].mean():.1%}")
    print("rows per org:", df["org_id"].value_counts().sort_index().to_dict())
    print("wrote", (out / "feature_snapshots.csv").resolve())


if __name__ == "__main__":
    main()
