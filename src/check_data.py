#!/usr/bin/env python
"""
check_data.py
=============
Sanity checks for the synthetic data. Run it after generate_data.py and again after
every change to the generator:

    python src/check_data.py

It answers "does the data look the way the design note says it should?" with
PASS / WARN lines, then prints the numbers behind them.

Important: this script reads data/_ground_truth/, which the feature and model code
must NEVER do. Here it is used on purpose, to judge the generator itself (for example:
"does a usage-drop signal catch the 'fade' churners but not the 'nowhere' ones?").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTAGES = [  # keep in sync with generate_data.py
    ("org_03", "product_events", "2024-11-04", "2024-11-18"),
    ("org_04", "zd_tickets", "2025-03-10", "2025-05-20"),
    ("org_05", "gmail_messages", "2025-08-01", "2025-10-15"),
]
results = []


def check(name, value, lo, hi, fmt="{:.3f}"):
    ok = lo <= value <= hi
    results.append(ok)
    print(f"  [{'PASS' if ok else 'WARN'}] {name}: {fmt.format(value)}   (expected {fmt.format(lo)} to {fmt.format(hi)})")


def auc(score, y):
    """Probability a random churner scores higher than a random non-churner (rank based)."""
    s = pd.Series(np.asarray(score, dtype=float))
    y = np.asarray(y, dtype=bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    r = s.rank(method="average").to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def head(title):
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    d = Path(ap.parse_args().data)
    raw, tr = d / "raw", d / "_ground_truth"

    ts = {"product_events": ["event_time", "received_at"], "zd_tickets": ["opened_at", "closed_at", "received_at"],
          "gmail_messages": ["sent_at", "received_at"],
          "stripe_invoices": ["invoice_date", "failed_at", "paid_at", "received_at"]}
    T = {k: pd.read_csv(raw / f"{k}.csv", parse_dates=v) for k, v in ts.items()}
    sf = pd.read_csv(raw / "sf_accounts.csv")
    hist = pd.read_csv(raw / "sf_owner_history.csv")
    ren = pd.read_csv(raw / "sf_renewals.csv", parse_dates=["renewal_date", "outcome_date", "outcome_recorded_at"])
    truth = pd.read_csv(tr / "account_truth.csv")
    ev, tk, gm, inv = T["product_events"], T["zd_tickets"], T["gmail_messages"], T["stripe_invoices"]
    END = pd.Timestamp("2026-10-01")

    # ------------------------------------------------------------------ 1. volume
    head("1. Volume")
    real = sf[~sf["account_id"].str.startswith("acc_d")]
    n_events = sum(len(v) for v in T.values())
    print(f"  accounts (real)      {len(real):>9,}   (+{len(sf) - len(real)} duplicate CRM records)")
    for k, v in T.items():
        print(f"  {k:<20} {len(v):>9,}")
    print(f"  {'salesforce rows':<20} {len(sf) + len(hist) + len(ren):>9,}")
    check("events across 4 systems (k)", n_events / 1000, 190, 260, "{:.0f}")
    print("\n  accounts per org:")
    print("   ", real["org_id"].value_counts().sort_index().to_dict())

    # ------------------------------------------------------------------ 2. mess
    head("2. The mess is really there")
    for name, df, tcol in [("product_events", ev, "event_time"), ("zd_tickets", tk, "opened_at"),
                           ("gmail_messages", gm, "sent_at"), ("stripe_invoices", inv, "invoice_date")]:
        lag = (df["received_at"] - df[tcol]).dt.total_seconds() / 86400
        idc = df.columns[0]
        dup = df[idc].duplicated().mean()
        print(f"  {name:<16} late>1d {np.mean(lag > 1):6.1%} | late>14d {np.mean(lag > 14):6.2%} | "
              f"duplicate ids {dup:6.2%} | arrival order != event order: "
              f"{(df[tcol].diff().dt.total_seconds() < 0).mean():5.1%} of rows step backwards")
    lag = (ev["received_at"] - ev["event_time"]).dt.total_seconds() / 86400
    check("product events arriving >1 day late", np.mean(lag > 1), 0.03, 0.08, "{:.1%}")
    check("duplicate product event ids", ev["event_id"].duplicated().mean(), 0.015, 0.025, "{:.2%}")
    check("product events with no user_id", ev["user_id"].isna().mean(), 0.02, 0.04, "{:.1%}")
    check("product events with no account (orphans)", ev["account_id"].isna().mean(), 0.002, 0.007, "{:.2%}")
    ghost = (~ev["account_id"].isna()) & (~ev["account_id"].isin(sf["account_id"]))
    check("events pointing at an unknown account", ghost.mean(), 0.0005, 0.002, "{:.2%}")
    check("tickets with no priority", tk["priority"].isna().mean(), 0.03, 0.12, "{:.1%}")
    check("accounts with blank plan", sf["plan"].isna().mean(), 0.005, 0.03, "{:.1%}")
    check("accounts with no current owner", sf["owner_id"].isna().mean(), 0.03, 0.12, "{:.1%}")
    plans = sf["plan"].dropna().unique()
    print(f"  plan spellings in the CRM: {sorted(map(repr, plans))[:8]}")
    print(f"  accounts with more than one owner in history: "
          f"{(hist.groupby('account_id').size() > 1).mean():.1%} of accounts that have history")

    print("\n  source outages (rows whose EVENT time is in the gap, and when they arrived):")
    srcmap = {"product_events": (ev, "event_time"), "zd_tickets": (tk, "opened_at"), "gmail_messages": (gm, "sent_at")}
    for org, src, g0, g1 in OUTAGES:
        df, col = srcmap[src]
        sub = df[(df["org_id"] == org) & (df[col] >= g0) & (df[col] < g1)]
        in_gap = (sub["received_at"] < g1).mean() if len(sub) else float("nan")
        burst = ((sub["received_at"] >= pd.Timestamp(g1)) & (sub["received_at"] < pd.Timestamp(g1) + pd.Timedelta(days=2))).mean() if len(sub) else float("nan")
        print(f"   {org} {src:<15} {len(sub):>6,} rows survive | arrived during gap {in_gap:5.1%} | in backfill burst {burst:5.1%}")

    # ------------------------------------------------------------------ 3. sparse accounts
    head("3. Sparse accounts")
    life = ev[ev["account_id"].isin(real["account_id"])].drop_duplicates("event_id").groupby("account_id").size()
    life = life.reindex(real["account_id"], fill_value=0)
    tenure = (END - pd.to_datetime(real["customer_since"]).clip(lower=pd.Timestamp("2023-10-01"))).dt.days.to_numpy()
    old_enough = tenure >= 180        # a 2-month-old account has had no time to generate 5 events
    print(f"  all accounts with <5 events: {(life < 5).mean():.1%}   (includes accounts that are simply new)")
    check("accounts >=180 days old with <5 product events in their whole life",
          (life.to_numpy()[old_enough] < 5).mean(), 0.15, 0.25, "{:.1%}")
    print("   by org (>=180d old):", (pd.Series(life.to_numpy()[old_enough] < 5)
          .groupby(real["org_id"].to_numpy()[old_enough]).mean().round(3).to_dict()))
    since = pd.to_datetime(real["customer_since"])
    print(f"  customers since before data starts: {(since < '2023-10-01').mean():.1%} | "
          f"started in last 6 months: {(since > END - pd.Timedelta(days=183)).mean():.1%} | "
          f"started 3y ago or more: {(since < END - pd.Timedelta(days=1000)).mean():.1%}")

    # ------------------------------------------------------------------ 4. labels
    head("4. Labels")
    known = ren[ren["outcome"].notna()].copy()
    known["y"] = (known["outcome"] == "churned").astype(int)
    print(f"  renewal rows {len(ren):,} | outcome known {len(known):,} | "
          f"upcoming/pending {ren['outcome'].isna().sum():,}")
    pend = ren[(ren["outcome"].isna()) & (ren["renewal_date"] < END)]
    print(f"  renewals already past but outcome not recorded yet at pull time: {len(pend)}  "
          f"(label maturity: do not treat these as 'renewed')")
    check("overall churn rate among known renewals", known["y"].mean(), 0.09, 0.14, "{:.1%}")
    by_org = known.groupby("org_id")["y"].agg(["size", "mean"]).round(3)
    print("\n  churn rate by org (known renewals):")
    print(by_org.rename(columns={"size": "renewals", "mean": "churn_rate"}).to_string())
    t = truth[truth["churned"] == True]  # noqa: E712
    share = t["churn_type"].value_counts(normalize=True)
    print("\n  churn types (ground truth):", share.round(3).to_dict())
    check("churn with no warning signs ('nowhere')", share.get("nowhere", 0), 0.20, 0.32, "{:.1%}")
    print(f"  churn kind: {known.loc[known['y'] == 1, 'churn_kind'].value_counts(normalize=True).round(2).to_dict()}")

    # churned accounts must go quiet after their renewal date
    cd = known[known["y"] == 1].set_index("account_id")["renewal_date"]
    e2 = ev.dropna(subset=["account_id"]).drop_duplicates("event_id")
    e2 = e2[e2["account_id"].isin(cd.index)]
    after = (e2["event_time"] > e2["account_id"].map(cd) + pd.Timedelta(days=1)).sum()
    check("product events after a churn date", after, 0, 0, "{:.0f}")

    # ------------------------------------------------------------------ 5. is the signal there?
    head("5. Is there signal, and is it imperfect? (as-of discipline: received_at <= scoring date)")
    L = known.copy()
    L["asof"] = L["renewal_date"] - pd.Timedelta(days=45)
    L = L[L["asof"] >= pd.Timestamp("2024-02-15")].reset_index(drop=True)
    L["rid"] = np.arange(len(L))
    print(f"  scoring each known renewal 45 days before its renewal date: {len(L):,} rows, {L['y'].sum()} churn")

    def win(df, tcol, older, newer, mask=None):
        d_ = df.dropna(subset=["account_id"]).drop_duplicates(df.columns[0])
        if mask is not None:
            d_ = d_[mask(d_)]
        m = d_[["account_id", tcol, "received_at"]].merge(L[["rid", "account_id", "asof"]], on="account_id")
        k = ((m[tcol] > m["asof"] - pd.Timedelta(days=older)) & (m[tcol] <= m["asof"] - pd.Timedelta(days=newer))
             & (m["received_at"] <= m["asof"]))
        return m[k].groupby("rid").size().reindex(L["rid"], fill_value=0).to_numpy()

    use_new, use_old = win(ev, "event_time", 45, 0), win(ev, "event_time", 135, 45)
    tix = win(tk, "opened_at", 60, 0)
    esc = win(tk, "opened_at", 60, 0, mask=lambda x: x["escalated"] == True)  # noqa: E712
    badpay = win(inv, "invoice_date", 120, 0, mask=lambda x: x["status"].isin(["failed"]) | (x["attempts"] > 1))
    outb = win(gm, "sent_at", 120, 0, mask=lambda x: x["direction"] == "outbound")
    inb = win(gm, "sent_at", 120, 0, mask=lambda x: x["direction"] == "inbound")
    sig = pd.DataFrame({
        "usage drop (45d vs prior 90d)": -(use_new + 1) / (use_old / 2 + 1),
        "ticket count, 60d": tix, "escalated tickets, 60d": esc,
        "failed/retried payments, 120d": badpay, "low CSM-email reply rate, 120d": -(inb / (outb + 1))})
    L = L.merge(truth[["account_id", "is_sparse", "churn_type"]], on="account_id", how="left")
    y = L["y"].to_numpy() == 1
    rows = []
    for c in sig.columns:
        r = {"signal": c, "AUC all": auc(sig[c], y)}
        for ct in ["loud", "fade", "nowhere"]:
            m = (~y) | (L["churn_type"] == ct).to_numpy()
            r[f"vs {ct}"] = auc(sig[c][m], y[m])
        m = ~L["is_sparse"].fillna(False).astype(bool).to_numpy()
        r["non-sparse only"] = auc(sig[c][m], y[m])
        rows.append(r)
    print(pd.DataFrame(rows).set_index("signal").round(2).to_string())
    print("  (0.50 = coin flip. 'fade' should light up on usage / unanswered emails, 'loud' on tickets,\n"
          "   'nowhere' should sit near 0.50 for every signal.)")
    best_nowhere = max(r["vs nowhere"] for r in rows)
    check("best single-signal AUC against 'nowhere' churners", best_nowhere, 0.40, 0.60, "{:.2f}")
    check("best single-signal AUC overall", max(r["AUC all"] for r in rows), 0.58, 0.80, "{:.2f}")

    # ------------------------------------------------------------------ 6. why as-of matters
    head("6. Why the as-of rule matters")
    asof = pd.Timestamp("2025-10-01")
    e = ev.drop_duplicates("event_id")
    happened = e[e["event_time"] <= asof]
    late = (happened["received_at"] > asof).mean()
    print(f"  events that had HAPPENED by {asof.date()} but had NOT ARRIVED yet: {late:.2%} of them")
    last30 = happened[happened["event_time"] > asof - pd.Timedelta(days=30)]
    print(f"  ...and in the last 30 days before that date: {(last30['received_at'] > asof).mean():.2%}")
    print("  Using event_time instead of received_at would quietly show the model data it could not have had.")
    still_open = tk["closed_at"].isna().mean()
    print(f"  tickets with no closed_at at pull time: {still_open:.1%} (closed_at is a current-state field:\n"
          f"  as of an earlier date, a ticket whose closed_at is later than that date was still OPEN)")

    # ------------------------------------------------------------------ summary
    head("Summary")
    print(f"  {sum(results)}/{len(results)} checks passed" + ("" if all(results) else "   -> look at the WARN lines above"))


if __name__ == "__main__":
    main()
