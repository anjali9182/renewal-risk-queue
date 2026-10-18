#!/usr/bin/env python
"""
generate_data.py
================
Synthetic, multi-tenant customer data for the renewal-risk project.

It produces ~5,000 accounts and ~200k events across five "systems":

    Salesforce  -> sf_accounts, sf_owner_history, sf_renewals
    Zendesk     -> zd_tickets
    Product     -> product_events
    Stripe      -> stripe_invoices
    Gmail       -> gmail_messages

Everything is driven by ONE seed, so the same command always gives the same data.

How the data is built (read this before reading the code)
---------------------------------------------------------
1. Each account gets a hidden risk score `z` (the model never sees it) plus ordinary
   attributes: org, plan, ARR, start date, billing cycle, how chatty it is, etc.
2. We walk the account's renewal dates in order. At each renewal that falls before the
   data cutoff we decide churn / renew from `z`. A churn ends the account.
3. The BEHAVIOUR of the account (usage, tickets, payments, email replies) is generated
   AFTER the outcome is known, and it is bent in the ~150 days before a churn:
       loud     -> tickets spike and escalate, mild usage drop
       fade     -> usage decays quietly, tickets go away, emails stop being answered
       nowhere  -> no warning signs at all (some churn is simply unpredictable)
   Healthy renewals sometimes get DECOYS (a ticket burst, a usage dip, a late payment),
   so a signal on its own is never a verdict.
4. Mess is added on top: late / out-of-order arrival, duplicates, missing fields,
   orphan events, sparse accounts, ownership changes, duplicate CRM records, and
   outages in one source for one org.
5. Rows are written in ARRIVAL order (received_at), not event order.

Files written
-------------
data/raw/*.csv                      what the pipeline is allowed to read
data/_ground_truth/account_truth.csv  hidden risk, churn type, decoys. For checking the
                                      generator and evaluating the model ONLY. The
                                      feature / training code must never read it.
data/_ground_truth/generation_params.json

All timestamps are UTC, written as 'YYYY-MM-DD HH:MM:SS'.

Usage
-----
    python src/generate_data.py                 # defaults: 5000 accounts, seed 42
    python src/generate_data.py --accounts 500  # quick run while developing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- calendar
START = pd.Timestamp("2023-10-01")      # first day of data (three years of history)
END = pd.Timestamp("2026-10-01")        # the day the data was "pulled": nothing arrives later
N_DAYS = int((END - START).days)
DAY = 86_400
END_SEC = N_DAYS * DAY
BASE = np.datetime64("2023-10-01T00:00:00")

# --------------------------------------------------------------------------- knobs
TARGET_FIRST_RENEWAL_CHURN = 0.12   # intercept is solved so first renewals churn at ~12%
DECOY_P = 0.14                      # share of healthy renewals that get a decoy pattern
ESC0 = 0.08                         # baseline share of tickets that get escalated
SIGNAL_LEAD = 150                   # days before renewal when a churn pattern starts
SIGNAL_FULL = 120                   # days over which it ramps to full strength

PLANS = ["starter", "growth", "enterprise"]
PLAN_ARR_MEDIAN = {"starter": 6_000, "growth": 26_000, "enterprise": 110_000}
PLAN_Z = {"starter": 0.35, "growth": 0.0, "enterprise": -0.45}
PLAN_SEATS = {"starter": (3, 15), "growth": (10, 60), "enterprise": (50, 400)}

# Customer organisations = the tenants that use this system. They differ on purpose.
ORGS = [
    dict(org_id="org_01", name="Northwind Cloud", share=0.30, event_mult=1.0, ticket_mult=1.0,
         late_mult=1.0, sparse_share=0.07, z_shift=0.00, plan_mix=(0.45, 0.40, 0.15),
         monthly_share=0.15, pay_fail=0.025, owner_null=0.03, reassign=0.15),
    dict(org_id="org_02", name="Helix Analytics", share=0.22, event_mult=1.8, ticket_mult=1.5,
         late_mult=1.0, sparse_share=0.04, z_shift=0.05, plan_mix=(0.30, 0.50, 0.20),
         monthly_share=0.12, pay_fail=0.020, owner_null=0.02, reassign=0.15),
    dict(org_id="org_03", name="Pinecrest Software", share=0.18, event_mult=0.6, ticket_mult=0.6,
         late_mult=1.5, sparse_share=0.09, z_shift=-0.05, plan_mix=(0.50, 0.40, 0.10),
         monthly_share=0.15, pay_fail=0.025, owner_null=0.04, reassign=0.15),
    dict(org_id="org_04", name="Orbit Logistics Tech", share=0.12, event_mult=1.2, ticket_mult=1.0,
         late_mult=0.8, sparse_share=0.04, z_shift=-0.35, plan_mix=(0.10, 0.40, 0.50),
         monthly_share=0.03, pay_fail=0.010, owner_null=0.02, reassign=0.10),
    dict(org_id="org_05", name="Tidewater Apps", share=0.10, event_mult=0.9, ticket_mult=1.2,
         late_mult=1.0, sparse_share=0.09, z_shift=0.45, plan_mix=(0.70, 0.27, 0.03),
         monthly_share=0.40, pay_fail=0.050, owner_null=0.05, reassign=0.20),
    dict(org_id="org_06", name="Lumen Systems", share=0.08, event_mult=1.0, ticket_mult=1.0,
         late_mult=2.0, sparse_share=0.08, z_shift=0.10, plan_mix=(0.45, 0.40, 0.15),
         monthly_share=0.15, pay_fail=0.030, owner_null=0.15, reassign=0.40),
]

# Source outages, expressed as (org_id, source, gap start, gap end, share lost for good)
OUTAGES = [
    ("org_03", "product_events", "2024-11-04", "2024-11-18", 0.20),   # pipeline broke, backfilled
    ("org_04", "zd_tickets",     "2025-03-10", "2025-05-20", 0.00),   # sync paused, fully backfilled
    ("org_05", "gmail_messages", "2025-08-01", "2025-10-15", 0.50),   # sync broke, half lost
]

EVENT_TYPES = ["login", "report_view", "export", "api_call", "dashboard_edit", "invite_user",
               "integration_sync"]
EVENT_P = [0.35, 0.25, 0.08, 0.15, 0.07, 0.02, 0.08]
INDUSTRIES = ["fintech", "healthcare", "retail", "logistics", "education", "media", "manufacturing"]
REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
WORDS_A = ["Blue", "Red", "Silver", "North", "Summit", "Apex", "Vertex", "Cobalt", "Ember", "Harbor",
           "Iron", "Juniper", "Kestrel", "Lattice", "Meridian", "Nimbus", "Onyx", "Prairie", "Quartz",
           "Redwood", "Sable", "Timber", "Umber", "Vista", "Willow", "Zenith", "Alder", "Birch",
           "Cinder", "Delta"]
WORDS_B = ["Logic", "Works", "Dynamics", "Ridge", "Forge", "Pulse", "Grid", "Stream", "Craft", "Labs",
           "Path", "Gate", "Loop", "Shift", "Point", "Wave", "Bridge", "Field", "Mill", "Port",
           "Core", "Link", "Span", "Yard", "Nest", "Peak", "Arc", "Bay", "Dock", "Hub"]
SUFFIXES = ["Inc", "LLC", "Group", "Systems", "Co", "Holdings", "Partners", "Labs"]

# --------------------------------------------------------------------------- calendar arrays
_DAYS = np.arange(N_DAYS)
_DATES = START + pd.to_timedelta(_DAYS, unit="D")
_DOW = _DATES.dayofweek.to_numpy()
_WEEK_F = np.where(_DOW < 5, 1.25, 0.125)           # weekdays busy, weekends nearly dead
_WEEK_F = _WEEK_F / _WEEK_F.mean()
_DOY = _DATES.dayofyear.to_numpy()
_HOL = np.exp(-((((_DOY - 360 + 182) % 365) - 182) / 7.0) ** 2)   # dip around Dec 25
_SEASON = 1.0 - 0.45 * _HOL


# --------------------------------------------------------------------------- small helpers
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def to_dt(sec) -> np.ndarray:
    """seconds-since-BASE (float, NaN allowed) -> datetime64[ns] (NaT where NaN)."""
    s = np.asarray(sec, dtype="float64")
    out = np.full(s.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = ~np.isnan(s)
    out[ok] = (BASE + np.floor(s[ok]).astype("int64").astype("timedelta64[s]")).astype("datetime64[ns]")
    return out


def day_of(ts: str) -> int:
    return int((pd.Timestamp(ts) - START).days)


def stamp(days_arr, counts, rng, mu=13.0, sd=4.0):
    """Turn per-day event counts into second-level timestamps (business hours, roughly)."""
    idx = np.repeat(np.arange(len(counts)), counts)
    hour = np.clip(rng.normal(mu, sd, idx.size), 0.0, 23.99)
    sec = np.floor(days_arr[idx] * DAY + hour * 3600.0)
    return sec, idx


def arrival(event_sec, rng, late_mult, mean_delay_s, p_batch, p_very):
    """When does the row actually reach the warehouse?  Mostly minutes, sometimes days or weeks."""
    n = event_sec.size
    delay = rng.exponential(mean_delay_s, n)
    u = rng.random(n)
    m1 = u < p_batch * late_mult
    delay[m1] = rng.uniform(1 * DAY, 14 * DAY, m1.sum())
    m2 = u < p_very * late_mult
    delay[m2] = rng.uniform(15 * DAY, 60 * DAY, m2.sum())
    return event_sec + delay


def solve_intercept(z, target):
    """Find b so that the average first-renewal churn probability equals `target`."""
    lo, hi = -8.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(mid + z + 0.35).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def concat(chunks, key, dtype=None):
    if not chunks:
        return np.array([], dtype=dtype if dtype else float)
    return np.concatenate([c[key] for c in chunks])


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accounts", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.accounts
    out = Path(args.out)
    raw, truth_dir = out / "raw", out / "_ground_truth"
    raw.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. account attributes
    org_idx = rng.choice(len(ORGS), size=n, p=[o["share"] for o in ORGS])
    org_ids = np.array([ORGS[i]["org_id"] for i in org_idx])
    o_event = np.array([ORGS[i]["event_mult"] for i in org_idx])
    o_ticket = np.array([ORGS[i]["ticket_mult"] for i in org_idx])
    o_late = np.array([ORGS[i]["late_mult"] for i in org_idx])
    o_payfail = np.array([ORGS[i]["pay_fail"] for i in org_idx])

    plan_idx = np.empty(n, dtype=int)
    for j, o in enumerate(ORGS):
        m = org_idx == j
        plan_idx[m] = rng.choice(3, size=m.sum(), p=o["plan_mix"])
    plan = np.array(PLANS)[plan_idx]
    arr_med = np.array([PLAN_ARR_MEDIAN[p] for p in plan], dtype=float)
    arr0 = np.round(arr_med * rng.lognormal(0, 0.35, n), -2)
    seats = np.array([rng.integers(PLAN_SEATS[p][0], PLAN_SEATS[p][1] + 1) for p in plan])

    # hidden risk: org effect + plan effect + individual noise
    z = (np.array([ORGS[i]["z_shift"] for i in org_idx])
         + np.array([PLAN_Z[p] for p in plan]) + rng.normal(0, 0.9, n))
    b = solve_intercept(z, TARGET_FIRST_RENEWAL_CHURN)

    sparse = rng.random(n) < np.array([ORGS[i]["sparse_share"] for i in org_idx])
    size_f = (arr0 / arr_med) ** 0.3
    lam0 = rng.lognormal(np.log(0.195), 0.9, n) * np.exp(-0.45 * z) * size_f   # product events / week
    lam0 = np.where(sparse, rng.uniform(0.003, 0.02, n), lam0)
    tk0 = rng.lognormal(np.log(0.040), 0.7, n) * np.exp(0.45 * z) * (arr0 / arr_med) ** 0.25
    tk0 = np.where(sparse, tk0 * 0.15, tk0)
    slope = np.select([rng.random(n) < 0.5, rng.random(n) < 0.6],
                      [np.zeros(n), rng.uniform(0.1, 0.5, n)], -rng.uniform(0.05, 0.25, n))
    reply_base = rng.beta(6, 4, n)
    touch_gap = np.select([plan == "enterprise", plan == "growth"],
                          [rng.uniform(45, 90, n), rng.uniform(90, 160, n)], rng.uniform(200, 400, n))  # big accounts get more CSM time

    unowned = rng.random(n) < np.array([ORGS[i]["owner_null"] for i in org_idx])
    has_touch = ~(unowned | (sparse & (rng.random(n) < 0.5)))
    monthly_p = np.array([ORGS[i]["monthly_share"] for i in org_idx]) * np.where(
        plan == "enterprise", 0.3, np.where(plan == "starter", 1.3, 1.0))
    monthly = rng.random(n) < np.clip(monthly_p, 0, 0.95)
    term = np.where((plan == "enterprise") & (rng.random(n) < 0.2), 730, 365)

    # start dates: 8% were customers before our data begins; most others start spread across
    # 2023-10..2026-01; a quarter are recent (< 8 months old).
    pre = rng.random(n) < 0.08
    recent = rng.random(n) < 0.25
    origin = np.where(recent, rng.integers(800, N_DAYS - 30, n), rng.integers(0, 800, n))
    origin = np.where(pre, -rng.integers(30, 1200, n), origin)

    # unique, realistic-looking account names (no external library needed)
    names, seen = [], set()
    for i in range(n):
        while True:
            nm = f"{rng.choice(WORDS_A)} {rng.choice(WORDS_B)} {rng.choice(SUFFIXES)}"
            if (org_ids[i], nm) not in seen:
                seen.add((org_ids[i], nm))
                names.append(nm)
                break
    account_ids = np.array([f"acc_{i + 1:05d}" for i in range(n)])

    # CSM pool per org (~120 accounts each)
    csm_pool = {}
    for j, o in enumerate(ORGS):
        cnt = int((org_idx == j).sum())
        k = max(1, int(np.ceil(cnt / 120)))
        csm_pool[o["org_id"]] = ([f"csm_{o['org_id'][-2:]}_{c + 1:02d}" for c in range(k)],
                                 rng.dirichlet(np.full(k, 6.0)))

    # ------------------------------------------------------------------ 2. simulate each account
    prod_c, tick_c, mail_c, inv_c = [], [], [], []
    ren_rows, hist_rows, acct_extra, truth_rows = [], [], [], []

    for i in range(n):
        org = ORGS[org_idx[i]]
        og = org["org_id"]
        o_ = int(origin[i])
        tm = int(term[i])

        # ---- renewal walk: decide outcomes; build the windows that bend behaviour
        windows, decoy_log = [], []
        resolved = []
        cur_arr = float(arr0[i])
        churn_day = churn_type = churn_kind = None
        upcoming = None
        j = 0
        while True:
            r_day = o_ + tm * (j + 1)
            if r_day < 0:                       # renewal happened before our data starts
                j += 1
                continue
            if r_day >= N_DAYS:
                upcoming = r_day
                break
            p = sigmoid(b + z[i] + (0.35 if j == 0 else -0.10))
            if rng.random() < p:
                probs = [0.15, 0.25, 0.60] if sparse[i] else [0.38, 0.42, 0.20]
                churn_type = str(rng.choice(["loud", "fade", "nowhere"], p=probs))
                kp = {"loud": 0.45, "fade": 0.20, "nowhere": 0.50}[churn_type]
                churn_kind = "cancelled" if rng.random() < kp else "lapsed"
                churn_day = r_day
                resolved.append(dict(r_day=r_day, outcome="churned", kind=churn_kind,
                                     arr_before=cur_arr, arr_after=np.nan))
                windows.append(("churn_" + churn_type, r_day))
                break
            arr_after = float(np.round(cur_arr * np.exp(rng.normal(0.03 - 0.12 * z[i], 0.15)), -2))
            resolved.append(dict(r_day=r_day, outcome="renewed", kind=None,
                                 arr_before=cur_arr, arr_after=arr_after))
            cur_arr = arr_after
            if (not sparse[i]) and rng.random() < DECOY_P:
                dt = str(rng.choice(["decoy_tickets", "decoy_usage", "decoy_payment"]))
                windows.append((dt, r_day))
                decoy_log.append(dt)
            j += 1

        d0 = max(o_, 0)
        d1 = churn_day if churn_day is not None else N_DAYS
        L = d1 - d0
        if L <= 0:
            continue
        days = _DAYS[d0:d1]
        age = days - o_
        onboard = 1.0 + 0.6 * np.exp(-age / 20.0)
        trend = np.clip(1.0 + slope[i] * age / 365.0, 0.25, 3.0)

        # ---- behaviour modifiers (1.0 = normal)
        umod, tmod, rmod = np.ones(L), np.ones(L), np.ones(L)
        pmod, esc = np.zeros(L), np.full(L, ESC0)
        for wtype, r_day in windows:
            rr = np.clip((days - (r_day - SIGNAL_LEAD)) / SIGNAL_FULL, 0.0, 1.0)
            if wtype == "churn_loud":
                K = rng.uniform(4.0, 9.0)
                tmod *= 1.0 + (K - 1.0) * rr
                esc += 0.30 * rr
                umod *= 1.0 - rng.uniform(0.15, 0.50) * rr
                rmod *= 1.0 - 0.30 * rr
                pmod += 0.15 * rr
            elif wtype == "churn_fade":
                umod *= 1.0 - rng.uniform(0.75, 0.97) * rr
                tmod *= 1.0 - 0.50 * rr
                rmod *= 1.0 - 0.75 * rr
                pmod += 0.20 * rr
            elif wtype in ("decoy_tickets", "decoy_usage", "decoy_payment"):
                st = r_day - int(rng.integers(20, 150))
                mask = ((days >= st) & (days < st + int(rng.integers(20, 50)))).astype(float)
                if wtype == "decoy_tickets":
                    tmod *= 1.0 + (rng.uniform(3.0, 5.5) - 1.0) * mask
                    esc += 0.05 * mask
                elif wtype == "decoy_usage":
                    umod *= 1.0 - rng.uniform(0.40, 0.70) * mask
                else:
                    pmod += 0.25 * mask

        # ---- product events
        rate_u = lam0[i] * o_event[i] / 7.0 * _WEEK_F[d0:d1] * _SEASON[d0:d1] * onboard * trend * umod
        cnt_u = rng.poisson(rate_u)
        if cnt_u.sum():
            sec, idx = stamp(days, cnt_u, rng)
            users = max(1, int(round(seats[i] * 0.5)))
            pool = np.maximum(1, np.rint(users * np.sqrt(umod[idx]))).astype(int)
            prod_c.append(dict(acc=np.full(sec.size, i), sec=sec,
                               user=(rng.random(sec.size) * pool).astype(int),
                               etype=rng.choice(len(EVENT_TYPES), size=sec.size, p=EVENT_P)))

        # ---- zendesk tickets
        rate_t = (tk0[i] * o_ticket[i] / 7.0 * _WEEK_F[d0:d1]
                  * (1.0 + 0.8 * np.exp(-age / 25.0)) * tmod)
        cnt_t = rng.poisson(rate_t)
        if cnt_t.sum():
            sec, idx = stamp(days, cnt_t, rng)
            escalated = rng.random(sec.size) < np.clip(esc[idx], 0, 0.9)
            prio = rng.choice(4, size=sec.size, p=[0.35, 0.45, 0.15, 0.05])
            prio = np.where(escalated, np.maximum(prio, 2), prio)
            res_h = rng.lognormal(np.log(30.0), 1.0, sec.size) * np.where(escalated, 3.0, 1.0)
            tick_c.append(dict(acc=np.full(sec.size, i), sec=sec, prio=prio, esc=escalated,
                               closed=sec + res_h * 3600.0))

        # ---- gmail (CSM <-> customer)
        gap = touch_gap[i]
        if has_touch[i]:
            n_est = int(L / gap * 1.6) + 4
            t_days = d0 + rng.uniform(0, gap) + np.cumsum(rng.gamma(2.0, gap / 2.0, n_est))
            di = np.floor(t_days[t_days < d1 - 1]).astype(int)
            di = di + np.where(_DOW[di] == 5, 2, np.where(_DOW[di] == 6, 1, 0))
            di = di[di < d1 - 1]
            if di.size:
                out_sec = np.floor(di * DAY + rng.uniform(9, 17, di.size) * 3600.0)
                got = rng.random(di.size) < np.clip(reply_base[i] * rmod[di - d0], 0, 1)
                rep_sec = np.floor(out_sec + rng.lognormal(np.log(6 * 3600.0), 1.0, di.size))
                got &= rep_sec < d1 * DAY
                mail_c.append(dict(acc=np.full(di.size, i), sec=out_sec,
                                   dir=np.zeros(di.size, dtype=int)))
                if got.any():
                    mail_c.append(dict(acc=np.full(got.sum(), i), sec=rep_sec[got],
                                       dir=np.ones(got.sum(), dtype=int)))
        cnt_m = rng.poisson(0.004 * np.sqrt(tmod))                      # unprompted inbound mail
        if cnt_m.sum():
            sec, idx = stamp(days, cnt_m, rng)
            mail_c.append(dict(acc=np.full(sec.size, i), sec=sec, dir=np.ones(sec.size, dtype=int)))

        # ---- stripe invoices
        if monthly[i]:
            m_lo = int(np.ceil((d0 - o_) / 30.4))
            inv_days = o_ + np.round(30.4 * np.arange(m_lo, m_lo + int(L / 30.4) + 2)).astype(int)
            factor = 1.0 / 12.0
        else:
            j_lo = int(np.ceil((d0 - o_) / tm))
            inv_days = o_ + tm * np.arange(j_lo, j_lo + int(L / tm) + 2)
            factor = tm / 365.0
        inv_days = inv_days[(inv_days >= d0) & (inv_days < d1)]
        if inv_days.size:
            step_days = np.array([r["r_day"] for r in resolved if r["outcome"] == "renewed"])
            step_arr = np.array([arr0[i]] + [r["arr_after"] for r in resolved if r["outcome"] == "renewed"])
            arr_at = step_arr[np.searchsorted(step_days, inv_days, side="right")]
            pm = pmod[inv_days - d0]
            p_fail = np.clip(o_payfail[i] * (1.0 if monthly[i] else 0.4) + pm, 0, 0.95)
            fail = rng.random(inv_days.size) < p_fail
            hard = fail & (rng.random(inv_days.size) < np.clip(0.30 + 1.5 * pm, 0, 0.9))
            attempts = np.where(fail & ~hard, rng.integers(2, 4, inv_days.size), 1)
            attempts = np.where(hard, 3, attempts)
            status = np.where(hard, 2, np.where(rng.random(inv_days.size) < 0.004, 3, 1))  # 1 paid 2 failed 3 refunded
            inv_sec = inv_days * DAY + rng.uniform(2, 6, inv_days.size) * 3600.0
            inv_c.append(dict(acc=np.full(inv_days.size, i), sec=np.floor(inv_sec),
                              amount=np.round(arr_at * factor, 2), status=status,
                              attempts=attempts, failed=fail))

        # ---- salesforce bits (owner history, renewals, account extras)
        csms, w = csm_pool[og]
        o1 = str(rng.choice(csms, p=w))
        change = int(d0 + rng.uniform(0.25, 0.75) * L)
        reassigned = rng.random() < org["reassign"]
        if unowned[i]:
            if rng.random() < 0.4:
                hist = [(None, o_)]
            else:
                hist = [(o1, o_), (None, change)]
        else:
            hist = [(o1, o_)]
            if reassigned:
                hist.append((str(rng.choice([c for c in csms if c != o1] or csms)), change))
        for k, (ow, vf) in enumerate(hist):
            vt = hist[k + 1][1] if k + 1 < len(hist) else None
            hist_rows.append((account_ids[i], og, ow, vf, vt))

        lag_mean = 6.0 * org["late_mult"]
        for r in resolved:
            rec = r["r_day"] + int(rng.exponential(lag_mean))
            known = rec <= N_DAYS
            ren_rows.append(dict(
                account_id=account_ids[i], org_id=og, renewal_day=r["r_day"],
                outcome=r["outcome"] if known else None,
                outcome_day=r["r_day"] if known else None,
                churn_kind=r["kind"] if known else None,
                arr_before=r["arr_before"], arr_after=r["arr_after"] if known else np.nan,
                recorded_day=rec if known else None))
        if upcoming is not None:
            ren_rows.append(dict(account_id=account_ids[i], org_id=og, renewal_day=upcoming,
                                 outcome=None, outcome_day=None, churn_kind=None,
                                 arr_before=cur_arr, arr_after=np.nan, recorded_day=None))

        contract_end = churn_day if churn_day is not None else upcoming
        acct_extra.append(dict(i=i, cur_arr=cur_arr, contract_end=contract_end,
                               contract_start=contract_end - tm,
                               owner=hist[-1][0], status="churned" if churn_day is not None else "active"))
        truth_rows.append(dict(account_id=account_ids[i], org_id=og, hidden_risk_z=round(float(z[i]), 4),
                               is_sparse=bool(sparse[i]), base_events_per_week=round(float(lam0[i]), 4),
                               churned=churn_day is not None, churn_type=churn_type,
                               churn_kind=churn_kind, churn_day=churn_day,
                               n_decoys=len(decoy_log), decoys="|".join(decoy_log) or None))

    # ------------------------------------------------------------------ 3. assemble the sources
    acc_org = lambda a: org_ids[a]                            # noqa: E731
    acc_late = lambda a: o_late[a]                            # noqa: E731
    acc_id = lambda a: account_ids[a]                         # noqa: E731

    # product events
    pa = concat(prod_c, "acc", int)
    p_sec = concat(prod_c, "sec")
    products = pd.DataFrame(dict(
        event_id=[f"evt_{k + 1:09d}" for k in range(pa.size)],
        org_id=acc_org(pa), account_id=acc_id(pa),
        user_id=np.char.add(np.char.add("u_", (pa + 1).astype(str)), np.char.add("_", concat(prod_c, "user").astype(str))),
        event_type=np.array(EVENT_TYPES)[concat(prod_c, "etype", int)],
        event_sec=p_sec))
    products["recv_sec"] = arrival(p_sec, rng, acc_late(pa), 20 * 60, 0.040, 0.004)
    products.loc[rng.random(len(products)) < 0.03, "user_id"] = None
    orph = rng.random(len(products))
    products.loc[orph < 0.004, "account_id"] = None                      # unattributed events
    products.loc[(orph >= 0.004) & (orph < 0.005), "account_id"] = "acc_99999"   # ghost account

    # zendesk tickets
    ta = concat(tick_c, "acc", int)
    t_sec = concat(tick_c, "sec")
    tickets = pd.DataFrame(dict(
        ticket_id=[f"zd_{k + 1:08d}" for k in range(ta.size)],
        org_id=acc_org(ta), account_id=acc_id(ta),
        event_sec=t_sec, closed_sec=concat(tick_c, "closed"),
        priority=np.array(["low", "normal", "high", "urgent"])[concat(tick_c, "prio", int)],
        escalated=concat(tick_c, "esc", bool)))
    tickets.loc[tickets["closed_sec"] > END_SEC, "closed_sec"] = np.nan        # still open at pull time
    tickets.loc[rng.random(len(tickets)) < 0.01, "closed_sec"] = np.nan        # never closed (data issue)
    miss_p = np.where(tickets["org_id"] == "org_06", 0.15, np.where(tickets["org_id"] == "org_03", 0.10, 0.04))
    tickets.loc[rng.random(len(tickets)) < miss_p, "priority"] = None
    tickets["recv_sec"] = arrival(t_sec, rng, acc_late(ta), 5 * 60, 0.025, 0.003)

    # gmail
    ma = concat(mail_c, "acc", int)
    m_sec = concat(mail_c, "sec")
    mails = pd.DataFrame(dict(
        message_id=[f"gm_{k + 1:08d}" for k in range(ma.size)],
        org_id=acc_org(ma), account_id=acc_id(ma),
        direction=np.where(concat(mail_c, "dir", int) == 0, "outbound", "inbound"),
        event_sec=m_sec))
    mails["recv_sec"] = arrival(m_sec, rng, acc_late(ma), 15 * 60, 0.020, 0.002)

    # stripe
    ia = concat(inv_c, "acc", int)
    i_sec = concat(inv_c, "sec")
    st_code = concat(inv_c, "status", int)
    att = concat(inv_c, "attempts", int)
    failed = concat(inv_c, "failed", bool)
    invoices = pd.DataFrame(dict(
        invoice_id=[f"in_{k + 1:08d}" for k in range(ia.size)],
        org_id=acc_org(ia), account_id=acc_id(ia),
        event_sec=i_sec, amount=concat(inv_c, "amount"),
        status=np.array(["", "paid", "failed", "refunded"])[st_code], attempts=att,
        failed_sec=np.where(failed, i_sec + 86400.0, np.nan),
        paid_sec=np.where(st_code == 2, np.nan, i_sec + (att - 1) * 3 * 86400.0)))
    invoices["recv_sec"] = arrival(i_sec, rng, acc_late(ia), 2 * 60, 0.010, 0.001)

    # ------------------------------------------------------------------ 4. outages, cutoff, duplicates
    sources = {"product_events": products, "zd_tickets": tickets,
               "gmail_messages": mails, "stripe_invoices": invoices}
    for og, src, g0, g1, lost in OUTAGES:
        df = sources[src]
        m = (df["org_id"] == og) & (df["event_sec"] >= day_of(g0) * DAY) & (df["event_sec"] < day_of(g1) * DAY)
        df.loc[m, "recv_sec"] = day_of(g1) * DAY + DAY + rng.uniform(0, 6 * 3600, m.sum())   # backfill burst
        if lost > 0:
            drop = m.to_numpy() & (rng.random(len(df)) < lost)
            sources[src] = df.loc[~drop]
    products, tickets = sources["product_events"], sources["zd_tickets"]
    mails, invoices = sources["gmail_messages"], sources["stripe_invoices"]

    def finish(df, id_col, dup_frac, jitter):
        df = df[df["recv_sec"] < END_SEC].copy()                    # not yet arrived at pull time
        k = int(len(df) * dup_frac)
        if k:
            dup = df.sample(k, random_state=int(rng.integers(1 << 31))).copy()   # same id, re-delivered
            if jitter:
                j = rng.random(k) < 0.25                      # a quarter get a slightly different timestamp
                dup.loc[dup.index[j], "event_sec"] += rng.integers(-90, 91, j.sum())
            dup["recv_sec"] = np.minimum(dup["recv_sec"] + rng.uniform(1, 600, k), END_SEC - 1)
            df = pd.concat([df, dup], ignore_index=True)
        return df.sort_values("recv_sec", kind="stable").reset_index(drop=True)

    products = finish(products, "event_id", 0.020, True)
    tickets = finish(tickets, "ticket_id", 0.020, False)
    mails = finish(mails, "message_id", 0.020, True)
    invoices = finish(invoices, "invoice_id", 0.010, False)

    products_o = products[["event_id", "org_id", "account_id", "user_id", "event_type"]].copy()
    products_o["event_time"] = to_dt(products["event_sec"]); products_o["received_at"] = to_dt(products["recv_sec"])
    tickets_o = tickets[["ticket_id", "org_id", "account_id"]].copy()
    tickets_o["opened_at"] = to_dt(tickets["event_sec"]); tickets_o["closed_at"] = to_dt(tickets["closed_sec"])
    tickets_o["priority"] = tickets["priority"]; tickets_o["escalated"] = tickets["escalated"]
    tickets_o["received_at"] = to_dt(tickets["recv_sec"])
    mails_o = mails[["message_id", "org_id", "account_id", "direction"]].copy()
    mails_o["sent_at"] = to_dt(mails["event_sec"]); mails_o["received_at"] = to_dt(mails["recv_sec"])
    inv_o = invoices[["invoice_id", "org_id", "account_id"]].copy()
    inv_o["invoice_date"] = to_dt(invoices["event_sec"]); inv_o["amount"] = invoices["amount"]
    inv_o["status"] = invoices["status"]; inv_o["attempts"] = invoices["attempts"]
    inv_o["failed_at"] = to_dt(invoices["failed_sec"]); inv_o["paid_at"] = to_dt(invoices["paid_sec"])
    inv_o["received_at"] = to_dt(invoices["recv_sec"])

    # ------------------------------------------------------------------ 5. salesforce tables
    ex = pd.DataFrame(acct_extra).set_index("i")
    sel = ex.index.to_numpy()
    plan_noisy = plan[sel].astype(object)
    v = rng.random(sel.size)
    noisy = v < 0.08                                   # inconsistent casing / trailing spaces
    plan_noisy[noisy] = [p.upper() if rng.random() < 0.5 else p.capitalize() + " "
                         for p in plan[sel][noisy]]
    plan_noisy[(v >= 0.08) & (v < 0.095)] = None       # blank plan
    arr_cur = ex["cur_arr"].to_numpy().astype(float)
    arr_init = arr0[sel].astype(float)
    miss = rng.random(sel.size) < 0.015
    arr_cur[miss] = np.nan
    arr_init[rng.random(sel.size) < 0.015] = np.nan
    ind = np.array(INDUSTRIES, dtype=object)[rng.integers(0, len(INDUSTRIES), sel.size)]
    ind[rng.random(sel.size) < 0.06] = None
    reg = np.array(REGIONS, dtype=object)[rng.integers(0, len(REGIONS), sel.size)]
    reg[rng.random(sel.size) < 0.03] = None
    bill = np.where(monthly[sel], "monthly", "annual").astype(object)
    bill[rng.random(sel.size) < 0.03] = None

    def day_to_date(d):
        return (START + pd.to_timedelta(np.asarray(d, dtype=int), unit="D")).strftime("%Y-%m-%d")

    sf = pd.DataFrame(dict(
        account_id=account_ids[sel], org_id=org_ids[sel], account_name=np.array(names, dtype=object)[sel],
        plan=plan_noisy, industry=ind, region=reg, billing_cycle=bill, seats=seats[sel],
        initial_arr=arr_init, arr=arr_cur,
        customer_since=day_to_date(origin[sel]),
        contract_start=day_to_date(ex["contract_start"].to_numpy()),
        contract_end=day_to_date(ex["contract_end"].to_numpy()),
        owner_id=ex["owner"].to_numpy(), status=ex["status"].to_numpy()))

    # ~1% duplicate CRM records: same company, new id, mostly empty. Pipelines must dedupe.
    nd = max(1, int(0.01 * len(sf)))
    base = sf.sample(nd, random_state=7).copy()
    base["account_id"] = [f"acc_d{k + 1:04d}" for k in range(nd)]
    base["account_name"] = [s.replace(" Inc", " Inc.").replace(" Co", " Co.").upper() if k % 2 else s + " "
                            for k, s in enumerate(base["account_name"])]
    base["plan"], base["arr"], base["initial_arr"], base["owner_id"] = None, np.nan, np.nan, None
    base["status"] = "active"
    dup_truth = [dict(account_id=a, org_id=o, duplicate_of=src)
                 for a, o, src in zip(base["account_id"], base["org_id"], sf.loc[base.index, "account_id"])]
    sf = pd.concat([sf, base], ignore_index=True)

    hist = pd.DataFrame(hist_rows, columns=["account_id", "org_id", "owner_id", "vf", "vt"])
    hist["valid_from"] = day_to_date(hist["vf"].to_numpy())
    hist["valid_to"] = [None if pd.isna(x) else d for x, d in zip(hist["vt"], day_to_date(hist["vt"].fillna(0).to_numpy()))]
    hist = hist[["account_id", "org_id", "owner_id", "valid_from", "valid_to"]]

    ren = pd.DataFrame(ren_rows)
    ren_o = pd.DataFrame(dict(
        account_id=ren["account_id"], org_id=ren["org_id"],
        renewal_date=day_to_date(ren["renewal_day"].to_numpy()),
        outcome=ren["outcome"], churn_kind=ren["churn_kind"],
        outcome_date=[None if pd.isna(x) else d for x, d in zip(ren["outcome_day"], day_to_date(ren["outcome_day"].fillna(0).to_numpy()))],
        outcome_recorded_at=[None if pd.isna(x) else d for x, d in zip(ren["recorded_day"], day_to_date(ren["recorded_day"].fillna(0).to_numpy()))],
        arr_before=ren["arr_before"], arr_after=ren["arr_after"]))

    # ------------------------------------------------------------------ 6. write everything
    products_o.to_csv(raw / "product_events.csv", index=False)
    tickets_o.to_csv(raw / "zd_tickets.csv", index=False)
    mails_o.to_csv(raw / "gmail_messages.csv", index=False)
    inv_o.to_csv(raw / "stripe_invoices.csv", index=False)
    sf.to_csv(raw / "sf_accounts.csv", index=False)
    hist.to_csv(raw / "sf_owner_history.csv", index=False)
    ren_o.to_csv(raw / "sf_renewals.csv", index=False)

    truth = pd.DataFrame(truth_rows).merge(pd.DataFrame(dup_truth), how="outer", on=["account_id", "org_id"])
    truth.to_csv(truth_dir / "account_truth.csv", index=False)
    (truth_dir / "generation_params.json").write_text(json.dumps(dict(
        seed=args.seed, accounts=n, start=str(START.date()), data_pulled=str(END.date()),
        intercept=round(float(b), 4), target_first_renewal_churn=TARGET_FIRST_RENEWAL_CHURN,
        orgs=ORGS, outages=OUTAGES), indent=2, default=str))

    counts = {"product_events": len(products_o), "zd_tickets": len(tickets_o),
              "gmail_messages": len(mails_o), "stripe_invoices": len(inv_o)}
    sf_rows = len(sf) + len(hist) + len(ren_o)
    print("Wrote data to", raw.resolve())
    for k, v_ in counts.items():
        print(f"  {k:<16}{v_:>10,}")
    print(f"  {'salesforce (3)':<16}{sf_rows:>10,}   (accounts {len(sf):,}, owner history {len(hist):,}, renewals {len(ren_o):,})")
    print(f"  {'TOTAL rows':<16}{sum(counts.values()) + sf_rows:>10,}")


if __name__ == "__main__":
    main()
