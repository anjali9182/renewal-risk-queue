# Renewal Risk for Customer Success

A Monday morning priority list for Customer Success Managers (CSMs). It ranks
every account by how likely it is to not renew in the next 90 days, and shows
the reasons behind each score so a CSM knows what to do about it.

Built for the Fello GTM AI Team take-home exercise. All data is synthetic.

## What problem I think this solves

The request I got asked for a red/yellow/green health score and automatic
emails to red accounts. After thinking about it, I don't believe a color is
the real problem. A CSM with about 120 accounts can't tell which handful need
attention this week, and two renewals were lost even though the warning signs
were sitting in Zendesk and product usage.

So I built a ranked list instead of a color. Each account gets a probability
of churning, which can be checked against what actually happened. A health
score of "62" can't be.

I also did not build automatic emailing. A generic email sent by a bot to a
customer with an angry open ticket can make things worse, and false alarms
teach healthy customers to ignore us. The system drafts the outreach and the
CSM decides whether to send it.

## What it does

- Combines five sources (Salesforce, Zendesk, product events, Stripe, Gmail)
  into one view of each account.
- Fits a churn model on historical renewals and scores every account.
- Runs on a schedule with nobody pressing a button.
- Shows each CSM a ranked list with the top reasons for each account.
- Keeps customer organisations separate, so one org never sees another's data.
- Lets you trace any number on screen back to the model run and the exact
  inputs that produced it.

## Key decisions

- **What the model predicts:** whether an account fails to renew within 90 days
  of a scoring date. One row is one account on one date, scored monthly.
- **No peeking at the future:** a feature only uses data that had actually
  arrived by the scoring date. Events show up late in real life, so this
  matters.
- **Time-based evaluation:** I train on earlier dates and test on later ones.
  A random split would leak the future into training.
- **Sparse accounts:** an account with almost no history is marked "not enough
  data" instead of being scored as safe.
- **Churn definition:** non-renewal at the contract end date, whether the
  customer cancelled or just let it lapse. Downgrades are not counted yet. I'd
  confirm this with the business first.

_(I'll add to this list as I make more choices.)_

## What I deliberately skipped

_(Fill in as I build.)_

## What breaks first at 10x

_(Fill in as I build.)_

## How to run it

1. Create the environment: `python -m venv .venv`
2. Activate it on Windows PowerShell: `.venv\Scripts\Activate.ps1`
3. Install packages: `pip install -r requirements.txt`
4. Generate the data: `python src/generate_data.py`

More steps will be added as the model and the app come together.

## Project layout

| Folder | What's in it |
| --- | --- |
| `src/` | Data generator, features, model training, scoring |
| `app/` | The view a CSM opens on Monday |
| `tests/` | Tests |
| `data/` | Generated data (not committed to git) |

## A note on the data

Everything here is synthetic, so the results show that the pipeline works, not
that the model would be accurate on real customers. The generator uses a fixed
random seed, so anyone can recreate the same dataset.