# EXHIBIT D-1 — THE LEDGER
### The trade-by-trade receipts behind the 70% verdict · sealed 2026-07-21

**What the court demanded:** not claims — receipts. Below is the actual ledger: real
prices from the committed M1 record, a $10,000 starting account, every entry, every exit,
every dip, every leverage figure. Anyone can reprint this page with one command:

```
python scripts/exhibit_d_ledger.py
```

**The rules the ledger obeys:** trades happen on the bot's own 5-minute decision grid;
a window is taken only if it is profitable after paying the spread both ways; every trade
is sized so its worst dip — measured from the window's **real intrabar high/low**, not the
closes — can cost at most 4.5% of the account at that moment (sized a hair inside the law,
never touching past it); leverage is capped at 1:100; profits compound. It is a hindsight
best-case — the same evidentiary standard as Part One's oracle: it proves what the market
**offered**. Learning to take it without hindsight is precisely what training buys.

---

## THE STAR WITNESS — January 29, 2026: $10,000 → $18,451 in 15 minutes, 3 trades

```
  min  side   entry      exit     price-move   worst-dip   leverage   gain     balance
   0-5   LONG   5422.72    5438.88   +0.298%      -4.50%       37x    +11.0%   $11,101
   5-10  LONG   5438.88    5448.61   +0.179%      -4.50%       37x     +6.6%   $11,839
  10-15  LONG   5448.61    5479.12   +0.560%      -0.30%      100x    +55.8%   $18,451

  +70% CROSSED at minute 15 | worst single dip: -4.50% of equity | trades used: 3
```

**Read it in simple words.** Gold rose about **three tenths of one percent** in the first
five minutes. At 37× leverage that tiny move pays **+11%** on the account. Five minutes
later, another small rise: **+6.6%**. Then a half-percent surge at full 100× leverage:
**+55.8%**. Three trades. Fifteen minutes. The account never dipped more than the 4.5%
law allows — the dips column is measured from the actual lows of those very bars. This is
the owner's rule made visible: *the price only has to move a little when the lot size is
the lever.*

## THE CORROBORATION — the other cited days, same rules

```
  2026-03-23: +70% at minute  45 |  9 trades used | worst dip -4.50% (never past 4.5%)
  2026-02-02: +70% at minute  35 |  7 trades used | worst dip -4.50% (never past 4.5%)
  2026-01-30: +70% at minute  40 |  8 trades used | worst dip -4.50% (never past 4.5%)
  2026-05-22: +70% at minute 115 | 22 trades used | worst dip -4.50% (never past 4.5%)
```

That last line is the quiet killer: **May 22 was the sleepiest day in the entire 90-day
book** (total range 1.22%) — and even it offered +70% by minute 115, in 22 small trades,
inside the 4.5% law. If the quietest day qualifies, the argument is over.

## THE FULL COUNT

```
  days where +70% was reached under the 4.5% law: 90 of 90
```

## CHAIN OF CUSTODY (sources — all committed, nothing made up)

- **The price record:** [`data/XAUUSD_curriculum_2026.csv`](https://github.com/monty313/the-truth/blob/main/data/XAUUSD_curriculum_2026.csv) — 90 real XAUUSD trading days (2026-01-20 → 2026-05-26), M1 bars from the owner's MT5 broker export.
- **The measuring instrument:** [`scripts/exhibit_d_ledger.py`](https://github.com/monty313/the-truth/blob/main/scripts/exhibit_d_ledger.py) (this ledger) and [`scripts/prove_70.py`](https://github.com/monty313/the-truth/blob/main/scripts/prove_70.py) (the 90-day sweep). Deterministic — same command, same numbers, forever.
- **The case it seals:** [`THE_CLOSING_ARGUMENT.md`](https://github.com/monty313/the-truth/blob/main/doctrine/flea-jar/THE_CLOSING_ARGUMENT.md) · the diagnosis: [`THE_FLEA_CURE.md`](https://github.com/monty313/the-truth/blob/main/doctrine/flea-jar/THE_FLEA_CURE.md).

*Exhibit admitted. The defense rests — on the receipts.* ⚖️
