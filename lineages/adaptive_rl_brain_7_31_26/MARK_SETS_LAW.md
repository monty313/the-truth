# MARK SETS LAW — immutable (Mark on the chart)

**Owner:** MARK HERE / Monty  
**Code pin:** `perception/sets.py` · `assert_mark_sets_law()`  
**Status:** LAW — do not “optimize” TFs without explicit human rewrite of this file + tests.

---

## Official four sets (only)

| Set | LTF (first) | HTF confirm (2nd, 3rd) | Role |
|----:|-------------|------------------------|------|
| 1 | **1m** | 15m, 30m | Micro: pullbacks / cont / adds on 1m |
| 2 | **5m** | 30m, 1h | Intraday |
| 3 | **15m** | 1h, 4h | Swing stack |
| 4 | **30m** | 4h, 1d | Macro gravity |

**Stack string form:**  
`1m,15m,30m` · `5m,30m,1h` · `15m,1h,4h` · `30m,4h,1d`

---

## Roles (Mark law)

1. **First TF = LTF / entry**  
   - Identify **pullbacks**, **continuations**, and **adds** on that TF only.  
   - LTF **times** the trade; it does **not** redefine side against HTF.

2. **Second + third TF = HTF confirmation**  
   - **Two** higher timeframes confirm **trend / permission**.  
   - Side permission comes from HTF stack, not from LTF noise.

3. **Scan all four sets** every decision for opportunities.  
   - Never collapse to Official Set 2 only for Mark-path eyes.  
   - (`legacy_set2` remains claim baseline only — not Mark.)

4. **Pullback / continuation / add**  
   - Allowed only **with** that set’s HTF trend.  
   - Against HTF = stand down or wait (slingshot load), not reverse thrash.

5. **Runtime target% / risk%**  
   - Shell inputs; **no retrain** to change numbers. Sets law is independent of pair.

---

## Forbidden without human rewrite

- Changing any TF in the four stacks  
- Using set2 alone as “the” Mark eye  
- LTF voting side against its HTF pair  
- PROVEN overwrite “to fix sets”

---

## How agents enforce

| Path | Must |
|------|------|
| Teacher / Mark eyes | `eyes_mode=mark_doctrine` or `mark_all_sets` over **all** `OFFICIAL_SETS` |
| BC / new policy | Same Channel1 obs packing all 4 sets; never invent TFs |
| Tests | `test_mark_sets_law.py` pins stacks |

*Mark on the chart = these sets as law + pullback/continuation on LTF + HTF permission — not a vibes persona.*
