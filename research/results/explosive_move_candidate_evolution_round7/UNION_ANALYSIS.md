# Round 7 Checkpoint-Union Addendum

**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.

This addendum combines the independently selected causal +3h/+6h/+12h/+24h/+48h frontier rule sets with OR logic. Combination selection uses development (2019-2024) only. 2025-2026 remains frozen confirmation.

## Key result

Combining high-precision checkpoint signals improves total winner retention at low noise, but does not solve the weekly-coverage problem.

- Union of all development-zero-noise checkpoint rules: **349 wanted candidate episodes vs 23 noise** across all periods, **93.8% precision**, **161 weeks covered**.
- Winner-maximising union under development noise cap 2: **396 wanted vs 29 noise**, **93.2% precision**, **164 weeks**.
- Winner-maximising union under development noise cap 5: **429 wanted vs 36 noise**, **92.3% precision**, **174 weeks**.
- Winner-maximising union under development noise cap 20: **544 wanted vs 57 noise**, **90.5% precision**, **196 weeks**.
- Weekly-coverage-maximising union under development noise cap 5: **395 wanted vs 31 noise**, **92.7% precision**, **179 weeks**.
- To reach about **350 weeks**, the weekly-coverage frontier requires **2,203 wanted vs 1,219 noise** (**64.4% precision**).
- Maximum observed weekly coverage in the searched checkpoint unions is **358 weeks**, with **2,912 wanted vs 1,796 noise** (**61.9% precision**).

Thus candidate evolution materially improves precision, but near-zero candidate noise and near-weekly opportunity coverage remain incompatible in this feature/rule family.

The checkpoint-union search is a derived research analysis only. It does not alter the live strategy.
