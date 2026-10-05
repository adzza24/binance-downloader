# High-Purity A/B/C Sampled-Signal Overlap Analysis

Research-only. This measures overlap among the constructed event/control signal rows; it is not yet the full-universe portfolio replay.

- Raw signal rows: 1874
- Duplicate symbol/time rows participating in an exact duplicate: 28
- Unique sampled coin/time signals: 1857
- Occupancy assumption: every accepted position holds its slot for the full 24h. This is intentionally conservative because compact samples do not preserve target/stop hit hour.

## By year

```text
 year  signals  span_days  signals_per_day  w3_rate  hours_with_2plus  hours_with_3plus  max_same_hour  signal_rows_in_multi_signal_hours_pct  median_gap_hours  pct_gap_lt_6h  pct_gap_lt_12h  pct_gap_lt_24h  max_concurrent_24h_hold  one_slot_capture_pct  one_slot_blocked  two_slot_capture_pct  two_slot_blocked  three_slot_capture_pct  three_slot_blocked
 2021       85         61         1.393443 0.941176                12                 5              7                               0.388235               6.5       0.428571        0.571429        0.761905                       11              0.352941                55              0.552941                38                0.705882                  25
 2022      362        357         1.014006 0.988950                51                 9              5                               0.323204              13.0       0.409972        0.487535        0.606648                        9              0.441989               202              0.709945               105                0.859116                  51
 2023      358        364         0.983516 0.966480                39                11              5                               0.270950              14.0       0.336134        0.445378        0.627451                        9              0.463687               192              0.731844                96                0.854749                  52
 2024      470        365         1.287671 0.948936                50                14              5                               0.257447               8.0       0.420043        0.567164        0.720682                       14              0.372340               295              0.617021               180                0.776596                 105
 2025      329        364         0.903846 0.896657                35                 7              7                               0.246201              17.0       0.344512        0.439024        0.570122                        8              0.492401               167              0.750760                82                0.872340                  42
 2026      253        243         1.041152 0.913043                19                 1              3                               0.154150              15.5       0.273810        0.448413        0.630952                        7              0.474308               133              0.774704                57                0.905138                  24
```

Slot capture uses deterministic A>B>C / score ordering only to estimate capacity. It is not a proposed ranking rule.
