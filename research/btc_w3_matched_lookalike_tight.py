from pathlib import Path
import btc_w3_matched_lookalike_analysis as analysis

# Sensitivity run: use the genuinely five closest eligible lookalikes for every
# winner event. Controls may be reused across different events, avoiding the
# state-balance degradation caused by forcing 540 unique controls.
analysis.MAX_CONTROL_REUSE = 10_000
analysis.OUT = Path("research/results/daily_return_harvest/btc_w3_matched_lookalikes_tight_2024")
analysis.OUT.mkdir(parents=True, exist_ok=True)
analysis.main()
