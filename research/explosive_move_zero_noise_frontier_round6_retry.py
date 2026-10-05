from pathlib import Path

source_path = Path('research/explosive_move_zero_noise_frontier_round6.py')
source = source_path.read_text()
source = source.replace(
    'candidate_idx, prefilter = prefilter(ep_by_split, rule_masks, wanted_bits, noise_bits)',
    'candidate_idx, prefilter_df = prefilter(ep_by_split, rule_masks, wanted_bits, noise_bits)',
)
source = source.replace(
    "prefilter.to_csv(OUT / 'candidate_prefilter.csv', index=False)",
    "prefilter_df.to_csv(OUT / 'candidate_prefilter.csv', index=False)",
)
exec(compile(source, str(source_path), 'exec'), {'__name__': '__main__'})
