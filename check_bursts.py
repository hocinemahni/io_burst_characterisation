import pandas as pd

for app in ['HACC', 'E3SM']:
    df = pd.read_csv(f'article_reproduced_figures_same/csv/{app}_adaptive_detection_events.csv')
    fixed = df['threshold_bw_fixed'].dropna().iloc[0]
    only_adaptive = df[(df['burst_adaptive'] == True) & (df['burst_fixed'] == False)]
    fixed_bursts = df[df['burst_fixed'] == True]
    bw_oa = only_adaptive['bandwidth_mb_s']
    bw_fb = fixed_bursts['bandwidth_mb_s']
    print(f"{app}: threshold={fixed:.1f}")
    print(f"  Only-adaptive (red): n={len(only_adaptive)}, BW min={bw_oa.min():.1f}, max={bw_oa.max():.1f}")
    print(f"  Fixed (blue X):      n={len(fixed_bursts)}, BW min={bw_fb.min():.1f}, max={bw_fb.max():.1f}")
    print()
