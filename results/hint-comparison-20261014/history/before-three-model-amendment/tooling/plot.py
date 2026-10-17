"""Plot all three prompt conditions, labeling incomplete panels explicitly."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/Users/yp9846/Desktop/PhysLiteBench')
NAME = 'hint-comparison-20261014'
RESULTS = ROOT / 'results' / NAME
OUTPUT = ROOT / 'output' / NAME


def main():
    plan = json.loads((RESULTS / 'plan.json').read_text())
    data = json.loads((RESULTS / 'results.json').read_text())
    if not data['status']['recorded_attempts']:
        return
    labels = {'baseline': 'No hint', 'general': 'General hint', 'physics': 'Physics hint'}
    colors = {'baseline': '#64748B', 'general': '#D08A19', 'physics': '#147D92'}
    names = ['GPT-5.6 Luna', 'GPT-5.6 Terra', 'GPT-6 Sol', 'GPT-6 Astra', 'DeepSeek V4.1 Flash']
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(5)
    width = .25
    for i, condition in enumerate(plan['conditions']):
        rows = [next(r for r in data['summary'] if r['model'] == model and r['condition'] == condition) for model in plan['models']]
        positions = x + (i - 1) * width
        values = [100 * r['passes'] / r['attempts'] if r['attempts'] else 0 for r in rows]
        bars = ax.bar(positions, values, width=.22, color=colors[condition], label=labels[condition])
        for bar, position, value, row in zip(bars, positions, values, rows):
            if not row['complete']:
                bar.set_alpha(.4)
                bar.set_hatch('//')
            label = f"{row['passes']}/{row['attempts']}" if row['attempts'] else 'unrun'
            ax.text(position, value + 2, label, ha='center', fontsize=8)
            ax.text(position, -8, str(row['errors']) if row['attempts'] else '—', ha='center', fontsize=8, color='#475569')
    ax.set_xticks(x, names)
    ax.set_ylim(-12, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('Completed passing attempts (%)')
    ax.set_title('Physics benchmark: benefit of hints at High reasoning effort', loc='left', fontsize=16, weight='bold', pad=35)
    subtitle = f"13 frozen tasks · 3 repeats per condition · {data['status']['recorded_attempts']}/585 attempts recorded"
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, color='#475569', fontsize=10)
    ax.grid(axis='y', color='#E2E8F0')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3, frameon=False)
    fig.text(.08, .025, 'Numbers above bars: passes / recorded attempts. Bottom row: execution errors. Hatched bars are incomplete.\nAll baselines are fresh. Failed grades are not automatically classified as physics errors. Task selection used GPT-5.6 Luna.', fontsize=9, color='#475569')
    fig.subplots_adjust(bottom=.23, top=.82)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUTPUT / f'hint-comparison.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
