"""Display-only conversions and figures; no model math or training."""
import numpy as np
import matplotlib.pyplot as plt
from tensor import Vector, Matrix
from layer import Layer


def display_values(value):
    kind = type(value)
    contracts = {('tensor', 'Vector'): 1, ('tensor', 'Matrix'): 2, ('layer', 'Layer'): 1}
    expected = contracts.get((kind.__module__, kind.__name__))
    if expected is None or not callable(getattr(value, 'to_list', None)):
        raise TypeError('Expected Vector, Matrix, or Layer')
    result = np.array(value.to_list(), dtype=float, copy=True)
    if result.ndim != expected or not np.isfinite(result).all():
        raise ValueError('Expected finite values with the original shape')
    return result


def architecture_figure(sizes):
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = [np.linspace(0.1, 0.9, size) if size > 1 else [0.5] for size in sizes]
    for i in range(len(sizes)-1):
        for y1 in positions[i]:
            for y2 in positions[i+1]:
                ax.annotate('', (i+1, y2), (i, y1),
                            arrowprops=dict(arrowstyle='->', color='#cbd5e1', lw=0.5), zorder=1)
    for i, ys in enumerate(positions):
        ax.scatter([i]*len(ys), ys, s=150, facecolor='#dbeafe', edgecolor='#2563eb', zorder=3)
        for j, y in enumerate(ys):
            ax.text(i, y, str(j), ha='center', va='center', fontsize=6, zorder=4)
        label = ['Input', 'fc1 · ReLU', 'fc2 · ReLU', 'fc3 · logits'][i]
        ax.text(i, -0.03, f'{label}\n{sizes[i]} units', ha='center')
    ax.set(xlim=(-0.4, len(sizes)-0.6), ylim=(-0.15, 1.05))
    ax.axis('off')
    fig.tight_layout()
    return fig


def layer_figure(step):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    xb, z, y = step['xb'], step['z'], step['y']
    axes[0].bar(range(len(xb)), xb, color=['#3b82f6']*(len(xb)-1)+['#f59e0b'])
    axes[0].set_xticks(range(len(xb)), [str(i) for i in range(len(xb)-1)] + ['bias'])
    axes[0].set(title='Input x → append constant 1 → xb', xlabel='Input unit', ylabel='Value')
    positions = np.arange(len(z))
    axes[1].bar(positions-0.2, z, width=0.4, label='Weighted sum z', color='#64748b')
    axes[1].bar(positions+0.2, y, width=0.4, label='Activation y', color='#14b8a6')
    axes[1].set(title=f"z → {step['activation']} → y", xlabel='Output unit', ylabel='Value')
    axes[1].legend()
    for ax in axes:
        ax.axhline(0, color='#334155', linewidth=0.5)
    fig.tight_layout()
    return fig
