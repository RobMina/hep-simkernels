"""Two-sample comparison helpers for slab samples (fast vs a reference such as a Geant4 slab run): per-bin and total
chi2 of two samples of any sizes, equal-probability bin edges, and a summary of a sample. NumPy only."""
import numpy as np


def chi2_bins(a, b, edges):
    """Per-bin two-sample chi2 terms (0 for empty bins), bin mask with entries, and the ratio b/a scaled to equal
    sample sizes (nan where a is empty)."""
    ha, hb = np.histogram(a, edges)[0].astype(float), np.histogram(b, edges)[0].astype(float)
    na, nb = ha.sum(), hb.sum()
    m = (ha + hb) > 0
    if na == 0 or nb == 0:
        return np.full(ha.shape, np.nan), m, np.full(ha.shape, np.nan)
    num = (np.sqrt(nb / na) * ha - np.sqrt(na / nb) * hb) ** 2
    terms = np.where(m, num / np.where(m, ha + hb, 1.0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (hb / nb) / (ha / na)
    return terms, m, ratio


def chi2(a, b, edges):
    """Two-sample chi2 and number of bins with entries for samples a, b of any sizes."""
    terms, m, _ = chi2_bins(a, b, edges)
    return [float(terms.sum()), int(m.sum())]


def quantile_edges(x, nbins, lo=None, hi=None):
    """Bin edges at equal-probability quantiles of x, widened to [lo, hi]."""
    e = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    if lo is not None:
        e[0] = min(e[0], lo)
    if hi is not None:
        e[-1] = max(e[-1], hi)
    return e


def summary(x):
    if x.size == 0:
        return dict(n=0)
    q = np.quantile(x, [0.1, 0.5, 0.9, 0.99])
    return dict(n=int(x.size), mean=float(x.mean()), err=float(x.std() / np.sqrt(x.size)),
                rms=float(np.sqrt((x * x).mean())),
                q10=float(q[0]), q50=float(q[1]), q90=float(q[2]), q99=float(q[3]))
