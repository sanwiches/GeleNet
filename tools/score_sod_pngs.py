"""Score paired uint8 PNG saliency maps with pinned PySODMetrics."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from PIL import Image
import py_sod_metrics as metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    version = importlib.metadata.version('pysodmetrics')
    assert version == '1.6.2', 'Expected pysodmetrics 1.6.2, got ' + version
    pred_root, gt_root = Path(args.pred).resolve(), Path(args.gt).resolve()
    pred = {p.stem: p for p in pred_root.glob('*.png')}
    gt = {p.stem: p for p in gt_root.glob('*.png')}
    assert len(pred) == len(gt) == 600 and pred.keys() == gt.keys(), 'Expected 600 exact pairs'
    out = Path(args.output)
    assert not out.exists(), 'Refusing to overwrite output'
    fm = metrics.Fmeasure(beta=0.3)
    wfm = metrics.WeightedFmeasure(beta=1)
    sm, em, mae = metrics.Smeasure(), metrics.Emeasure(), metrics.MAE()
    checksums = []
    for i, stem in enumerate(sorted(pred), 1):
        with Image.open(pred[stem]) as im:
            assert im.mode == 'L', 'Prediction is not grayscale: ' + stem
            a = np.array(im)
        with Image.open(gt[stem]) as im:
            b = np.array(im.convert('L'))
        assert a.dtype == b.dtype == np.uint8 and a.shape == b.shape
        # Explicit agreed evaluation rule; never modify source GT files.
        b = np.where(b > 127, 255, 0).astype(np.uint8)
        for metric in (fm, wfm, sm, em, mae):
            metric.step(pred=a, gt=b)
        checksums.append([stem, hashlib.sha256(pred[stem].read_bytes()).hexdigest(),
                          hashlib.sha256(gt[stem].read_bytes()).hexdigest()])
        if i % 100 == 0:
            print('Scored', i, '/600', flush=True)
    f = fm.get_results()
    e = em.get_results()['em']
    result = dict(images=600, pysodmetrics=version, S= float(sm.get_results()['sm']),
                  wF=float(wfm.get_results()['wfm']),
                  maxF=float(f['fm']['curve'].max()), meanF=float(f['fm']['curve'].mean()),
                  adpF=float(f['fm']['adp']), maxE=float(e['curve'].max()),
                  meanE=float(e['curve'].mean()), adpE=float(e['adp']),
                  MAE=float(mae.get_results()['mae']))
    assert all(np.isfinite(result[k]) for k in ('S','wF','maxF','maxE','MAE'))
    out.mkdir(parents=True)
    (out/'metrics.json').write_text(json.dumps(result, indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(pred=str(pred_root), gt=str(gt_root),
        pysodmetrics=version, input='saved uint8 grayscale PNG', beta_squared_F=0.3,
        weightedF_beta=1, reduction='library per-image average curves; then max/mean',
        gt_preprocessing='grayscale uint8; >127 foreground 255, <=127 background 0; in memory only',
        prediction_preprocessing='saved PNG unchanged; library default preprocessing preserved',
        checkpoint_selection='not performed by this script'), indent=2))
    (out/'pair_hashes.json').write_text(json.dumps(checksums, indent=2))
    np.savez(out/'curves.npz', F=f['fm']['curve'], E=e['curve'],
             precision=f['pr']['p'], recall=f['pr']['r'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
