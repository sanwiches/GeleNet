"""Prepare deterministic train/internal-val split and MAT-derived maps without GT."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat


def canonical(a):
    _, first, inverse = np.unique(a, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int32)
    rank[np.argsort(first)] = np.arange(len(first), dtype=np.int32)
    return rank[inverse].reshape(a.shape)


def main():
    ap = argparse.ArgumentParser()
    inputs=ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest')
    inputs.add_argument('--images', help='Inference-only image directory; no GT required')
    ap.add_argument('--regions', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-count', type=int, default=140)
    ap.add_argument('--size', type=int, default=352)
    args = ap.parse_args()
    mf = Path(args.manifest).resolve() if args.manifest else None
    if mf:
        entries = json.loads(mf.read_text())
    else:
        entries = [[str(f.resolve())]*3 for f in sorted(Path(args.images).iterdir())
                   if f.is_file() and f.suffix.lower() in ('.jpg','.jpeg','.png')]
    if not isinstance(entries, list) or not entries:
        raise ValueError('Expected manifest triplets')
    samples = []
    for e in entries:
        if not isinstance(e, list) or len(e) != 3:
            raise ValueError('Each row must contain image,target,valid-mask')
        paths = [((mf.parent if mf else Path.cwd()) / x).resolve() for x in e]
        if not all(x.is_file() for x in paths):
            raise ValueError('Missing sample input')
        samples.append(paths)
    ids = [x[0].stem for x in samples]
    if len(set(ids)) != len(ids) or (mf and not 0 < args.val_count < len(ids)):
        raise ValueError('Duplicate IDs or invalid validation size')
    val_ids = set(random.Random(args.seed).sample(sorted(ids), args.val_count)) if mf else set()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    maps_out = out / 'regions'; maps_out.mkdir()
    records = []
    for paths in samples:
        image = paths[0]; mat = Path(args.regions) / (image.stem + '.mat')
        raw = mat.read_bytes()
        segs = loadmat(mat)['segs'].ravel()
        if not len(segs):
            raise ValueError('Empty hierarchy: ' + image.stem)
        with Image.open(image) as im:
            shape = (im.height, im.width)
        actual = [min(l, len(segs)) for l in (10, 20, 40)]
        maps = []
        for level in actual:
            a = np.asarray(segs[level - 1])
            if a.shape != shape or not np.issubdtype(a.dtype, np.integer):
                raise ValueError('Invalid hierarchy map: ' + image.stem)
            a = canonical(a)
            a = np.array(Image.fromarray(a).resize((args.size, args.size), Image.NEAREST))
            maps.append(a.astype(np.int32))
        np.savez_compressed(maps_out/(image.stem+'.npz'), maps=np.stack(maps),
                            actual_levels=np.asarray(actual, dtype=np.int64))
        records.append(dict(image=image.stem, available_levels=len(segs),
                            selected_levels=actual, mat_sha256=hashlib.sha256(raw).hexdigest()))
    for split, keep in ([('train', False), ('val', True)] if mf else []):
        rows = [[str(x) for x in sample] for sample in samples
                if (sample[0].stem in val_ids) == keep]
        (out/(split+'.json')).write_text(json.dumps(rows, indent=2))
    (out/'preparation.json').write_text(json.dumps(dict(
        source_manifest=str(mf), manifest_sha256=hashlib.sha256(mf.read_bytes()).hexdigest() if mf else None,
        seed=args.seed, size=args.size, requested_levels=[10,20,40],
        validation_ids=sorted(val_ids), records=records,
        validation='Held-out scribble Partial BCE; dense GT not used' if mf else 'Inference maps only'), indent=2))
    print('PREPARED', out, 'train', len(ids)-len(val_ids), 'val', len(val_ids), flush=True)


if __name__ == '__main__':
    main()
