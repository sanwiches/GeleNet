"""Reuse frozen internal split; propagate training scribbles only at native size."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from region_targets import quartile_levels,propagate


def read_samples(path):
    rows=json.loads(path.read_text())
    if not isinstance(rows,list) or not rows:
        raise ValueError('Nonempty manifest required')
    samples=[]
    for row in rows:
        if not isinstance(row,list) or len(row)!=3:
            raise ValueError('Expected image,target,valid-mask triplets')
        paths=[(path.parent/x).resolve() for x in row]
        if not all(f.is_file() for f in paths): raise ValueError('Missing source input')
        samples.append(paths)
    return samples


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split-source',required=True)
    ap.add_argument('--regions',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--size',type=int,default=352)
    args=ap.parse_args()
    src=Path(args.split_source).resolve()
    train=read_samples(src/'train.json'); val=read_samples(src/'val.json')
    train_ids=[s[0].stem for s in train]; val_ids=[s[0].stem for s in val]
    if len(set(train_ids))!=len(train_ids) or len(set(val_ids))!=len(val_ids) or set(train_ids)&set(val_ids):
        raise ValueError('Duplicate or overlapping split IDs')
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    maps=out/'targets'; maps.mkdir()
    for split,rows in [('train',train),('val',val)]:
        (out/(split+'.json')).write_text(json.dumps([[str(f) for f in s] for s in rows],indent=2))
    records=[]
    for i,(image,target_path,mask_path) in enumerate(train,1):
        with Image.open(image) as im: native=(im.height,im.width)
        arrays=[]
        for f in [target_path,mask_path]:
            with Image.open(f) as im:
                a=np.asarray(im)
                if im.mode!='L' or a.shape!=native or not np.isin(a,[0,255]).all():
                    raise ValueError('Invalid native scribble: '+str(f))
                arrays.append((a//255).astype(np.uint8))
        target,valid=arrays
        mat=Path(args.regions)/(image.stem+'.mat')
        segs=loadmat(mat)['segs'].ravel(); levels=quartile_levels(len(segs))
        targets=[]; masks=[]; stats=[]
        for level in levels:
            y,m,st=propagate(np.asarray(segs[level-1]),target,valid)
            resized=[]
            for a in [y,m]:
                resized.append(np.array(Image.fromarray(a).resize((args.size,args.size),Image.NEAREST)))
            targets.append(resized[0]); masks.append(resized[1]); stats.append(st)
        np.savez_compressed(maps/(image.stem+'.npz'),targets=np.stack(targets),
            masks=np.stack(masks),levels=np.asarray(levels,dtype=np.int64))
        records.append(dict(image=image.stem,available_levels=len(segs),levels=levels,
            duplicate_levels=len(set(levels))<3,statistics=stats,
            image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            mat_sha256=hashlib.sha256(mat.read_bytes()).hexdigest(),
            target_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(),
            valid_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest()))
        if i%100==0 or i==len(train): print('Prepared',i,'/',len(train),flush=True)
    summary=dict(size=args.size,train_count=len(train),val_count=len(val),
        split_source=str(src),split_hashes={name:hashlib.sha256((src/name).read_bytes()).hexdigest()
            for name in ['train.json','val.json']},
        policy='Native pure-label propagation; conflicts/empty ignored; observed pixels excluded',
        weights=[0.5,0.1,0.15,0.25],dense_gt_used=False,records=records)
    (out/'preparation.json').write_text(json.dumps(summary,indent=2))
    print('PREPARED',out,'train',len(train),'val',len(val),flush=True)


if __name__=='__main__': main()
