"""Matched fixed20/uniform/adaptive pilot with explicit seed and internal validation."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model.adaptive_GeleNet import ScaleGeleNet
from scale_data import ScaleDataset
from train_v3_GeleNet import read_samples
from weak_loss import partial_bce_with_logits
from utils import adjust_lr, clip_gradient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--mode', choices=['fixed20','uniform','adaptive'], required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=44, help='Actual epoch count, inclusive')
    ap.add_argument('--batchsize', type=int, default=8)
    ap.add_argument('--size', type=int, default=352)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max-steps', type=int, default=0, help='Smoke only; 0 means full epochs')
    args = ap.parse_args()
    if args.epochs < 1 or args.max_steps < 0:
        ap.error('Invalid epoch/step count')
    if not torch.cuda.is_available():
        ap.error('CUDA required')
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prep = Path(args.prepared).resolve()
    preparation = json.loads((prep/'preparation.json').read_text())
    if preparation['size'] != args.size:
        raise ValueError('Prepared size does not match training size')
    train_samples, val_samples = (read_samples(prep/(s+'.json')) for s in ('train','val'))
    if {s[0].stem for s in train_samples} & {s[0].stem for s in val_samples}:
        raise ValueError('Train/val overlap')
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'checkpoints').mkdir()
    state = out/'status.json'
    state.write_text(json.dumps(dict(state='running')))
    try:
        config = vars(args).copy()
        config.update(torch=torch.__version__, train_count=len(train_samples),
                      val_count=len(val_samples), loss='Partial BCE', RC=False,
                      optimizer='Adam', decay_epoch=30, decay_rate=0.1, clip=0.5,
                      validation='pixel-weighted held-out scribble BCE',
                      smoke=bool(args.max_steps),
                      note='Pilot uses1260/140 split; old1400-image runs are historical only')
        hashes = {}
        for rel in ['train_scale_GeleNet.py','scale_data.py','model/scale_prior.py',
                    'model/adaptive_GeleNet.py','model/GeleNet_models.py',
                    'weak_data.py','weak_loss.py','utils.py','model/pvt_v2_b2.pth']:
            f=Path(rel)
            hashes[rel]=hashlib.sha256(f.read_bytes()).hexdigest()
        for name in ['train.json','val.json','preparation.json']:
            hashes['prepared/'+name]=hashlib.sha256((prep/name).read_bytes()).hexdigest()
        config['hashes']=hashes
        (out/'config.json').write_text(json.dumps(config,indent=2))
        model=ScaleGeleNet(mode=args.mode).cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        # Dedicated generator keeps batch order matched across all three modes.
        generator=torch.Generator(); generator.manual_seed(args.seed)
        loader=DataLoader(ScaleDataset(train_samples,args.size,prep/'regions'),
                          batch_size=args.batchsize,shuffle=True,num_workers=0,
                          pin_memory=True,generator=generator)
        val=DataLoader(ScaleDataset(val_samples,args.size,prep/'regions'),
                       batch_size=args.batchsize,shuffle=False,num_workers=0)
        best=float('inf')
        with (out/'epochs.jsonl').open('w') as log:
            for epoch in range(1,args.epochs+1):
                adjust_lr(optimizer,args.lr,epoch,0.1,30)
                model.train(); loss_sum=0.; steps=0
                for batch in loader:
                    images=batch['image'].cuda(); ids=batch['region_ids'].cuda()
                    target=batch['target'].cuda(); valid=batch['valid_mask'].cuda()
                    optimizer.zero_grad()
                    logits,_=model(images,region_ids=ids)
                    loss=partial_bce_with_logits(logits,target,valid)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite train loss')
                    loss.backward(); clip_gradient(optimizer,0.5); optimizer.step()
                    loss_sum+=loss.item(); steps+=1
                    if steps % 20==0: print('TRAIN',args.mode,epoch,steps,loss.item(),flush=True)
                    if args.max_steps and steps>=args.max_steps: break
                model.eval(); total=0.; count=0.; weights=[]
                with torch.no_grad(), (out/'validation_weights.jsonl').open('a') as wf:
                    for batch in val:
                        target=batch['target'].cuda(); mask=batch['valid_mask'].cuda()
                        logits,_=model(batch['image'].cuda(),region_ids=batch['region_ids'].cuda())
                        loss=partial_bce_with_logits(logits,target,mask)
                        n=mask.sum().item(); total+=loss.item()*n; count+=n
                        prior=model.KTM_23.scale_prior
                        ws=prior.last_weights.cpu().tolist(); active=prior.last_active.cpu().tolist()
                        for stem,w,a,levels in zip(batch['id'],ws,active,batch['actual_levels'].tolist()):
                            wf.write(json.dumps(dict(epoch=epoch,image=stem,weights=w,
                                active_candidates=a,actual_levels=levels))+'\n')
                        weights.extend(ws)
                if count==0: raise ValueError('No validation scribble pixels')
                value=total/count
                if not np.isfinite(value): raise RuntimeError('Nonfinite validation loss')
                row=dict(epoch=epoch,train_BCE=loss_sum/steps,val_BCE=value,
                         mean_weights=np.mean(weights,axis=0).tolist())
                log.write(json.dumps(row)+'\n'); log.flush(); print(row,flush=True)
                if value<best:
                    best=value
                    torch.save(model.state_dict(),out/'checkpoints/best_val.pth',
                               _use_new_zipfile_serialization=False)
                    (out/'best_val.json').write_text(json.dumps(row,indent=2))
                if (epoch+1)%5==0 or epoch==args.epochs:
                    torch.save(model.state_dict(),out/('checkpoints/GeleNet.pth.'+str(epoch)),
                               _use_new_zipfile_serialization=False)
        state.write_text(json.dumps(dict(state='smoke_completed' if args.max_steps else 'completed')))
    except BaseException as exc:
        state.write_text(json.dumps(dict(state='failed',error=str(exc))))
        raise


if __name__=='__main__':
    main()
