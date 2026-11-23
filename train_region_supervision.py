"""Unmodified V2 architecture with baseline or multi-level region-supervision loss."""
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model.GeleNet_models import GeleNet
from weak_data import WeakSalObjDataset
from weak_loss import partial_bce_with_logits
from region_supervision_data import RegionSupervisionDataset
from region_supervision_loss import region_supervision_loss,WEIGHTS
from region_weight_schedule import epoch_weights,INITIAL_WEIGHTS
from utils import adjust_lr,clip_gradient


def samples(path):
    rows=json.loads(path.read_text())
    if not isinstance(rows,list) or not rows: raise ValueError('Empty manifest')
    result=[]
    for row in rows:
        if not isinstance(row,list) or len(row)!=3: raise ValueError('Invalid triplet')
        result.append(tuple((path.parent/x).resolve() for x in row))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--mode',choices=['baseline','region'],required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--epochs',type=int,default=44)
    ap.add_argument('--batchsize',type=int,default=8)
    ap.add_argument('--size',type=int,default=352)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--max-steps',type=int,default=0)
    ap.add_argument('--weight-schedule',choices=['staged','constant'],default='staged')
    ap.add_argument('--warmup-epochs',type=int,default=10)
    ap.add_argument('--ramp-epochs',type=int,default=10)
    args=ap.parse_args()
    if not torch.cuda.is_available(): ap.error('CUDA required')
    if args.epochs<1 or args.max_steps<0 or args.batchsize<1: ap.error('Invalid counts')
    if args.warmup_epochs<0 or args.ramp_epochs<1: ap.error('Invalid weight schedule')
    if (args.mode=='region' and args.weight_schedule=='staged' and not args.max_steps
            and args.epochs<args.warmup_epochs+args.ramp_epochs):
        ap.error('Full region run must reach final scheduled weights')
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    prep=Path(args.prepared).resolve()
    meta=json.loads((prep/'preparation.json').read_text())
    if meta['size']!=args.size: raise ValueError('Preparation size mismatch')
    train=samples(prep/'train.json'); val=samples(prep/'val.json')
    if set(x[0].stem for x in train)&set(x[0].stem for x in val): raise ValueError('Split overlap')
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    (out/'checkpoints').mkdir(); status=out/'status.json'
    status.write_text(json.dumps(dict(state='running')))
    try:
        config=vars(args).copy()
        config.update(architecture='V2 unmodified GeleNet',KTM_region_bias=False,RC=False,
            train_count=len(train),val_count=len(val),torch=torch.__version__,
            validation='Original held-out scribble BCE only',smoke=bool(args.max_steps),
            final_loss_weights=list(WEIGHTS) if args.mode=='region' else [1,0,0,0],
            initial_loss_weights=list(INITIAL_WEIGHTS) if args.mode=='region' and args.weight_schedule=='staged' else (list(WEIGHTS) if args.mode=='region' else [1,0,0,0]),
            weights_by_epoch=[list(epoch_weights(e,args.weight_schedule,args.warmup_epochs,args.ramp_epochs))
                if args.mode=='region' else [1,0,0,0] for e in range(1,args.epochs+1)],
            clip=0.5,decay_epoch=30,decay_rate=0.1,optimizer='Adam')
        files=list(Path('model').glob('*.py'))+[
            Path(x) for x in ['train_region_supervision.py','region_supervision_data.py',
             'region_supervision_loss.py','region_weight_schedule.py','weak_data.py','weak_loss.py','utils.py',
             'model/pvt_v2_b2.pth']]+[prep/x for x in ['preparation.json','train.json','val.json']]
        config['hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        (out/'config.json').write_text(json.dumps(config,indent=2))
        model=GeleNet().cuda()
        # Fail if the selected code root accidentally contains the V3 architecture.
        if any('region_strength' in k for k in model.state_dict()):
            raise ValueError('Expected V2 architecture without region_strength')
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        ds=(RegionSupervisionDataset(train,args.size,prep/'targets') if args.mode=='region'
            else WeakSalObjDataset(train,args.size))
        generator=torch.Generator(); generator.manual_seed(args.seed)
        loader=DataLoader(ds,batch_size=args.batchsize,shuffle=True,num_workers=0,
            pin_memory=True,generator=generator)
        val_loader=DataLoader(WeakSalObjDataset(val,args.size),batch_size=args.batchsize,
            shuffle=False,num_workers=0)
        best=float('inf')
        with (out/'epochs.jsonl').open('w') as log:
            for epoch in range(1,args.epochs+1):
                weights=(epoch_weights(epoch,args.weight_schedule,args.warmup_epochs,args.ramp_epochs)
                    if args.mode=='region' else (1.,0.,0.,0.))
                print('LOSS_WEIGHTS',epoch,weights,flush=True)
                adjust_lr(optimizer,args.lr,epoch,0.1,30)
                model.train(); totals=[]; raw_terms=[]; coverage=[]
                for step,batch in enumerate(loader,1):
                    optimizer.zero_grad()
                    logits,_=model(batch['image'].cuda())
                    target=batch['target'].cuda(); valid=batch['valid_mask'].cuda()
                    if args.mode=='region':
                        rm=batch['region_masks'].cuda()
                        loss,terms=region_supervision_loss(logits,target,valid,
                            batch['region_targets'].cuda(),rm,weights=weights)
                        coverage.append(rm.mean(dim=(0,2,3)).cpu().tolist())
                    else:
                        loss=partial_bce_with_logits(logits,target,valid)
                        terms=torch.stack([loss,loss*0,loss*0,loss*0])
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    loss.backward(); clip_gradient(optimizer,0.5); optimizer.step()
                    totals.append(loss.item()); raw_terms.append(terms.detach().cpu().tolist())
                    if step%20==0 or step==len(loader):
                        print('TRAIN',args.mode,epoch,step,loss.item(),flush=True)
                    if args.max_steps and step>=args.max_steps: break
                model.eval(); numerator=0.; denominator=0.
                with torch.no_grad():
                    for batch in val_loader:
                        logits,_=model(batch['image'].cuda())
                        target=batch['target'].cuda(); valid=batch['valid_mask'].cuda()
                        n=valid.sum().item()
                        numerator+=partial_bce_with_logits(logits,target,valid).item()*n
                        denominator+=n
                if not denominator: raise ValueError('No validation annotations')
                score=numerator/denominator
                if not np.isfinite(score): raise RuntimeError('Nonfinite validation loss')
                row=dict(epoch=epoch,train_total=float(np.mean(totals)),
                    loss_weights=list(weights),
                    raw_BCE_terms=np.mean(raw_terms,axis=0).tolist(),
                    weighted_BCE_terms=(np.mean(raw_terms,axis=0)*np.asarray(weights)).tolist(),
                    expanded_coverage=np.mean(coverage,axis=0).tolist() if coverage else [],
                    val_scribble_BCE=score)
                print(row,flush=True); log.write(json.dumps(row)+'\n'); log.flush()
                if score<best:
                    best=score
                    torch.save(model.state_dict(),out/'checkpoints/best_val.pth',
                        _use_new_zipfile_serialization=False)
                    (out/'best_val.json').write_text(json.dumps(row,indent=2))
                if (epoch+1)%5==0 or epoch==args.epochs:
                    torch.save(model.state_dict(),out/('checkpoints/GeleNet.pth.'+str(epoch)),
                        _use_new_zipfile_serialization=False)
        status.write_text(json.dumps(dict(state='smoke_completed' if args.max_steps else 'completed')))
    except BaseException as exc:
        status.write_text(json.dumps(dict(state='failed',error=str(exc))))
        raise


if __name__=='__main__': main()
