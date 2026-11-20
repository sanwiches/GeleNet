"""Export scale-model predictions and per-image weights; scoring stays separate."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.adaptive_GeleNet import ScaleGeleNet


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',required=True)
    ap.add_argument('--config',required=True,help='Training config pins the scale mode')
    ap.add_argument('--images',required=True)
    ap.add_argument('--prepared-regions',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    config=json.loads(Path(args.config).read_text())
    if config.get('smoke'): raise ValueError('Smoke weights are not experiment results')
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    pred=out/'predictions'; pred.mkdir()
    model=ScaleGeleNet(mode=config['mode']).cuda()
    model.load_state_dict(torch.load(args.weights,map_location='cpu'),strict=True)
    model.eval(); size=config['size']
    transform=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    files={}
    for f in Path(args.images).iterdir():
        if f.is_file() and f.suffix.lower() in ('.jpg','.jpeg','.png'):
            if f.stem in files: raise ValueError('Duplicate image stem')
            files[f.stem]=f
    if not files: raise ValueError('No images')
    records=[]
    with torch.no_grad():
        for stem,f in sorted(files.items()):
            with Image.open(f) as im:
                native=(im.height,im.width); x=transform(im.convert('RGB')).unsqueeze(0).cuda()
            with np.load(Path(args.prepared_regions)/(stem+'.npz'),allow_pickle=False) as z:
                maps=z['maps']; actual=z['actual_levels'].tolist()
            if maps.shape!=(3,size,size) or not np.issubdtype(maps.dtype,np.integer):
                raise ValueError('Invalid prepared map: '+stem)
            logits,_=model(x,region_ids=torch.from_numpy(maps.astype(np.int64)).unsqueeze(0).cuda())
            probability=F.interpolate(logits,size=native,mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()
            probability=(probability-probability.min())/(probability.max()-probability.min()+1e-8)
            Image.fromarray(np.uint8(np.clip(probability*255+0.5,0,255))).save(pred/(stem+'.png'))
            prior=model.KTM_23.scale_prior
            records.append(dict(image=stem,weights=prior.last_weights[0].cpu().tolist(),
                active_candidates=prior.last_active[0].cpu().tolist(),actual_levels=actual))
    (out/'scale_weights.json').write_text(json.dumps(records,indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(mode=config['mode'],size=size,
        normalization='resize logits, sigmoid, minmax, round uint8',images=len(records)),indent=2))
    print('PREDICTED',len(records),out,flush=True)


if __name__=='__main__': main()
