"""CPU-only paired visual audit of the frozen three-group predictions."""
import argparse
import csv
import html
import json
import random
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

GROUPS=('baseline','constant','staged')


def index_images(root):
    result={}
    for f in Path(root).iterdir():
        if f.is_file() and f.suffix.lower() in ('.jpg','.jpeg','.png'):
            if f.stem in result: raise ValueError('Duplicate stem: '+str(f))
            result[f.stem]=f
    return result


def diagnostic(pred,gt):
    if pred.shape!=gt.shape: raise ValueError('Prediction/GT shape mismatch')
    return dict(MAE=float(np.abs(pred-gt).mean()),
        foreground_soft_error=float((1-pred[gt]).mean()) if gt.any() else None,
        background_soft_error=float(pred[~gt].mean()) if (~gt).any() else None)


def audit(root,selection,gts):
    predictions={g:index_images(root/g/selection/'predictions') for g in GROUPS}
    for group,files in predictions.items():
        if files.keys()!=gts.keys(): raise ValueError('Prediction pairing mismatch: '+group)
    rows=[]
    for stem in sorted(gts):
        with Image.open(gts[stem]) as im: gt=np.array(im.convert('L'))>127
        row=dict(image=stem,gt_foreground_fraction=float(gt.mean()))
        for group in GROUPS:
            with Image.open(predictions[group][stem]) as im:
                if im.mode!='L': raise ValueError('Expected grayscale PNG')
                a=np.array(im,dtype=np.float64)/255
            row.update({group+'_'+k:v for k,v in diagnostic(a,gt).items()})
        row['constant_minus_baseline']=row['constant_MAE']-row['baseline_MAE']
        row['staged_minus_baseline']=row['staged_MAE']-row['baseline_MAE']
        row['staged_minus_constant']=row['staged_MAE']-row['constant_MAE']
        rows.append(row)
    return rows,predictions


def choose_cases(rows,top=3):
    ids=sorted(r['image'] for r in rows)
    categories={'random_seed42':random.Random(42).sample(ids,min(6,len(ids)))}
    for key in ('constant_minus_baseline','staged_minus_baseline','staged_minus_constant'):
        better=sorted((r for r in rows if r[key]<-1e-12),key=lambda r:(r[key],r['image']))
        worse=sorted((r for r in rows if r[key]>1e-12),key=lambda r:(-r[key],r['image']))
        categories[key+'_improved']=[r['image'] for r in better[:top]]
        categories[key+'_degraded']=[r['image'] for r in worse[:top]]
    # Previously discussed failure cases; interpretive diagnostics, not random samples.
    categories['previous_cases']=[s for s in ('1935','0301','1610','0062','0963','0754') if s in ids]
    return categories


def detail_box(gt):
    ys,xs=np.where(gt)
    if not len(xs): return None
    h,w=gt.shape
    margin=max(12,round(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)*.2))
    return (max(0,int(xs.min())-margin),max(0,int(ys.min())-margin),
            min(w,int(xs.max())+margin+1),min(h,int(ys.max())+margin+1))


def font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf'):
        if Path(path).exists(): return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def render(stem,selection,image_path,gt_path,pred_paths,row,target):
    with Image.open(image_path) as im: rgb=im.convert('RGB').copy()
    with Image.open(gt_path) as im: gt=np.array(im.convert('L'))>127
    if rgb.size!=(gt.shape[1],gt.shape[0]): raise ValueError('RGB/GT size mismatch')
    tiles=[rgb,Image.fromarray(gt.astype(np.uint8)*255)]
    for g in GROUPS:
        with Image.open(pred_paths[g][stem]) as im: tiles.append(im.convert('L').copy())
    if any(t.size!=rgb.size for t in tiles): raise ValueError('Panel size mismatch')
    box=detail_box(gt)
    w,h,head=350,280,43
    row_count=2 if box else 1
    canvas=Image.new('RGB',(w*5,64+row_count*(h+head)+32),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((8,7),f'{stem} / {selection} / delta MAE constant={row["constant_minus_baseline"]:+.5f}, staged={row["staged_minus_baseline"]:+.5f}',font=font(18),fill='black')
    draw.text((8,34),'Top: full image. Bottom: identical GT-box crop.' if box else 'Full image; GT has no foreground. No GT-box crop.',font=font(16),fill='black')
    titles=['RGB','GT >127']+[g+' MAE='+format(row[g+'_MAE'],'.5f') for g in GROUPS]
    for r in range(row_count):
        for col,(tile,title) in enumerate(zip(tiles,titles)):
            shown=tile if r==0 else tile.crop(box)
            resample=Image.NEAREST if col==1 else Image.BILINEAR
            shown=ImageOps.contain(shown.convert('RGB'),(w-10,h),resample)
            y=64+r*(h+head)
            draw.text((col*w+6,y+5),title,font=font(17),fill='black')
            canvas.paste(shown,(col*w+(w-shown.width)//2,y+head+(h-shown.height)//2))
    draw.text((8,canvas.height-25),'Saved-PNG MAE is diagnostic. Dark=background; bright=foreground. GT crop is for inspection only.',font=font(15),fill='black')
    canvas.save(target)
    return box


def main():
    ap=argparse.ArgumentParser()
    p=Path('/home/data2/XintongHu/GeleNet')
    ap.add_argument('--eval-root',default=str(p/'evaluation_runs/v2_region_threegroups_YqmWlAII'))
    ap.add_argument('--images',default=str(p/'resources/datasets/EORSSD/raw/test-images'))
    ap.add_argument('--gt',default=str(p/'resources/datasets/EORSSD/raw/test-labels'))
    ap.add_argument('--output')
    ap.add_argument('--top',type=int,default=3)
    ap.add_argument('--expected-images',type=int,default=600)
    args=ap.parse_args()
    if args.top<1: ap.error('top must be positive')
    root=Path(args.eval_root).resolve()
    images=index_images(args.images); gts=index_images(args.gt)
    if images.keys()!=gts.keys() or len(gts)!=args.expected_images:
        raise ValueError('Unexpected image/GT pairing or count')
    data={s:audit(root,s,gts) for s in ('final44','best_val')}
    # Both checkpoint choices use cases selected from final44; avoids changing
    # examples between the two views. All600 images are diagnosed in both.
    cases=choose_cases(data['final44'][0],args.top)
    selected=sorted(set(s for ids in cases.values() for s in ids))
    if args.output:
        out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    else:
        out=Path(tempfile.mkdtemp(prefix='region_threegroups_visual_',dir=root.parent))
    summary={}; render_records=[]
    for selection,(rows,pred_paths) in data.items():
        with (out/(selection+'_per_image.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        summary[selection]={}
        print('\n===',selection,'===',flush=True)
        for key in ('constant_minus_baseline','staged_minus_baseline','staged_minus_constant'):
            better=sum(r[key]<-1e-12 for r in rows); worse=sum(r[key]>1e-12 for r in rows)
            result=dict(improved=better,degraded=worse,tied=len(rows)-better-worse,
                        mean_delta=float(np.mean([r[key] for r in rows])))
            summary[selection][key]=result; print(key,result,flush=True)
        folder=out/selection; folder.mkdir()
        indexed={r['image']:r for r in rows}
        for stem in selected:
            box=render(stem,selection,images[stem],gts[stem],pred_paths,indexed[stem],folder/(stem+'.png'))
            render_records.append(dict(selection=selection,image=stem,gt_detail_box=box))
    (out/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2))
    (out/'selected_cases.json').write_text(json.dumps(cases,indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(eval_root=str(root),
        images_per_selection=len(gts),random_seed=42,case_selection='final44 only; same cases in both views',
        diagnostic='uint8 PNG /255, GT>127, no further normalization; not benchmark replacement',
        errors='Foreground mean(1-p); background mean(p); empty-class errors null',
        panels=['RGB','GT','baseline','constant','staged'],rendered=render_records),indent=2))
    readme='''三组区域监督可视化\n\n打开index.html，先看random_seed42，再看previous_cases及改善退化分类。\n每张PNG列顺序：原图、GT、V2基线、固定权重区域监督、渐进权重区域监督。\n上排完整图，下排相同GT外接框加边距的局部放大。无前景GT时无放大图。\nfinal44为统一第44轮；best_val为各组内部验证选出的轮次。两套视图采用同一批图片编号。\n统计覆盖全部600张；导出的PNG只是代表样本，不是缺失预测。\nMAE排序和前景背景误差是诊断，不取代原pysodmetrics评分。极端案例不是随机代表，不用于挑新配置。\n只读取原预测、图像和GT，不重新推理，不改原始文件，不使用GPU。\n'''
    (out/'README.txt').write_text(readme,encoding='utf-8')
    parts=['<!doctype html><meta charset="utf-8"><title>Region supervision comparison</title>',
           '<style>body{font:16px sans-serif;max-width:1750px;margin:24px auto;padding:0 12px}img{width:100%;height:auto}a{margin-right:14px}section{margin:24px 0}details{margin:18px 0}summary{cursor:pointer}p{line-height:1.6}</style>',
           '<h1>区域监督三组对照</h1><p>列顺序：原图 / GT / V2基线 / 固定权重 / 渐进权重。上排完整图，下排局部放大。</p>',
           '<p>先看固定随机样本。两种checkpoint视图使用相同图片，MAE为诊断数值。</p>']
    for category,ids in cases.items():
        parts.append('<details'+(' open' if category=='random_seed42' else '')+'><summary>'+html.escape(category)+' ('+str(len(ids))+')</summary>')
        for stem in ids:
            parts.append('<section><h3>'+html.escape(stem)+'</h3>')
            for selection in ('final44','best_val'):
                rel=selection+'/'+stem+'.png'
                parts.append('<p>'+selection+'</p><a href="'+rel+'"><img loading="lazy" src="'+rel+'"></a>')
            parts.append('</section>')
        parts.append('</details>')
    (out/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    print('\nPASS: audited',len(gts),'images per selection; rendered',len(render_records),'PNGs',flush=True)
    print('OUTPUT:',out,flush=True)


if __name__=='__main__': main()

