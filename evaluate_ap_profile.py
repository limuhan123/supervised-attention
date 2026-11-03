"""Read-only AP-versus-IoU evaluation of existing checkpoints; no training/selection."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import torch
from torch.utils.data import DataLoader
import network
from network.ap_profile import profile_records
from network.train_base import collect_map_records
from train import get_eval_preds, set_seed


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def evaluate(folder, output, device):
    start = time.time()
    ckpt = torch.load(folder/'best.pth', weights_only=False, map_location='cpu')
    cfg = ckpt['config']
    set_seed(cfg['seed'])
    model = getattr(network, cfg['model_name'])(nc=cfg['num_classes'], scales=cfg['depth'])
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device).eval()
    dataset = network.YOLO_Dataset(cfg['data_path'], mode='val', image_size=cfg['image_size'],
                                   label_filter_classes=cfg['label_filter_classes'])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False,
                        num_workers=0, collate_fn=dataset.collate_fn)
    preds, gts, offset = [], [], 0
    with torch.inference_mode():
        for batch in loader:
            images = batch['images'].to(device)
            with torch.autocast(device.type, enabled=cfg['use_amp'] and device.type=='cuda'):
                result = get_eval_preds(model(images))
            pp, gg = collect_map_records(result, batch['targets'], cfg['num_classes'],
                img_size=cfg['image_size'], image_offset=offset,
                conf_threshold=cfg['val_conf_threshold'], nms_iou_threshold=cfg['val_nms_iou_threshold'],
                pre_nms_topk=cfg['val_pre_nms_topk'], max_det=cfg['val_max_det'])
            preds.extend(pp); gts.extend(gg); offset += len(images)
    profile = profile_records(preds, gts, cfg['num_classes'])
    original = json.loads((folder/'best_metrics.json').read_text())
    profile.update(run=folder.name, checkpoint=str(folder/'best.pth'), seed=cfg['seed'],
        validation_images=offset, prediction_count=len(preds), elapsed_seconds=time.time()-start,
        original_AP50=original['map50'], original_AP50_95=original['map50_95'],
        reevaluation_delta_AP50=profile['AP50']-original['map50'],
        reevaluation_delta_AP50_95=profile['AP50_95']-original['map50_95'],
        device=str(device), amp=cfg['use_amp'] and device.type=='cuda',
        interpretation='IoU-band AP differences are descriptive, not a causal decomposition of localization versus classification.')
    save(output/(folder.name+'.json'), profile)
    return profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', nargs='+', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--groups', nargs='+', default=['A0','A1','A2','A3','E0','E1'])
    p.add_argument('--device', default='cpu')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    folders = {}
    for root in args.roots:
        for folder in sorted(root.glob('*_seed*')):
            if folder.name.split('_')[0] not in args.groups: continue
            if not (folder/'status.json').exists(): continue
            if json.loads((folder/'status.json').read_text())['state'] != 'complete': continue
            if folder.name in folders: raise ValueError('Duplicate run '+folder.name)
            folders[folder.name] = folder
    if not folders: raise ValueError('No complete runs found')
    rows=[]
    save(args.output/'status.json',dict(state='running',total=len(folders),completed=0))
    for name, folder in sorted(folders.items()):
        print('EVALUATE',name,flush=True)
        try:
            row=evaluate(folder,args.output,device)
        except Exception as exc:
            save(args.output/'status.json',dict(state='failed',total=len(folders),
                 completed=len(rows),failed_run=name,error=repr(exc)))
            raise
        rows.append(row)
        print('DONE',name,'AP75',row['AP75'],'AP50_95',row['AP50_95'],
              'delta_from_original',row['reevaluation_delta_AP50_95'],flush=True)
        save(args.output/'status.json',dict(state='running',total=len(folders),completed=len(rows),last=name))
    fields=['run','seed','AP50','AP75','AP90','AP50_95','AP50_70','AP75_95',
            'original_AP50_95','reevaluation_delta_AP50_95']
    with (args.output/'per_run.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    summary={}
    for group in args.groups:
        rr=[r for r in rows if r['run'].split('_')[0]==group]
        if not rr:continue
        summary[group]={'n':len(rr),'metrics':{k:{'mean':statistics.mean(r[k] for r in rr),
            'sample_sd':statistics.stdev(r[k] for r in rr) if len(rr)>1 else None}
            for k in ['AP50','AP75','AP90','AP50_95','AP50_70','AP75_95']},
            'AP_per_iou_mean':[statistics.mean(r['AP_per_iou'][i] for r in rr) for i in range(10)]}
    save(args.output/'summary.json',summary)
    save(args.output/'status.json',dict(state='complete',total=len(folders),completed=len(rows)))


if __name__=='__main__': main()
