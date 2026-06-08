#!/usr/bin/env python3
"""Batch VideoSeal encoder with sequential watermark IDs, same CLI idea as old project."""
import argparse, csv, subprocess, sys, time, json
from pathlib import Path
from datetime import datetime

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}

def generate_watermark_id(sequence: int, base_date: str | None = None) -> str:
    if base_date:
        if len(base_date) != 6 or not base_date.isdigit():
            raise ValueError('base_date must be DDMMYY')
        date_part = base_date
    else:
        date_part = datetime.now().strftime('%d%m%y')
    if not 0 <= sequence <= 999:
        raise ValueError('sequence must be 0-999')
    return date_part + f'{sequence:03d}'

def get_video_info(video_path: Path):
    try:
        cmd = ['ffprobe','-v','quiet','-print_format','json','-show_format','-show_streams',str(video_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0: return {}
        data=json.loads(r.stdout)
        vs=next((s for s in data.get('streams',[]) if s.get('codec_type')=='video'), {})
        return {
            'duration_sec': float(data.get('format',{}).get('duration',0) or 0),
            'resolution': f"{vs.get('width',0)}x{vs.get('height',0)}",
            'codec': vs.get('codec_name','unknown')
        }
    except Exception:
        return {}

def encode_one(input_path: Path, output_path: Path, watermark_id: str, args):
    t0=time.time()
    cmd=[sys.executable, 'encode.py', '--input', str(input_path), '--output', str(output_path), '--id', watermark_id, '--model', args.model]
    if args.no_gpu: cmd.append('--no-gpu')
    if args.first_minute: cmd.append('--first-minute')
    if args.scaling_w is not None: cmd += ['--scaling-w', str(args.scaling_w)]
    if args.step_size is not None: cmd += ['--step-size', str(args.step_size)]
    print(f"Encoding {input_path.name} -> {watermark_id}")
    r=subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent, timeout=args.timeout)
    status='SUCCESS' if r.returncode==0 else 'FAILED'
    err='' if r.returncode==0 else (r.stderr[-500:] or r.stdout[-500:])
    return {'input': input_path.name, 'output': output_path.name, 'watermark_id': watermark_id, 'status': status, 'duration': time.time()-t0, 'error': err}

def main():
    p=argparse.ArgumentParser(description='Batch encode videos with VideoSeal and sequential IDs')
    p.add_argument('--input-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--date')
    p.add_argument('--start-sequence', type=int, default=1)
    p.add_argument('--model', default='ckpts/y_256b_img.jit')
    p.add_argument('--first-minute', action='store_true')
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--scaling-w', type=float)
    p.add_argument('--step-size', type=int)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--summary', type=Path)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos=sorted({v for ext in VIDEO_EXTENSIONS for v in list(args.input_dir.glob(f'*{ext}'))+list(args.input_dir.glob(f'*{ext.upper()}'))})
    if not videos:
        print(f'No videos found in {args.input_dir}')
        sys.exit(1)
    results=[]
    for i,v in enumerate(videos,1):
        wm_id=generate_watermark_id(args.start_sequence+i-1, args.date)
        out=args.output_dir / f'{v.stem}_watermarked{v.suffix}'
        row=encode_one(v,out,wm_id,args)
        row.update(get_video_info(v))
        results.append(row)
    summary=args.summary or (args.output_dir/'encoding_summary.csv')
    fields=['input','output','watermark_id','status','duration','duration_sec','resolution','codec','error']
    with open(summary,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k,'') for k in fields} for r in results)
    ok=sum(r['status']=='SUCCESS' for r in results)
    print(f'\nDone: {ok}/{len(results)} successful. Summary: {summary}')
    sys.exit(0 if ok==len(results) else 1)

if __name__=='__main__': main()
