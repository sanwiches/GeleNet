"""Evaluate matched scale pilots using their frozen training code and scorer."""
import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT = Path('/home/data2/XintongHu/GeleNet')
MODES = ('fixed20', 'uniform', 'adaptive')
EPOCHS = (4, 9, 14, 19, 24, 29, 34, 39, 44)
METRICS = ('S', 'wF', 'maxF', 'maxE', 'MAE')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def make_jobs(best_epochs, all_checkpoints):
    jobs = [(mode, 'best_val', best_epochs[mode]) for mode in MODES]
    jobs.extend((mode, 'epoch_44', 44) for mode in MODES)
    if all_checkpoints:
        jobs.extend((mode, 'epoch_' + str(epoch), epoch)
                    for mode in MODES for epoch in EPOCHS[:-1])
    return jobs


def gate_statistics(records):
    complete = [record['weights'] for record in records
                if all(record['active_candidates'])]
    result = dict(images=len(records), all_three_active=len(complete),
                  deduplicated_images=len(records) - len(complete))
    if complete:
        result.update(level_order=[10, 20, 40],
                      mean=[statistics.mean(column) for column in zip(*complete)],
                      std=[statistics.pstdev(column) for column in zip(*complete)],
                      minimum=[min(column) for column in zip(*complete)],
                      maximum=[max(column) for column in zip(*complete)])
    return result


def preflight(args):
    if args.output.exists():
        raise ValueError('Output must be a new path: ' + str(args.output))
    if (args.run / 'status.txt').read_text().strip() != 'COMPLETED':
        raise ValueError('Training batch is not completed')
    ast.parse(args.scorer.read_text(encoding='utf-8'))
    if sha256(args.scorer) != sha256(args.reference_scorer):
        raise ValueError('Scorer differs from the historical completed V3 evaluation')
    expected_inputs = None
    cached_hashes = {}
    best_epochs = {}
    configs = {}
    for mode in MODES:
        run = args.run / mode
        config = read_json(run / 'config.json')
        if (config['mode'] != mode or config.get('smoke') is not False
                or config['epochs'] != 44 or config['size'] != 352
                or config['seed'] != 42
                or (config['train_count'], config['val_count']) != (1260, 140)
                or read_json(run / 'status.json')['state'] != 'completed'):
            raise ValueError('Unexpected training configuration: ' + mode)
        rows = [json.loads(line) for line in (run / 'epochs.jsonl').read_text().splitlines()]
        if [row['epoch'] for row in rows] != list(range(1, 45)):
            raise ValueError('Incomplete epoch history: ' + mode)
        if not all(math.isfinite(row['val_BCE']) for row in rows):
            raise ValueError('Nonfinite validation history: ' + mode)
        best = read_json(run / 'best_val.json')
        expected_best = min(rows, key=lambda row: row['val_BCE'])
        if (best['epoch'] != expected_best['epoch']
                or best['val_BCE'] != expected_best['val_BCE']):
            raise ValueError('best_val does not match validation history: ' + mode)
        for relative, expected_hash in config['hashes'].items():
            if relative.startswith('prepared/'):
                source = Path(config['prepared']) / relative.split('/', 1)[1]
            else:
                source = args.run / 'code' / relative
            if source not in cached_hashes:
                cached_hashes[source] = sha256(source)
            if cached_hashes[source] != expected_hash:
                raise ValueError('Training source/input changed: ' + str(source))
        signature = {key: value for key, value in config.items()
                     if key not in ('mode', 'output')}
        if expected_inputs is not None and signature != expected_inputs:
            raise ValueError('Modes do not have matched training configurations')
        expected_inputs = signature
        best_epochs[mode] = best['epoch']
        configs[mode] = config
    jobs = make_jobs(best_epochs, args.all_checkpoints)
    for mode, selection, epoch in jobs:
        name = 'best_val.pth' if selection == 'best_val' else 'GeleNet.pth.' + str(epoch)
        if not (args.run / mode / 'checkpoints' / name).is_file():
            raise ValueError('Missing checkpoint: ' + mode + '/' + name)
    images = [path for path in args.images.iterdir()
              if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    image_ids = {path.stem for path in images}
    gt_ids = {path.stem for path in args.gt.glob('*.png')}
    if len(images) != 600 or len(image_ids) != 600 or image_ids != gt_ids:
        raise ValueError('Expected 600 exact image/GT pairs')
    if not all((args.regions / (stem + '.mat')).is_file() for stem in image_ids):
        raise ValueError('Missing test hierarchy')
    return configs, jobs, image_ids


def run_command(command, log, cwd):
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([str(value) for value in command], cwd=str(cwd),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def save_summary(output, rows):
    with (output / 'all_checkpoints.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = [row for row in rows if row['selection'] in ('best_val', 'epoch_44')]
    write_json(output / 'primary_results.json', primary)
    stability = {}
    for mode in MODES:
        window = [row for row in rows if row['mode'] == mode
                  and row['selection'] == 'epoch_' + str(row['epoch'])
                  and row['epoch'] in (24, 29, 34, 39, 44)]
        if len(window) == 5:
            stability[mode] = {metric: dict(
                mean=statistics.mean(row[metric] for row in window),
                std_population=statistics.pstdev(row[metric] for row in window))
                for metric in METRICS}
    write_json(output / 'late_window_stability.json', dict(
        epochs=[24, 29, 34, 39, 44], results=stability,
        interpretation='Checkpoint variation in one seeded run; not multi-seed robustness'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=PROJECT / 'experiments/v3_scale_seor_train_mPOzDLop')
    parser.add_argument('--regions', type=Path,
                        default=PROJECT / 'experiments/namlab_eorrsd_test_vuCvT51X/regions')
    parser.add_argument('--images', type=Path, default=PROJECT / 'resources/datasets/EORSSD/raw/test-images')
    parser.add_argument('--gt', type=Path, default=PROJECT / 'resources/datasets/EORSSD/raw/test-labels')
    parser.add_argument('--scorer', type=Path, default=PROJECT / 'experiments/score_sod_pngs_gt127.py')
    parser.add_argument('--reference-scorer', type=Path,
                        default=PROJECT / 'evaluation_runs/v3_both_epoch44_local_XHsQg5ad/score_used.py')
    parser.add_argument('--score-python', type=Path, default=PROJECT / 'envs/sod_eval_162/bin/python')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-checkpoints', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    configs, jobs, image_ids = preflight(args)
    subprocess.run([str(args.score_python), '-c',
                    'from importlib.metadata import version; '
                    'assert version("pysodmetrics") == "1.6.2"'], check=True)
    print('PREFLIGHT_OK jobs=' + str(len(jobs)), flush=True)
    if args.preflight_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'status.json', dict(state='running', jobs=len(jobs)))
    try:
        code = args.output / 'code'
        shutil.copytree(args.run / 'code', code, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(args.scorer, args.output / 'score_used.py')
        shutil.copy2(__file__, args.output / 'evaluate_used.py')
        shutil.copy2(args.run / 'environment.txt', args.output / 'training_environment.txt')
        shutil.copy2(args.run / 'source_hashes.json', args.output / 'training_source_hashes.json')
        inputs = args.output / 'training_records'
        inputs.mkdir()
        for mode in MODES:
            destination = inputs / mode
            destination.mkdir()
            for name in ('config.json', 'best_val.json', 'epochs.jsonl', 'validation_weights.jsonl', 'status.json'):
                shutil.copy2(args.run / mode / name, destination / name)
        write_json(args.output / 'protocol.json', dict(
            run=str(args.run), requested_levels=[10, 20, 40], size=352,
            selection='best_val: minimum held-out scribble BCE; epoch_44: fixed final',
            other_checkpoints='Descriptive trajectories only; no test-based checkpoint selection',
            gt='uint8 grayscale; >127 foreground, <=127 background; scoring only',
            pysodmetrics='1.6.2', scoring_script_sha256=sha256(args.output / 'score_used.py'),
            reference_scorer=str(args.reference_scorer),
            jobs=[dict(mode=mode, selection=selection, epoch=epoch) for mode, selection, epoch in jobs]))
        with (args.output / 'scoring_environment.txt').open('w') as environment:
            subprocess.run([str(args.score_python), '-m', 'pip', 'freeze'], stdout=environment, check=True)
        run_command([sys.executable, code / 'tools/prepare_scale_experiment.py',
                     '--images', args.images, '--regions', args.regions,
                     '--output', args.output / 'prepared_test', '--size', '352'],
                    args.output / 'preparation.log', code)
        rows = []
        for index, (mode, selection, epoch) in enumerate(jobs, 1):
            print('START', index, '/', len(jobs), mode, selection, 'epoch', epoch, flush=True)
            write_json(args.output / 'current.json', dict(mode=mode, selection=selection, epoch=epoch))
            destination = args.output / mode / selection
            destination.mkdir(parents=True)
            filename = 'best_val.pth' if selection == 'best_val' else 'GeleNet.pth.' + str(epoch)
            weights = args.run / mode / 'checkpoints' / filename
            write_json(destination / 'inputs.json', dict(
                checkpoint=str(weights), checkpoint_sha256=sha256(weights),
                config_sha256=sha256(inputs / mode / 'config.json')))
            run_command([sys.executable, code / 'tools/predict_scale_GeleNet.py',
                         '--weights', weights, '--config', inputs / mode / 'config.json',
                         '--images', args.images, '--prepared-regions', args.output / 'prepared_test/regions',
                         '--output', destination / 'export'], destination / 'inference.log', code)
            predictions = destination / 'export/predictions'
            if {path.stem for path in predictions.glob('*.png')} != image_ids:
                raise ValueError('Incomplete predictions: ' + str(destination))
            run_command([args.score_python, args.output / 'score_used.py',
                         '--pred', predictions, '--gt', args.gt,
                         '--output', destination / 'metrics'], destination / 'scoring.log', code)
            result = read_json(destination / 'metrics/metrics.json')
            if result['images'] != 600 or result['pysodmetrics'] != '1.6.2':
                raise ValueError('Unexpected scoring protocol')
            test_weights = read_json(destination / 'export/scale_weights.json')
            with (inputs / mode / 'validation_weights.jsonl').open() as stream:
                validation_weights = [record for record in map(json.loads, stream) if record['epoch'] == epoch]
            write_json(destination / 'gate_diagnostics.json', dict(
                validation=gate_statistics(validation_weights), test=gate_statistics(test_weights),
                note='Weight variation is summarized over images with all three candidates active; '
                     'fallback duplicates are counted separately'))
            rows.append(dict(mode=mode, selection=selection, epoch=epoch, **result))
            save_summary(args.output, rows)
            print(json.dumps(rows[-1]), flush=True)
            if index == 6:
                print('PRIMARY_SIX_COMPLETED', flush=True)
        write_json(args.output / 'status.json', dict(state='completed', completed=len(rows)))
        print('ALL_COMPLETED', args.output, flush=True)
    except BaseException as error:
        write_json(args.output / 'status.json', dict(state='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()
