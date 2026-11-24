import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE = Path(__file__).resolve().parents[1] / 'tools/evaluate_scale_experiment.py'
SPEC = importlib.util.spec_from_file_location('scale_evaluation', MODULE)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_primary_jobs_precede_all_remaining_checkpoints(self):
        best = dict(fixed20=13, uniform=20, adaptive=13)
        primary = evaluation.make_jobs(best, False)
        jobs = evaluation.make_jobs(best, True)
        self.assertEqual(len(primary), 6)
        self.assertEqual(jobs[:6], primary)
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len(set((mode, label) for mode, label, epoch in jobs)), 30)
        self.assertEqual(primary[1], ('uniform', 'best_val', 20))
        for mode in evaluation.MODES:
            self.assertEqual(sorted(epoch for current, label, epoch in jobs
                                    if current == mode and label != 'best_val'),
                             list(evaluation.EPOCHS))

    def test_gate_variation_excludes_forced_fallback_weights(self):
        records = [dict(weights=[0.2, 0.3, 0.5], active_candidates=[True] * 3),
                   dict(weights=[0.4, 0.3, 0.3], active_candidates=[True] * 3),
                   dict(weights=[0.0, 1.0, 0.0], active_candidates=[False, True, False])]
        result = evaluation.gate_statistics(records)
        self.assertEqual(result['all_three_active'], 2)
        self.assertEqual(result['deduplicated_images'], 1)
        self.assertAlmostEqual(result['mean'][0], 0.3)
        self.assertAlmostEqual(result['std'][0], 0.1)
        self.assertEqual(evaluation.gate_statistics(records[-1:])['all_three_active'], 0)

    def test_preflight_checks_best_selection_smoke_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            (run / 'code').mkdir(parents=True)
            source = run / 'code/source.py'
            source.write_text('value = 1\n')
            (run / 'status.txt').write_text('COMPLETED\n')
            scorer = root / 'scorer.py'
            scorer.write_text('pass\n')
            prepared = root / 'prepared'
            prepared.mkdir()
            (prepared / 'train.json').write_text('[]')
            for mode in evaluation.MODES:
                current = run / mode
                (current / 'checkpoints').mkdir(parents=True)
                config = dict(mode=mode, output=str(current), prepared=str(prepared),
                              smoke=False, epochs=44, size=352, seed=42,
                              train_count=1260, val_count=140,
                              hashes={'source.py': evaluation.sha256(source),
                                      'prepared/train.json': evaluation.sha256(prepared / 'train.json')})
                evaluation.write_json(current / 'config.json', config)
                evaluation.write_json(current / 'status.json', dict(state='completed'))
                rows = [dict(epoch=epoch, val_BCE=abs(epoch - 13) + 0.01)
                        for epoch in range(1, 45)]
                (current / 'epochs.jsonl').write_text('\n'.join(map(json.dumps, rows)))
                evaluation.write_json(current / 'best_val.json', rows[12])
                for filename in ['best_val.pth'] + ['GeleNet.pth.' + str(epoch) for epoch in evaluation.EPOCHS]:
                    (current / 'checkpoints' / filename).write_bytes(b'fixture')
            images, gt, regions = (root / name for name in ('images', 'gt', 'regions'))
            for folder in (images, gt, regions):
                folder.mkdir()
            for index in range(600):
                for folder, suffix in ((images, '.jpg'), (gt, '.png'), (regions, '.mat')):
                    (folder / (str(index) + suffix)).touch()
            args = SimpleNamespace(output=root / 'output', run=run, scorer=scorer,
                                   reference_scorer=scorer,
                                   all_checkpoints=True, images=images, gt=gt, regions=regions)
            configs, jobs, image_ids = evaluation.preflight(args)
            self.assertEqual(len(jobs), 30)
            self.assertEqual(len(image_ids), 600)
            adaptive_config = run / 'adaptive/config.json'
            changed = configs['adaptive'].copy()
            changed['smoke'] = True
            evaluation.write_json(adaptive_config, changed)
            with self.assertRaisesRegex(ValueError, 'Unexpected training'):
                evaluation.preflight(args)
            evaluation.write_json(adaptive_config, configs['adaptive'])
            evaluation.write_json(run / 'adaptive/best_val.json', dict(epoch=44, val_BCE=31.01))
            with self.assertRaisesRegex(ValueError, 'best_val does not match'):
                evaluation.preflight(args)
            evaluation.write_json(run / 'adaptive/best_val.json', dict(epoch=13, val_BCE=0.01))
            source.write_text('value = 2\n')
            with self.assertRaisesRegex(ValueError, 'source/input changed'):
                evaluation.preflight(args)

    def test_stability_excludes_best_val_and_requires_full_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [dict(mode='fixed20', selection='epoch_' + str(epoch), epoch=epoch,
                         **{metric: float(epoch) for metric in evaluation.METRICS})
                    for epoch in (24, 29, 34, 39, 44)]
            rows.append(dict(mode='fixed20', selection='best_val', epoch=13,
                             **{metric: 1000.0 for metric in evaluation.METRICS}))
            evaluation.save_summary(output, rows[:-2])
            self.assertEqual(evaluation.read_json(output / 'late_window_stability.json')['results'], {})
            evaluation.save_summary(output, rows)
            result = evaluation.read_json(output / 'late_window_stability.json')['results']
            self.assertEqual(result['fixed20']['S']['mean'], 34.0)
            self.assertNotIn('adaptive', result)


if __name__ == '__main__':
    unittest.main()
