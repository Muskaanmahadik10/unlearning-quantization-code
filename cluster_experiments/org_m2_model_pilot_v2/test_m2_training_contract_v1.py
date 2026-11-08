import copy
import unittest
from types import SimpleNamespace
import m2_training_core_v1 as c


def config(n, epochs, seed=20260908):
    import math
    return {'epoch_anchor_records': n, 'epochs': epochs, 'seed': seed, 'microbatch': 8,
            'gradient_accumulation': 4, 'intended_optimizer_steps': math.ceil(n/32)*epochs}


class Tests(unittest.TestCase):
    def test_exact_role_schedule_coverage(self):
        for role, n, epochs, steps in [('O',4000,5,625),('R',3600,5,565),('U',400,10,130)]:
            p = c.schedule(role, config(n, epochs))
            self.assertEqual(p['total_updates'], steps)
            for epoch in range(epochs):
                ids = [i for u in p['updates'] if u['epoch'] == epoch for m in u['microbatches'] for i in m['anchor']]
                self.assertEqual(sorted(ids), list(range(n)))
            if role == 'U':
                draws = [i for u in p['updates'] for m in u['microbatches'] for i in m['retain']]
                self.assertEqual(len(draws), 4000)
                self.assertTrue(all(0 <= i < 3600 for i in draws))

    def test_partial_accumulation_not_dropped(self):
        p = c.schedule('R', config(3600, 5))
        self.assertEqual([len(p['updates'][i]['microbatches']) for i in (112,225,338,451,564)], [2]*5)

    def test_schedule_reproducibility_and_seed(self):
        a = c.schedule('U', config(400,10))
        self.assertEqual(a, c.schedule('U',config(400,10)))
        self.assertNotEqual(a['sha256'], c.schedule('U',config(400,10,99))['sha256'])

    def test_refuse_silent_step_and_batch_changes(self):
        for change in ({'intended_optimizer_steps': 624}, {'epoch_anchor_records': 3999}):
            cfg = config(4000,5); cfg.update(change)
            with self.assertRaises(ValueError): c.schedule('O',cfg)

    def test_lr_boundaries(self):
        self.assertEqual(c.lr_multiplier(0,625,125),0)
        self.assertEqual(c.lr_multiplier(125,625,125),1)
        self.assertAlmostEqual(c.lr_multiplier(624,625,125),1/500)
        with self.assertRaises(ValueError): c.lr_multiplier(625,625,125)

    def test_prompt_prefix_answer_mask(self):
        t = SimpleNamespace(eos_token_id=3, apply_chat_template=lambda chat,**kw: [1,2] if len(chat)==2 else [1,2,4,3])
        out = c.encode_qa(t,'q','a')
        self.assertEqual(out['labels'],[-100,-100,4,3])
        self.assertEqual(out['prompt_ids'],[1,2])

    def test_no_truncation_and_no_prefix_guess(self):
        t = SimpleNamespace(eos_token_id=3, apply_chat_template=lambda chat,**kw: [1,2] if len(chat)==2 else [1,7,4,3])
        with self.assertRaises(ValueError): c.encode_qa(t,'q','a')
        t.apply_chat_template=lambda chat,**kw: [1,2] if len(chat)==2 else [1,2,4,3]
        with self.assertRaises(ValueError): c.encode_qa(t,'q','a',cap=3)

    def test_no_training_submission_entrypoint(self):
        import ast
        from pathlib import Path
        tree=ast.parse(Path(c.__file__).read_text())
        imports={alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names}
        self.assertNotIn('subprocess',imports)


if __name__=='__main__': unittest.main()
