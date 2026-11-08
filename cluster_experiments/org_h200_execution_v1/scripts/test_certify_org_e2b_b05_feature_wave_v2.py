import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import certify_org_e2b_b05_feature_wave_v2 as m


class ValidationTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(layer=f'l{i}', n_parameters=100, forget_qdp_signed=-1., retain_qdp_signed=1.,
                     protection_forget_benefit=1., protection_retain_cost=-1.,
                     displacement_sha256='a'*64, retain_independent_abs_delta=0.) for i in range(224)]
        item = dict(schema_version='org_e2b_b05_feature_cell_v2', status='PASS', mode='production',
                    production_eligible=True, outcome_bearing=False, protected_outcomes_inspected=False,
                    b05_behavior_authorized=False, atlas_production_authorized=False,
                    sample_counts=dict(forget=128, retain=128, calibration=128),
                    cell_id='ordinary_tofu_ga_seed42', quantizer='nf4', answer_tokens={'forget':20,'retain':30},
                    current_runtime_reference={}, prerequisites={}, frozen_v1_computation={}, v2_wrapper={},
                    target_count=224, per_layer=rows, per_layer_sha256=m.canonical_sha256(rows),
                    target_parameters=22400, conditions=m.select_sets(rows,22400))
        ref = dict(pre_outcome=True, protected_outcomes_inspected=False, cell_id=item['cell_id'],
                   quantizer='nf4', answer_tokens=item['answer_tokens'],
                   per_layer=[dict(layer=r['layer'],gqdp_signed=-1.,retain_qdp_signed=1.,displacement_sha256='a'*64) for r in rows])
        manifest={'references':[dict(cell_id=item['cell_id'],quantizer='nf4',sha256='hash')]}
        return item,ref,manifest

    def check(self, item, ref, manifest):
        with patch.object(m,'read',side_effect=[item,ref]), patch.object(m,'linked',return_value=Path('ref')), patch.object(m,'sha256_file',return_value='hash'):
            return m.validate(Path('receipt'),manifest)

    def test_valid(self):
        self.check(*self.fixture())

    def test_drift_and_corruption_rejected(self):
        for change in ('retain','nan','displacement','selector','digest','tokens'):
            with self.subTest(change=change):
                item,ref,manifest=self.fixture()
                if change=='retain': ref['per_layer'][0]['retain_qdp_signed']=2.
                if change=='nan': ref['per_layer'][0]['gqdp_signed']=float('nan')
                if change=='displacement': ref['per_layer'][0]['displacement_sha256']='b'*64
                if change=='selector': item['conditions']['pair']['protected_layers']=[]
                if change=='digest': item['per_layer_sha256']='bad'
                if change=='tokens': ref['answer_tokens']=dict(forget=21,retain=30)
                with self.assertRaises(ValueError): self.check(item,ref,manifest)


if __name__=='__main__': unittest.main()
