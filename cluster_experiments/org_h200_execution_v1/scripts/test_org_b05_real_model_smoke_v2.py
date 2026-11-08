import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import diagnose_org_b05_rtn_reload_v1 as diag


class LoaderTests(unittest.TestCase):
    def check_loader(self, failure=False):
        torch, tf, packed = MagicMock(), MagicMock(), MagicMock()
        torch.get_default_dtype.return_value = 'original'
        torch.bfloat16 = 'bf16'
        packed.RTN_INTERVENTION = 'rtn'
        metadata = {'org_quantization': {'intervention': 'rtn', 'packed_fqns': ['layer']}}
        safe = MagicMock()
        safe.load_model.return_value = ([], [])
        if failure:
            tf.LlamaForCausalLM.side_effect = RuntimeError('constructor failure')
        with patch.dict(sys.modules, {'safetensors.torch': safe}), patch.object(diag, 'read', return_value=metadata):
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'constructor failure'):
                    diag.corrected_load(torch, tf, packed, Path('artifact'), ['layer'])
            else:
                diag.corrected_load(torch, tf, packed, Path('artifact'), ['layer'])
                tf.LlamaForCausalLM.return_value.to.assert_called_once_with('cuda:0')
                self.assertTrue(safe.load_model.call_args.kwargs['strict'])
        self.assertEqual(torch.set_default_dtype.call_args_list[-1].args, ('original',))

    def test_no_blanket_cast(self):
        self.check_loader()

    def test_dtype_restored_after_exception(self):
        self.check_loader(True)


if __name__ == '__main__':
    unittest.main()
