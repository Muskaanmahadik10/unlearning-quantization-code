import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import m2_pilot_contract_v1 as c


class Tests(unittest.TestCase):
    def test_actual_overlay_all_7270_schema_and_pinned_hash(self):
        f=Path(__file__).parent/'regression_fixtures/overlay_file_manifest.json'
        binding=c.read(c.PKG/'RUNTIME_DATA_BINDING.json')
        self.assertEqual(c.sha(f),binding['overlay_manifest_sha256'])
        rows=c.read(f)['files'];self.assertEqual(len(rows),7270)
        c.manifest_schema(rows,hash_only=True)
        with self.assertRaises(ValueError):c.manifest_schema(rows)

    def test_hash_only_checks_real_file_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'x';f.write_bytes(b'abc')
            r={'path':'x','sha256':c.sha(f)}
            c.verify(p,[r],hash_only=True)
            f.write_bytes(b'abd')
            with self.assertRaises(ValueError):c.verify(p,[r],hash_only=True)

    def test_sized_manifests_still_require_exact_size(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'x';f.write_bytes(b'abc')
            r={'path':'x','sha256':c.sha(f),'bytes':3};c.verify(p,[r])
            for n in (None,True,-1,2,'3'):
                bad=dict(r,bytes=n)
                with self.assertRaises(ValueError):c.verify(p,[bad])
            del r['bytes']
            with self.assertRaises(ValueError):c.verify(p,[r])

    def test_hash_only_never_accepts_missing_hash_or_other_schema(self):
        r={'path':'x','sha256':'a'*64}
        for bad in ({'path':'x'},dict(r,sha256=''),dict(r,sha256='g'*64),dict(r,bytes=3)):
            with self.assertRaises(ValueError):c.manifest_schema([bad],hash_only=True)
        with self.assertRaises(ValueError):c.manifest_schema([r,r],hash_only=True)

    def test_hash_only_path_escape_and_missing_still_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for p in ('../x','/x','a\\x','missing'):
                with self.assertRaises(ValueError):c.verify(Path(d),[{'path':p,'sha256':'a'*64}],hash_only=True)

    def test_all_other_real_manifest_callers_keep_sized_contract(self):
        for name,key in [('BASE_INPUT_IDENTITY_LOCK.json','files'),('RUNTIME_DATA_BINDING.json','data_files'),('SELF_LOCK.json','files')]:
            c.manifest_schema(c.read(c.PKG/name)[key])
        text=(c.PKG/'m2_pilot_contract_v1.py').read_text()
        self.assertEqual(text.count("verify(setup,read(setup/'overlay_file_manifest.json')['files'],hash_only=True)"),1)
        self.assertIn("sha(setup/'overlay_file_manifest.json')==binding['overlay_manifest_sha256']",text)


if __name__=='__main__':unittest.main()
