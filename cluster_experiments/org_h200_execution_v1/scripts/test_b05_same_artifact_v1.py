import tempfile
import json
from pathlib import Path
import unittest
from b05_same_artifact_v1 import verify,sha

class Tests(unittest.TestCase):
    def test_hash_and_universe(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'artifact').mkdir(); w=p/'artifact/weight'
            w.write_bytes(b'test')
            (p/'artifact_hashes.json').write_text(json.dumps({'weight':sha(w)}))
            self.assertEqual(len(verify(p)),1)
            w.write_bytes(b'changed')
            with self.assertRaises(RuntimeError): verify(p)
    def test_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'artifact').mkdir()
            (p/'artifact_hashes.json').write_text('{"missing":"abc"}')
            with self.assertRaises(RuntimeError): verify(p)

if __name__=='__main__': unittest.main()
