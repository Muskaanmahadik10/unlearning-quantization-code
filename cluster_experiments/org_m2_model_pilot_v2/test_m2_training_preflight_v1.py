import ast
import json
from pathlib import Path
import tempfile
import unittest
import m2_training_cpu_preflight_v1 as p


class Tests(unittest.TestCase):
    def test_no_duplicate_json(self):
        with self.assertRaises(ValueError): p.pairs([('a',1),('a',2)])

    def test_manifest_missing_duplicate_and_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'x').write_bytes(b'test')
            rows=[{'path':'x','sha256':p.sha(root/'x'),'bytes':4}]
            p.verify(root,rows)
            for bad in (rows+rows,[dict(rows[0],sha256='0'*64)],[dict(rows[0],bytes=2)]):
                with self.assertRaises(ValueError): p.verify(root,bad)

    def test_path_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ('../x','/x','a\\b','missing'):
                with self.assertRaises(ValueError): p.confined(Path(td),name)

    def test_receipt_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'x.json'; p.save(path,{'ok':True})
            self.assertEqual(p.read(path),{'ok':True})
            with self.assertRaises(FileExistsError): p.save(path,{'ok':False})

    def test_variants_and_exact_qa_identity(self):
        row={'question':'q','answer':'a','paraphrased_answer':'b','perturbed_answer':['c','d'],'paraphrased_question':'qq'}
        self.assertEqual(len(list(p.variants(row))),5)
        self.assertNotEqual(p.qa_identity(row),p.qa_identity(dict(row,answer='z')))
        with self.assertRaises(ValueError): list(p.variants(dict(row,perturbed_answer='no')))

    def test_offline_cpu_only_entrypoint(self):
        source=Path(p.__file__).read_text(encoding='utf-8'); tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        pretrained=[n.func.value.id for n in calls if isinstance(n.func,ast.Attribute) and
                    n.func.attr=='from_pretrained' and isinstance(n.func.value,ast.Name)]
        self.assertEqual(pretrained,['AutoTokenizer'])
        self.assertNotIn('subprocess',source)
        self.assertIn("local_files_only=True",source)
        self.assertIn("'training_authorized':False",source)


if __name__=='__main__': unittest.main()
