from __future__ import annotations
import json,os,socket,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"executed_source"))
from r40_tree_closure import lexical_tree,write_terminal_ledger

class TreeClosure(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);(root/"regular").write_text("x");(root/"dir").mkdir();return root
    def test_terminal_ledger_lexically_enumerates_full_tree(self):
        root=self.fixture();output=root/"terminal.json";write_terminal_ledger(root,output);payload=json.loads(output.read_text())
        self.assertEqual(set(payload["nodes"]),{"dir","regular"});self.assertEqual(set(lexical_tree(root)),{"dir","regular","terminal.json"})
    def test_special_symlink_hardlink_nodes_fail(self):
        creators=(lambda root:os.mkfifo(root/"bad"),lambda root:(root/"bad").symlink_to(root/"regular"),lambda root:os.link(root/"regular",root/"bad"))
        for create in creators:
            root=self.fixture();create(root)
            with self.assertRaisesRegex(RuntimeError,"special|symlink|hardlink"):lexical_tree(root)
        root=self.fixture();sock=socket.socket(socket.AF_UNIX);self.addCleanup(sock.close);sock.bind(str(root/"bad"))
        with self.assertRaisesRegex(RuntimeError,"special"):lexical_tree(root)
    def test_terminal_output_path_cannot_be_preexisting_special_node(self):
        root=self.fixture();output=root/"terminal.json";output.symlink_to(root/"missing")
        with self.assertRaisesRegex(FileExistsError,"overwrite|special"):write_terminal_ledger(root,output)

if __name__=="__main__":unittest.main()
