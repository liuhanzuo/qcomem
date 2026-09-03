from __future__ import annotations
import hashlib,json,os,stat
from pathlib import Path

def lexical_tree(root: Path, excluded: set[str] | None = None) -> dict[str,dict[str,object]]:
    root=Path(root);excluded=excluded or set();result:dict[str,dict[str,object]]={}
    root_stat=os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):raise RuntimeError("tree root is not an exact regular directory")
    def walk(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries,key=lambda item:item.name):
                path=directory/entry.name;relative=path.relative_to(root).as_posix()
                if relative in excluded:continue
                metadata=entry.stat(follow_symlinks=False);mode=metadata.st_mode
                if stat.S_ISLNK(mode):raise RuntimeError("capture/terminal tree symlink forbidden")
                if stat.S_ISDIR(mode):result[relative]={"kind":"directory"};walk(path)
                elif stat.S_ISREG(mode):
                    if metadata.st_nlink != 1:raise RuntimeError("capture/terminal tree hardlink forbidden")
                    data=path.read_bytes();result[relative]={"kind":"regular","bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
                else:raise RuntimeError("capture/terminal tree special node forbidden")
    walk(root);return result

def write_terminal_ledger(root: Path, output: Path) -> None:
    lexical_root=Path(root).absolute();lexical_output=Path(output).absolute()
    try:relative_path=lexical_output.relative_to(lexical_root)
    except ValueError as error:raise RuntimeError("terminal ledger must be inside tree") from error
    root=lexical_root.resolve();output=root/relative_path;relative=relative_path.as_posix()
    if os.path.lexists(output):raise FileExistsError("terminal ledger overwrite or special node")
    nodes=lexical_tree(root,{relative})
    payload={"schema_version":"forkaudit-r40-v15-terminal-tree-v1","root":".","excluded_output":relative,"nodes":nodes}
    with output.open("x",encoding="utf-8") as stream:stream.write(json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n")
    # Re-scan after writing: only the declared ledger itself may be added.
    after=lexical_tree(root)
    expected=dict(nodes);data=output.read_bytes();expected[relative]={"kind":"regular","bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    if after != expected:raise RuntimeError("terminal lexical tree changed during closure")

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    write_terminal_ledger(args.root,args.output)
