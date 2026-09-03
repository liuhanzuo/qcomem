from __future__ import annotations
import argparse,gzip,io,tarfile
from pathlib import Path
FIXED_MTIME=1787760000
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if a.output.exists():raise FileExistsError("archive overwrite")
    files=sorted(path for path in root.rglob("*") if path.is_file() and "packages" not in path.relative_to(root).parts and "__pycache__" not in path.parts and path.suffix!=".pyc")
    with a.output.open("xb") as raw:
      with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
       with tarfile.open(fileobj=gz,mode="w",format=tarfile.USTAR_FORMAT) as tar:
        for path in files:
            data=path.read_bytes();info=tarfile.TarInfo(f"{root.name}/{path.relative_to(root).as_posix()}");info.size=len(data);info.mtime=FIXED_MTIME;info.uid=info.gid=0;info.uname=info.gname="";info.mode=0o755 if path.name=="launch_h20.sh" else 0o644;info.pax_headers={};tar.addfile(info,io.BytesIO(data))
    return 0
if __name__=="__main__":raise SystemExit(main())
