from pathlib import Path
import gzip,hashlib,json,tarfile,io
ROOT=Path(__file__).resolve().parent
FILES=('README.md','CONTRACT.md','EXTERNAL_BINDING.template.json','v4_guard.py','v4_runtime.py','v4_cuda_capture.py','test_v4_guard.py','freeze.py','designer_snapshot/README.md','designer_snapshot/PUBLIC_CONTRACT.md','designer_snapshot/FAULT_SET_SCHEMA.json')
def sha(b): return hashlib.sha256(b).hexdigest()
rows=[{'path':p,'sha256':sha((ROOT/p).read_bytes()),'size':(ROOT/p).stat().st_size} for p in FILES]
ledger=(json.dumps({'schema_version':'forkaudit-v4-source-ledger-v1','freeze_timestamp':'2026-08-27T00:00:00Z','files':rows},sort_keys=True,separators=(',',':'))+'\n').encode()
(ROOT/'source-ledger.json').write_bytes(ledger)
members=[('source-ledger.json',ledger)]+[(p,(ROOT/p).read_bytes()) for p in FILES]
raw=io.BytesIO()
with tarfile.open(fileobj=raw,mode='w',format=tarfile.USTAR_FORMAT) as t:
 for name,data in sorted(members):
  i=tarfile.TarInfo(name); i.size=len(data); i.uid=i.gid=i.mtime=0;i.uname=i.gname='';i.mode=0o444;t.addfile(i,io.BytesIO(data))
out=ROOT/'r40-method-v4-freeze-20260827b.tar.gz'
with out.open('wb') as f:
 with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g:g.write(raw.getvalue())
result={'schema_version':'forkaudit-v4-method-freeze-v1','status':'HOLD_PENDING_FRESH_AUDIT','freeze_timestamp':'2026-08-27T00:00:00Z','tests':8,'archive_sha256':sha(out.read_bytes()),'source_ledger_sha256':sha(ledger),'v3_archive_sha256':'b556351218e71c1280350ef159d20cfbc78b796c512e3ff4104259454ea4113d','fault_set':None,'formal_config':None,'gpu_execution':False}
(ROOT/'METHOD_FROZEN.json').write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps(result,indent=2,sort_keys=True))
