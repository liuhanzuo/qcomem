from pathlib import Path
import gzip,hashlib,io,json,tarfile
R=Path(__file__).resolve().parent
FILES=('README.md','operator-binding.template.json','v5_guard.py','v5_runtime.py','test_v5.py','freeze_v5.py','designer_snapshot/README.md','designer_snapshot/PUBLIC_CONTRACT.md','designer_snapshot/ATTESTATION_SCHEMA.json')
def H(b):return hashlib.sha256(b).hexdigest()
rows=[{'path':p,'sha256':H((R/p).read_bytes()),'size':(R/p).stat().st_size} for p in FILES]
ledger=(json.dumps({'schema_version':'forkaudit-v5-source-ledger-v1','freeze_timestamp':'2026-08-27T00:00:00Z','files':rows},sort_keys=True,separators=(',',':'))+'\n').encode();(R/'source-ledger.json').write_bytes(ledger)
raw=io.BytesIO()
with tarfile.open(fileobj=raw,mode='w',format=tarfile.USTAR_FORMAT) as t:
 for n,z in sorted([('source-ledger.json',ledger)]+[(p,(R/p).read_bytes()) for p in FILES]):
  x=tarfile.TarInfo(n);x.size=len(z);x.uid=x.gid=x.mtime=0;x.uname=x.gname='';x.mode=0o444;t.addfile(x,io.BytesIO(z))
p=R/'r40-method-v5-freeze-20260827a.tar.gz'
with p.open('xb') as f:
 with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as g:g.write(raw.getvalue())
snap=[x for x in rows if x['path'].startswith('designer_snapshot/')];snap_hash=H((json.dumps(snap,sort_keys=True,separators=(',',':'))+'\n').encode())
o={'schema_version':'forkaudit-v5-method-freeze-v1','status':'HOLD_PENDING_FRESH_AUDIT_AND_OPERATOR_BINDING','freeze_timestamp':'2026-08-27T00:00:00Z','archive_sha256':H(p.read_bytes()),'source_ledger_sha256':H(ledger),'designer_snapshot_inventory_sha256':snap_hash,'archive_members':len(FILES)+1,'tests':10,'fault_set':None,'formal_config':None,'operator_binding':None,'gpu_execution':False}
(R/'METHOD_FROZEN.json').write_text(json.dumps(o,sort_keys=True,separators=(',',':'))+'\n');print(json.dumps(o,indent=2,sort_keys=True))
