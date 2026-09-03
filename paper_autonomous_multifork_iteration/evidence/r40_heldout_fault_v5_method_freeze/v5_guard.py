from __future__ import annotations
import gzip,hashlib,importlib.util,json,os,stat,struct,sys,tarfile
from pathlib import Path
class Reject(ValueError):pass
def need(x,m):
 if not x: raise Reject(m)
def exact(x,ks,m): need(type(x) is dict and set(x)==set(ks),m)
def s(x,m): need(type(x) is str and bool(x),m);return x
def b(x,m):need(type(x) is bool,m);return x
def i(x,m):need(type(x) is int,m);return x
def h(x,m):need(type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x),m);return x
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def operator_binding(v,external_approved,signature_verify):
 exact(v,('schema_version','approved_archive_sha256','approved_source_ledger_sha256','approved_designer_snapshot_sha256','operator_id','operator_signature','published_uri'),'binding')
 need(v['schema_version']=='forkaudit-v5-operator-binding-v1','schema')
 vals=tuple(h(v[k],k) for k in ('approved_archive_sha256','approved_source_ledger_sha256','approved_designer_snapshot_sha256'))
 need(type(external_approved) is tuple and vals==external_approved,'external approval mismatch')
 s(v['operator_id'],'operator');s(v['operator_signature'],'signature');s(v['published_uri'],'publication')
 need(signature_verify(v) is True,'independent signature')

def designer_attestation(v,expected_snapshot):
 exact(v,('snapshot_sha256','snapshot_inventory_sha256','inputs_limited_to_snapshot','no_prior_faults_seen','no_private_source_seen'),'attestation')
 need(h(v['snapshot_sha256'],'snapshot')==expected_snapshot,'snapshot digest')
 h(v['snapshot_inventory_sha256'],'inventory')
 for k in ('inputs_limited_to_snapshot','no_prior_faults_seen','no_private_source_seen'):need(v[k] is True,k)

def raw_process_query(raw):
 need(type(raw) is bytes,'raw process bytes'); need(raw==b'','compute query must be zero bytes')

def gzip_archive(path,expected_archive,expected_ledger):
 data=Path(path).read_bytes();need(hashlib.sha256(data).hexdigest()==h(expected_archive,'archive'),'external archive')
 need(data[:2]==b'\x1f\x8b' and struct.unpack('<I',data[4:8])[0]==0,'gzip mtime')
 names=set()
 with tarfile.open(path,'r:gz') as t:
  ms=t.getmembers();need(ms,'members')
  for m in ms:
   need(m.name not in names,'duplicate');names.add(m.name);need(m.isfile() and m.uid==m.gid==m.mtime==0 and m.uname==m.gname=='','metadata')
  need('source-ledger.json' in names,'ledger')
  raw=t.extractfile('source-ledger.json').read();need(hashlib.sha256(raw).hexdigest()==h(expected_ledger,'ledger anchor'),'ledger hash')
  led=json.loads(raw);exact(led,('schema_version','freeze_timestamp','files'),'ledger schema');need(led['schema_version']=='forkaudit-v5-source-ledger-v1','ledger version')
  rows=led['files'];need(type(rows) is list and rows,'meaningful ledger');listed=set()
  for r in rows:
   exact(r,('path','sha256','size'),'ledger row');p=s(r['path'],'path');need(p not in listed and p in names,'ledger member');listed.add(p);h(r['sha256'],'sha');i(r['size'],'size');need(r['size']>=0,'size')
   blob=t.extractfile(p).read();need(len(blob)==r['size'] and hashlib.sha256(blob).hexdigest()==r['sha256'],'ledger content')
  need(listed==names-{'source-ledger.json'},'exact ledger inventory')

def import_provenance(module,expected_path,expected_sha):
 p=Path(expected_path).resolve();need(Path(module.__file__).resolve()==p and digest(p)==expected_sha,'module provenance')
 spec=importlib.util.find_spec(module.__name__);need(spec is not None and Path(spec.origin).resolve()==p,'import origin')

def torch_provenance(torch,expected_init,expected_sha,visibility,physical_uuid,index):
 import_provenance(torch,expected_init,expected_sha);need(os.environ.get('CUDA_VISIBLE_DEVICES')==visibility,'visibility')
 need(type(index) is int and type(index) is not bool,'index');props=torch.cuda.get_device_properties(index)
 need(str(getattr(props,'uuid',''))==physical_uuid,'physical UUID')

def runner(root,manifest,argv):
 root=Path(root).resolve();need(type(manifest) is list and type(argv) is list and argv,'runner')
 rows={}
 for r in manifest:exact(r,('path','sha256','size'),'row');p=s(r['path'],'path');need(p not in rows and not p.startswith('/') and '..' not in Path(p).parts,'path');rows[p]=r
 actual={}
 for base,ds,fs in os.walk(root):
  ds.sort();fs.sort()
  for n in ds:need(not (Path(base)/n).is_symlink(),'symlink')
  for n in fs:
   p=Path(base)/n;st=os.lstat(p);need(stat.S_ISREG(st.st_mode) and st.st_nlink==1,'regular unique');actual[p.relative_to(root).as_posix()]={'sha256':digest(p),'size':st.st_size}
 need(set(rows)==set(actual),'exact runner tree')
 for p,r in rows.items():need(actual[p]=={'sha256':r['sha256'],'size':r['size']},'runner hash')
 for arg in argv:
  s(arg,'argv');need('/etc/' not in arg and '..' not in Path(arg).parts,'ambient argv')
  if arg.startswith('/'):
   q=Path(arg).resolve();need(q.is_relative_to(root) and q.relative_to(root).as_posix() in rows,'argv file binding')

TERM_IDS=tuple(f'V5F{x:02d}' for x in range(1,9))
def terminals(root,pre,post):
 need(type(pre) is dict and type(post) is dict and pre==post,'pre post')
 root=Path(root);expected={f'{x}.json' for x in TERM_IDS};need({p.name for p in root.iterdir()}==expected,'exact terminals')
 for fid in TERM_IDS:
  v=json.loads((root/f'{fid}.json').read_text());exact(v,('schema_version','fault_id','status','pre_hashes','post_hashes'),'terminal')
  need(v['schema_version']=='forkaudit-v5-terminal-v1' and v['fault_id']==fid and v['status']=='success' and v['pre_hashes']==pre and v['post_hashes']==post,'terminal success')

def success(reason,hashes_ok,workers_ok,verify_ok,rehash_ok,kill_errors):
 return reason=='success' and hashes_ok is True and workers_ok is True and verify_ok is True and rehash_ok is True and kill_errors==[]
def signal_exit(signum):need(signum in (2,15),'signal');return 128+signum
