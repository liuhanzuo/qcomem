import hashlib,io,json,os,sys,tarfile,tempfile,types,unittest
from pathlib import Path
from v5_guard import *
from v5_runtime import *
class T(unittest.TestCase):
 def bad(self,f,*a):self.assertRaises(Reject,f,*a)
 def test_binding_not_self_authorized(self):
  v={'schema_version':'forkaudit-v5-operator-binding-v1','approved_archive_sha256':'a'*64,'approved_source_ledger_sha256':'b'*64,'approved_designer_snapshot_sha256':'c'*64,'operator_id':'independent','operator_signature':'sig','published_uri':'https://operator.invalid/x'}
  operator_binding(v,('a'*64,'b'*64,'c'*64),lambda _:True);self.bad(operator_binding,v,('0'*64,'b'*64,'c'*64),lambda _:True)
  x=dict(v);x['approved_archive_sha256']=None;self.bad(operator_binding,x,('a'*64,'b'*64,'c'*64),lambda _:True)
 def test_attestation_exact_true(self):
  v={'snapshot_sha256':'a'*64,'snapshot_inventory_sha256':'b'*64,'inputs_limited_to_snapshot':True,'no_prior_faults_seen':True,'no_private_source_seen':True};designer_attestation(v,'a'*64)
  for k in ('inputs_limited_to_snapshot','no_prior_faults_seen','no_private_source_seen'):x=dict(v);x[k]=1;self.bad(designer_attestation,x,'a'*64)
 def test_raw_empty(self):
  raw_process_query(b'');
  for x in (b'No running processes found',b'\n',b'GPU-0, 1\n'):self.bad(raw_process_query,x)
 def test_success_and_signals(self):
  self.assertTrue(success('success',True,True,True,True,[]))
  for x in [('failure',True,True,True,True,[]),('success',1,True,True,True,[]),('success',True,False,True,True,[])]:self.assertFalse(success(*x))
  self.assertEqual(signal_exit(2),130);self.assertEqual(signal_exit(15),143)
 def test_protected_publish_rename_restore(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);p=base/'p';p.mkdir();g=ProtectedParent(p);m=base/'m';p.rename(m);p.mkdir();self.bad(g.publish,'x',{})
   p.rmdir();m.rename(p);g.publish('x',{'a':1});self.bad(g.publish,'x',{});g.close()
 def test_lifecycle_signal_eight_failures(self):
  with tempfile.TemporaryDirectory() as d:
   p=ProtectedParent(Path(d));l=Lifecycle(p,TERM_IDS);l.start();self.assertEqual(l.finalize('signal',15),143);self.assertEqual(len(list(Path(d).glob('*.failure.json'))),8);p.close()
 def test_terminal_schema(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);hsh={'archive':'a'*64}
   for fid in TERM_IDS:(r/f'{fid}.json').write_text(json.dumps({'schema_version':'forkaudit-v5-terminal-v1','fault_id':fid,'status':'success','pre_hashes':hsh,'post_hashes':hsh}))
   terminals(r,hsh,hsh);x=json.loads((r/'V5F01.json').read_text());x['status']='failure';(r/'V5F01.json').write_text(json.dumps(x));self.bad(terminals,r,hsh,hsh)
 def test_runner_ambient_and_extra(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d).resolve();p=r/'run';p.write_text('x');m=[{'path':'run','sha256':digest(p),'size':1}];runner(r,m,[str(p)]);self.bad(runner,r,m,[str(p),'/etc/passwd']);(r/'extra').write_text('x');self.bad(runner,r,m,[str(p)])
 def test_import_transplant_rejected(self):
  import v5_guard
  import_provenance(v5_guard,Path(v5_guard.__file__).resolve(),digest(v5_guard.__file__))
  fake=types.SimpleNamespace(__file__='/tmp/v5_guard.py',__name__='v5_guard');self.bad(import_provenance,fake,Path(v5_guard.__file__).resolve(),digest(v5_guard.__file__))
 def test_gzip_ledger_exact(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.tgz';blob=b'x';row={'path':'x','sha256':hashlib.sha256(blob).hexdigest(),'size':1};led=(json.dumps({'schema_version':'forkaudit-v5-source-ledger-v1','freeze_timestamp':'2026-08-27T00:00:00Z','files':[row]},sort_keys=True,separators=(',',':'))+'\n').encode();raw=io.BytesIO()
   with tarfile.open(fileobj=raw,mode='w',format=tarfile.USTAR_FORMAT) as t:
    for n,z in [('source-ledger.json',led),('x',blob)]:i=tarfile.TarInfo(n);i.size=len(z);i.uid=i.gid=i.mtime=0;i.uname=i.gname='';t.addfile(i,io.BytesIO(z))
   import gzip
   with p.open('wb') as f:
    with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as g:g.write(raw.getvalue())
   gzip_archive(p,digest(p),hashlib.sha256(led).hexdigest());self.bad(gzip_archive,p,'0'*64,hashlib.sha256(led).hexdigest())
if __name__=='__main__':unittest.main()
