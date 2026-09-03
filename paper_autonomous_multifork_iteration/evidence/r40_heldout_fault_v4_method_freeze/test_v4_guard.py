import io,json,os,tarfile,tempfile,unittest
from pathlib import Path
from v4_guard import *
from v4_runtime import *

class T(unittest.TestCase):
 def bad(self,fn,*a): self.assertRaises(Reject,fn,*a)
 def fault(self):
  return {'schema_version':'forkaudit-v4-fault-set-v1','designer_attestation':{'snapshot_sha256':'a'*64,'isolated':True,'no_prior_faults_seen':True},'faults':[{'fault_id':f'V4F{i:02d}','mechanism_family':f'm{i}','activation_call_index':i-1,'token_id':i,'fixed_payload':{},'implementation_mutation':'x'} for i in range(1,9)]}
 def test_primitives_and_faults(self):
  validate_fault_set(self.fault())
  for key,val in [('activation_call_index',True),('token_id',248320)]:
   x=self.fault(); x['faults'][0][key]=val; self.bad(validate_fault_set,x)
  x=self.fault(); x['faults'][1]['mechanism_family']='m1'; self.bad(validate_fault_set,x)
  x=self.fault(); x['faults'][0]['fault_id']='V4F08'; self.bad(validate_fault_set,x)
 def test_gpu_exact(self):
  rows=[{'index':i,'name':EXACT_SKU,'uuid':f'GPU-{i}','memory_used_mib':0} for i in range(8)]; u=[r['uuid'] for r in rows]
  parse_gpus(rows,[],u)
  for mutate in ('sku','process','torch','memory'):
   q=json.loads(json.dumps(rows)); p=[]; z=list(u)
   if mutate=='sku': q[0]['name']='NVIDIA H200'
   if mutate=='process': p=['GPU-0,123']
   if mutate=='torch': z[0]='GPU-X'
   if mutate=='memory': q[0]['memory_used_mib']=1
   self.bad(parse_gpus,q,p,z)
 def test_allocator(self):
  x={'H0':2,'H1':2,'H4':3,'H6':4,'H7':2,'sync_event_ids':['a','b'],'run_id':'r','lane':'clean','device_uuid':'GPU-0'}; validate_allocator(x)
  for k,v in [('H0',1),('H4',1),('H7',3)]: y=dict(x); y[k]=v; self.bad(validate_allocator,y)
  y=dict(x); y['sync_event_ids']=['a','a']; self.bad(validate_allocator,y)
 def test_tree_hardlink_extra_and_runner(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d).resolve(); p=r/'run'; p.write_text('x'); m=[{'path':'run','sha256':file_sha(p),'size':1}]
   validate_runner(r,m,[str(p)])
   (r/'extra').write_text('x'); self.bad(verify_manifest,r,m); (r/'extra').unlink()
   os.link(p,r/'hard'); self.bad(safe_tree,r)
 def test_archive_duplicates_metadata_and_external_hash(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.tgz'; ledger=b'{}'
   with tarfile.open(p,'w:gz',format=tarfile.PAX_FORMAT) as t:
    i=tarfile.TarInfo('source-ledger.json'); i.size=len(ledger); i.uid=i.gid=i.mtime=0; i.uname=i.gname=''; t.addfile(i,io.BytesIO(ledger))
   verify_archive(p,file_sha(p),hashlib.sha256(ledger).hexdigest())
   self.bad(verify_archive,p,'0'*64,hashlib.sha256(ledger).hexdigest())
   for kind in ('duplicate','metadata'):
    q=Path(d)/(kind+'.tgz')
    with tarfile.open(q,'w:gz',format=tarfile.PAX_FORMAT) as t:
     count=2 if kind=='duplicate' else 1
     for _ in range(count):
      i=tarfile.TarInfo('source-ledger.json'); i.size=len(ledger); i.uid=1 if kind=='metadata' else 0; i.gid=i.mtime=0;i.uname=i.gname='';t.addfile(i,io.BytesIO(ledger))
    self.bad(verify_archive,q,file_sha(q),hashlib.sha256(ledger).hexdigest())
 def test_no_replace_and_parent_rename(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d).resolve(); p=r/'x'; publish_new(p,{'a':1}); self.bad(publish_new,p,{'a':2})
   s=StableParent(r); moved=r.parent/(r.name+'-moved'); r.rename(moved)
   try: self.bad(s.check)
   finally: s.close(); moved.rename(r)
 def test_launcher_failure_propagation(self):
  self.assertEqual(launcher_exit_code([0]*24,True,True,[]),0)
  self.assertEqual(launcher_exit_code([0]*23+[1],True,True,[]),1)
  self.assertEqual(launcher_exit_code([0]*24,False,True,[]),1)
  self.assertEqual(launcher_exit_code([0]*24,True,False,[]),1)
  self.assertEqual(launcher_exit_code([0]*24,True,True,['kill']),1)
 def test_exact_terminal_tree(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d).resolve(); (r/'terminals').mkdir(); h={'method':'a'*64}
   for fid in FAULT_IDS: (r/'terminals'/f'{fid}.json').write_text(json.dumps({'fault_id':fid,'status':'success','pre_hashes':h,'post_hashes':h}))
   (r/'binding.json').write_text('{}'); (r/'execution-terminal.json').write_text('{}'); validate_terminals(r,h,h)
   (r/'extra').write_text('x'); self.bad(validate_terminals,r,h,h)

if __name__=='__main__': unittest.main()
