import json,os,signal,threading
from pathlib import Path
from v5_guard import Reject,need,signal_exit,success
class ProtectedParent:
 def __init__(self,path):
  self.path=Path(path).resolve();self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);st=os.fstat(self.fd);self.key=(st.st_dev,st.st_ino)
 def check(self):
  need((os.fstat(self.fd).st_dev,os.fstat(self.fd).st_ino)==self.key,'dirfd changed')
  try:st=os.stat(self.path,follow_symlinks=False)
  except OSError as e:raise Reject('parent absent') from e
  need((st.st_dev,st.st_ino)==self.key,'parent replaced')
 def publish(self,name,obj):
  self.check();need('/' not in name and name not in ('.','..'),'name');tmp=f'.{name}.{os.getpid()}.{threading.get_ident()}'
  fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.fd)
  try:os.write(fd,(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n').encode());os.fsync(fd)
  finally:os.close(fd)
  try:os.link(tmp,name,src_dir_fd=self.fd,dst_dir_fd=self.fd,follow_symlinks=False)
  except FileExistsError as e:raise Reject('no replace') from e
  finally:
   try:os.unlink(tmp,dir_fd=self.fd)
   except FileNotFoundError:pass
  os.fsync(self.fd);self.check()
 def close(self):os.close(self.fd)

class Lifecycle:
 def __init__(self,parent,ids):self.parent=parent;self.ids=tuple(ids);self.started=False;self.done=False;self.mu=threading.RLock();self.gates={};self.errors=[]
 def start(self):
  with self.mu:need(not self.started,'started');self.started=True
 def finalize(self,reason,signum=None):
  with self.mu:
   if self.done:return 1
   self.done=True;ok=success(reason,self.gates.get('hashes'),self.gates.get('workers'),self.gates.get('verify'),self.gates.get('rehash'),self.errors)
   if self.started and not ok:
    for fid in self.ids:
     try:self.parent.publish(fid+'.failure.json',{'schema_version':'forkaudit-v5-failure-v1','fault_id':fid,'reason':reason,'signal':signum,'errors':list(self.errors)})
     except BaseException as e:self.errors.append(type(e).__name__+':'+str(e))
   if signum is not None:return signal_exit(signum)
   return 0 if ok else 1
