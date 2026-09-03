"""Auditable runtime primitives. Formal launcher is intentionally not present."""
from __future__ import annotations
import json,os,signal,threading
from pathlib import Path
from v4_guard import Reject,need,safe_tree

def publish_new(path:Path,payload:dict):
    """Atomic no-replace publication using a same-directory temp inode and link."""
    need(not path.exists() and not path.is_symlink(),'publication exists')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.parent/(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        b=(json.dumps(payload,sort_keys=True,separators=(',',':'))+'\n').encode(); os.write(fd,b); os.fsync(fd)
    finally: os.close(fd)
    try: os.link(tmp,path,follow_symlinks=False)
    except FileExistsError as e: raise Reject('publication race') from e
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
    dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY); os.fsync(dfd); os.close(dfd)

class StableParent:
    """Retained directory descriptor detects rename/replacement during one shot."""
    def __init__(self,path:Path):
        self.path=path; self.fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); s=os.fstat(self.fd); self.key=(s.st_dev,s.st_ino)
    def check(self):
        try: s=os.stat(self.path,follow_symlinks=False)
        except OSError as e: raise Reject('campaign parent renamed/replaced') from e
        need((s.st_dev,s.st_ino)==self.key,'campaign parent renamed/replaced')
    def close(self): os.close(self.fd)

class Finalizer:
    """Installed before post-lock setup; atomic register/spawn contract and fail aggregation."""
    def __init__(self,root:Path,fault_ids):
        self.root=root; self.ids=tuple(fault_ids); self.mu=threading.RLock(); self.procs={}; self.errors=[]; self.done=False
    def spawn_registered(self,popen_factory,*a,**kw):
        with self.mu:
            need(not self.done,'finalized'); p=popen_factory(*a,**kw); self.procs[p.pid]=p; return p
    def reap(self,p):
        with self.mu:
            need(p.pid in self.procs,'unregistered'); need(p.poll() is not None,'process alive'); del self.procs[p.pid]
    def _kill_confirm(self,p):
        try:
            if p.poll() is None: os.killpg(p.pid,signal.SIGKILL)
            p.wait(timeout=10); need(p.poll() is not None,'kill unconfirmed')
        except BaseException as e: self.errors.append(f'{p.pid}:{type(e).__name__}:{e}')
    def finalize(self,reason,hashes_ok):
        with self.mu:
            if self.done: return not self.errors and hashes_ok
            self.done=True
            for p in tuple(self.procs.values()): self._kill_confirm(p)
            for fid in self.ids:
                p=self.root/'terminals'/f'{fid}.json'
                if not p.exists():
                    try: publish_new(p,{'fault_id':fid,'status':'failure','reason':reason,'kill_errors':list(self.errors)})
                    except BaseException as e: self.errors.append(f'publish:{fid}:{type(e).__name__}:{e}')
            return not self.errors and hashes_ok

def launcher_exit_code(worker_codes,verify_ok,post_rehash_ok,kill_errors):
    need(type(worker_codes) is list and all(type(x) is int for x in worker_codes),'worker codes')
    return 0 if len(worker_codes)==24 and all(x==0 for x in worker_codes) and verify_ok is True and post_rehash_ok is True and kill_errors==[] else 1
