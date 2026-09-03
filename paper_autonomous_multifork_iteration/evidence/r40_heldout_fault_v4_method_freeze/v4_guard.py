"""Pure fail-closed guards for the R40 v4 method freeze. No execution side effects."""
from __future__ import annotations
import hashlib, json, os, stat, tarfile
from pathlib import Path

FAULT_IDS=tuple(f"V4F{i:02d}" for i in range(1,9))
LANES=("reference","clean","mutant")
EXACT_SKU="NVIDIA H20-3e"

class Reject(ValueError): pass
def need(x,msg):
    if not x: raise Reject(msg)
def exact(d,keys,msg):
    need(type(d) is dict and set(d)==set(keys),msg)
def text(x,msg): need(type(x) is str and bool(x),msg); return x
def integer(x,msg,lo=None,hi=None):
    need(type(x) is int,msg)
    if lo is not None: need(x>=lo,msg)
    if hi is not None: need(x<=hi,msg)
    return x
def boolean(x,msg): need(type(x) is bool,msg); return x
def sha(x,msg): need(type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x),msg); return x
def file_sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def safe_tree(root: Path):
    """Return exact regular-file inventory; reject links, hardlinks and inode reuse."""
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(),'tree root')
    out=[]; inodes=set()
    for base,dirs,files in os.walk(root,followlinks=False):
        dirs.sort(); files.sort()
        for name in dirs:
            p=Path(base)/name; need(not p.is_symlink(),'tree symlink directory')
        for name in files:
            p=Path(base)/name; s=os.lstat(p)
            need(stat.S_ISREG(s.st_mode) and not p.is_symlink(),'regular file')
            need(s.st_nlink==1,'hardlink forbidden')
            key=(s.st_dev,s.st_ino); need(key not in inodes,'inode reuse'); inodes.add(key)
            out.append((p.relative_to(root).as_posix(),file_sha(p),s.st_size))
    return tuple(out)

def verify_manifest(root:Path, manifest:list[dict]):
    need(type(manifest) is list,'manifest')
    seen=set()
    for r in manifest:
        exact(r,('path','sha256','size'),'manifest row'); p=text(r['path'],'path')
        need(not p.startswith('/') and '..' not in Path(p).parts and p not in seen,'manifest path'); seen.add(p)
        sha(r['sha256'],'member sha'); integer(r['size'],'member size',0)
    observed=safe_tree(root)
    need(observed==tuple((r['path'],r['sha256'],r['size']) for r in manifest),'exact tree mismatch')

def verify_archive(path:Path, expected_sha:str, expected_ledger_sha:str):
    need(file_sha(path)==sha(expected_sha,'external archive sha'),'archive anchor')
    names=set(); inventory=[]
    with tarfile.open(path,'r:gz') as t:
        for m in t.getmembers():
            need(m.name not in names,'duplicate archive member'); names.add(m.name)
            need(m.uid==m.gid==0 and m.mtime==0 and m.uname==m.gname=='','archive metadata')
            need(m.isfile(),'archive regular members only')
            need(not m.name.startswith('/') and '..' not in Path(m.name).parts,'archive path')
            inventory.append(m.name)
        need('source-ledger.json' in names,'source ledger absent')
        b=t.extractfile('source-ledger.json').read()
        need(hashlib.sha256(b).hexdigest()==sha(expected_ledger_sha,'external ledger sha'),'ledger anchor')
    return tuple(inventory)

def validate_fault_set(v):
    exact(v,('schema_version','designer_attestation','faults'),'fault set')
    need(v['schema_version']=='forkaudit-v4-fault-set-v1','schema')
    a=v['designer_attestation']; exact(a,('snapshot_sha256','isolated','no_prior_faults_seen'),'attestation')
    sha(a['snapshot_sha256'],'snapshot'); boolean(a['isolated'],'isolated'); boolean(a['no_prior_faults_seen'],'blind')
    fs=v['faults']; need(type(fs) is list and len(fs)==8,'eight faults')
    mechanisms=[]
    for i,r in enumerate(fs):
        exact(r,('fault_id','mechanism_family','activation_call_index','token_id','fixed_payload','implementation_mutation'),'fault')
        need(r['fault_id']==FAULT_IDS[i],'ordered ids'); mechanisms.append(text(r['mechanism_family'],'mechanism'))
        integer(r['activation_call_index'],'call',0,15); integer(r['token_id'],'token',0,248319)
        need(type(r['fixed_payload']) is dict,'payload'); text(r['implementation_mutation'],'mutation')
    need(len(set(mechanisms))==8,'mechanism uniqueness')

def validate_runner(root:Path, manifest:list[dict], command:list[str]):
    verify_manifest(root,manifest)
    need(type(command) is list and command and all(type(x) is str and x for x in command),'command')
    exe=Path(command[0]); need(exe.is_absolute() and exe.is_file() and not exe.is_symlink(),'absolute executable')
    need(exe.resolve().is_relative_to(root.resolve()),'executable outside runner')
    listed={r['path'] for r in manifest}; need(exe.relative_to(root).as_posix() in listed,'executable unbound')
    for x in command: need('{' not in x and '}' not in x,'templates forbidden')

def parse_gpus(rows, process_rows, torch_uuids):
    need(type(rows) is list and len(rows)==8 and type(torch_uuids) is list and len(torch_uuids)==8,'eight GPUs')
    need(process_rows==[],'compute process query must be truly empty')
    uuids=[]
    for i,r in enumerate(rows):
        exact(r,('index','name','uuid','memory_used_mib'),'gpu'); integer(r['index'],'index'); need(r['index']==i,'order')
        need(r['name']==EXACT_SKU,'exact H20-3e SKU'); u=text(r['uuid'],'uuid'); need(u.startswith('GPU-'),'uuid'); uuids.append(u)
        integer(r['memory_used_mib'],'memory',0,0)
    need(len(set(uuids))==8 and uuids==torch_uuids,'torch physical UUID binding')
    return tuple(uuids)

def validate_allocator(obs):
    exact(obs,('H0','H1','H4','H6','H7','sync_event_ids','run_id','lane','device_uuid'),'allocator')
    vals=[integer(obs[k],k,0) for k in ('H0','H1','H4','H6','H7')]
    need(vals[0]==vals[1],'H0 peak=current'); need(vals[2]>=vals[1] and vals[3]>=vals[2],'peak monotone'); need(vals[4]==vals[0],'restoration')
    ids=obs['sync_event_ids']; need(type(ids) is list and len(ids)==len(set(ids)),'unique sync IDs')
    text(obs['run_id'],'run'); need(obs['lane'] in LANES,'lane'); text(obs['device_uuid'],'device')

def validate_terminals(root:Path, pre:dict, post:dict):
    need(pre==post,'pre/post method hashes')
    inv=safe_tree(root); expected={f'terminals/{x}.json' for x in FAULT_IDS}|{'execution-terminal.json','binding.json'}
    need({p for p,_,_ in inv}==expected,'exact terminal tree')
    for fid in FAULT_IDS:
        v=json.loads((root/f'terminals/{fid}.json').read_text())
        exact(v,('fault_id','status','pre_hashes','post_hashes'),'terminal')
        need(v=={'fault_id':fid,'status':'success','pre_hashes':pre,'post_hashes':post},'successful terminal')

