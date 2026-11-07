import bz2, collections, concurrent.futures, json, pathlib, re, zipfile
import numpy as np
d=pathlib.Path('inputs')
archive=d/'mahler_uploads.zip'
def job(run):
 with zipfile.ZipFile(archive) as z:
  logs=z.read(run+'/train.log').decode(errors='replace')
  res={}
  for split in ['valid','test1']:
   name=f'{run}/eval.{split}.epochs_0_499.json.bz2'
   rows=list(json.loads(bz2.decompress(z.read(name))).values())
   k=[x['problem']['k'] for x in rows]
   y=np.array([x['answer'] for x in rows])
   a=np.array([list(x['guesses_greedy'].values()) for x in rows],dtype=object)
   exact=np.zeros((len(rows),500),bool)
   factor=np.zeros_like(exact)
   preds=np.full(exact.shape,np.nan)
   for i,x in enumerate(rows):
    p,q=x['answer']; yy=p/q
    for ep,g in x['guesses_greedy'].items():
     if not isinstance(g,list) or len(g)!=2 or not all(isinstance(v,int) for v in g) or not g[1]: continue
     e=int(ep); val=g[0]/g[1]; preds[i,e]=val
     exact[i,e]=g[0]*q==p*g[1]
     factor[i,e]=exact[i,e] or yy!=0 and 0.5<=val/yy<=2
   path='work/'+run.split('/')[-1]+'.'+split+'.npz'
   np.savez_compressed(path,k=k,y=y,preds=preds,exact=exact,factor=factor)
   res[split]=[exact.sum(0).tolist(),factor.sum(0).tolist(),len(k),len(set(k)),list(rows[0]['problem'])]
  res['log']={'test1':len(re.findall('test1',logs,re.I)), 'lines':logs.splitlines()[:120]}
 return run,res
if __name__=='__main__':
 with zipfile.ZipFile(archive) as z: runs=[n.rsplit('/',1)[0] for n in z.namelist() if n.endswith('/train.log')]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool, open('work/metrics.jsonl','w') as f:
  for fut in concurrent.futures.as_completed([pool.submit(job,r) for r in runs]):
   try: f.write(json.dumps(fut.result())+'\n'); f.flush()
   except Exception as e: print(repr(e),flush=True)
