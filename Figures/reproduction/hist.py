import bz2,json,zipfile,pathlib,math,numpy as np
d=pathlib.Path('inputs')
def mu(k):
 z=16/k**2 if k>4 else k*k/16
 a=[1.5,1.5,1,1] if k>4 else [.5,.5,.5]
 b=[2,2,2] if k>4 else [1.5,1]
 v=t=1.
 for j in range(1000):
  t*=z*math.prod(x+j for x in a)/(math.prod(x+j for x in b)*(j+1)); v+=t
  if t<1e-15:break
 return math.log(k)-2*v/k**2 if k>4 else k*v/4
with zipfile.ZipFile(d/'mahler_uploads.zip') as z:
 for s,o in [('valid','test1'),('test1','valid')]:
  n=next(n for n in z.namelist() if '/rk_wN/' in n and '/eval.'+s+'.' in n)
  rows=list(json.loads(bz2.decompress(z.read(n))).values())
  keys={x['problem']['k']:4*math.pi**2*mu(x['problem']['k'])*x['problem']['w']/x['problem']['N'] for x in rows}
  ref=[keys[x['problem']['k']]*x['answer'][1]/x['answer'][0] for x in rows]
  print(s,'reference',min(ref),max(ref), 'bins',np.histogram(ref,bins=np.linspace(.4,1.8,71))[0].max())
  for r in [18,19,20]:
   def load(p):return np.load(f'work/1000262_{r}.{p}.npz')
   aa=load(o); f=aa['preds']/(aa['y'][:,0]/aa['y'][:,1])[:,None]
   ep=((f>.5)&(f<2)&(~np.isclose(f,.5,rtol=0,atol=1e-14))&(~np.isclose(f,2,rtol=0,atol=1e-14))).sum(0)[:300].argmax()
   a=load(s); preds=a['preds'][:,ep]; v=np.array([keys[i] for i in a['k']])/preds
   good=np.isfinite(v)&(preds!=0)
   print(r,ep,good.sum(),v[good].min(),v[good].max(),((v[good]>=.4)&(v[good]<=1.8)).mean())
