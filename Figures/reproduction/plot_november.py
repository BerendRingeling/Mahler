from pathlib import Path
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
out=Path('outputs')
d=np.load(Path('inputs')/'dataset_derived.npz')
base=dict(zip(d['k'],d['b'])); pred=dict(zip(d['k'],d['p']))
runs=[np.load(f'work/1000261_{r}.valid.npz') for r in (12,13,14)]
epochs=[int(a['exact'][:,:300].sum(0).argmax()) for a in runs]
assert epochs==[261,272,211]
k=runs[0]['k']; y=-runs[0]['y'][:,0]/runs[0]['y'][:,1]
assert all(np.array_equal(a['k'],k) and np.array_equal(a['y'],runs[0]['y']) for a in runs)
g=y-np.array([base[i] for i in k])
stages=[(f'Epoch {epoch}',[epoch]*3) for epoch in (0,8,40,100)]
stages += [('Selected checkpoints',epochs), ('Epoch 316',[316]*3)]
raw=[np.concatenate([-a['preds'][:,j] for a,j in zip(runs,es)]) for _,es in stages]
valid=[np.isfinite(z)&(z==np.floor(z)) for z in raw]
pooled=[z[m] for z,m in zip(raw,valid)]
targets=[('Phat',np.tile([pred[i] for i in k],3),'P(k)','Model prediction vs. explicit predictor P(k)'),
         ('truth',np.tile(y,3),'True valuation','True vs. predicted valuation')]
lo=int(min(z.min() for z in [*pooled,*(v for _,v,_,_ in targets)]))
hi=int(max(z.max() for z in [*pooled,*(v for _,v,_,_ in targets)]))
edges=np.arange(lo-.5,hi+1.5)
matrices=[[np.histogram2d(z,x[m],bins=(edges,edges))[0] for z,m in zip(pooled,valid)] for _,x,_,_ in targets]
assert all(h.sum()==len(z) for pair in matrices for h,z in zip(pair,pooled))
max_count=max(h.max() for pair in matrices for h in pair)
palette=['#440154','#414487','#2a788e','#22a884','#7ad151','#fde725']
def shade(n):
    if not n:return HexColor('#440154')
    value=np.log1p(n)/np.log1p(max_count)*(len(palette)-1)
    j=min(int(value),len(palette)-2); f=value-j
    a,b=HexColor(palette[j]),HexColor(palette[j+1])
    return a.__class__(*[u+(v-u)*f for u,v in zip((a.red,a.green,a.blue),(b.red,b.green,b.blue))])
for (stem,x,ylabel,heading),pair in zip(targets,matrices):
    c=canvas.Canvas(str(out/f'v2_november_confusion_{stem}.pdf'),pagesize=(993.6,619.2))
    size=228; dx=size/(hi-lo+1)
    c.setFont('Helvetica',13)
    c.drawCentredString(480,603,heading)
    for j,(h,z,m,(title,_)) in enumerate(zip(pair,pooled,valid,stages)):
        ox=68+(j%3)*280; oy=343 if j<3 else 49
        c.setFillColor(shade(0));c.rect(ox,oy,size,size,fill=1,stroke=0)
        for ix,iy in zip(*np.nonzero(h)):
            c.setFillColor(shade(h[ix,iy]))
            c.rect(ox+ix*dx,oy+iy*dx,dx,dx,fill=1,stroke=0)
        c.setStrokeColor(HexColor('#e9292f'));c.setLineWidth(.9)
        c.line(ox,oy,ox+size,oy+size)
        c.setStrokeColor(HexColor('#202020'));c.setLineWidth(.55);c.rect(ox,oy,size,size)
        c.setFillColor(HexColor('#202020'));c.setFont('Helvetica',10)
        for tick in range(0,hi+1,5):
            q=(tick-lo+.5)*dx
            c.drawCentredString(ox+q,oy-13,str(tick))
            c.drawRightString(ox-6,oy+q-3,str(tick))
        c.drawCentredString(ox+size/2,oy-27,'Predicted valuation')
        c.saveState();c.translate(ox-35,oy+size/2);c.rotate(90)
        c.drawCentredString(0,0,ylabel);c.restoreState()
        c.setFont('Helvetica',12)
        c.drawCentredString(ox+size/2,oy+size+13,f'{title} (r = {np.corrcoef(x[m],z)[0,1]:.2f})')
    for i in range(120):
        c.setFillColor(shade(np.expm1(i/119*np.log1p(max_count))))
        c.rect(892,142+i*2.7,15,2.72,fill=1,stroke=0)
    c.setFillColor(HexColor('#202020'));c.setFont('Helvetica',10)
    for n in (1,10,100,1000):
        if n<=max_count:c.drawString(912,140+322*np.log1p(n)/np.log1p(max_count),str(n))
    c.saveState();c.translate(957,301);c.rotate(90);c.drawCentredString(0,0,'Count (log scale)');c.restoreState()
    c.save()
q=100*np.stack([a['exact'][g==0,:317].mean(0) for a in runs])
mean=q.mean(0); sd=q.std(0,ddof=1)
c=canvas.Canvas(str(out/'v2_november_correct_given_G0.pdf'),pagesize=(600,340))
x0,y0,w,h=62,48,508,252
X=lambda x:x0+w*x/316
Y=lambda y:y0+h*y/80
c.setFont('Helvetica',9)
for i in range(0,81,10):
    c.setStrokeColor(HexColor('#dddddd'));c.line(x0,Y(i),x0+w,Y(i))
    c.setFillColor(HexColor('#222222'));c.drawRightString(x0-7,Y(i)-3,str(i))
for i in range(0,301,50):
    c.setStrokeColor(HexColor('#dddddd'));c.line(X(i),y0,X(i),y0+h)
    c.setFillColor(HexColor('#222222'));c.drawCentredString(X(i),y0-15,str(i))
c.setFillColor(HexColor('#d9ebd9'))
p=c.beginPath();p.moveTo(X(0),Y(min(80,mean[0]+sd[0])))
for i in range(1,317):p.lineTo(X(i),Y(min(80,mean[i]+sd[i])))
for i in range(316,-1,-1):p.lineTo(X(i),Y(max(0,mean[i]-sd[i])))
p.close();c.drawPath(p,stroke=0,fill=1)
p=c.beginPath();p.moveTo(X(0),Y(mean[0]))
for i in range(1,317):p.lineTo(X(i),Y(mean[i]))
c.setStrokeColor(HexColor('#008000'));c.setLineWidth(1);c.drawPath(p)
c.setFillColor(HexColor('#008000'))
for i,y in enumerate(mean):c.circle(X(i),Y(y),1,fill=1,stroke=0)
print('Maximum G0 uncertainty envelope:', (mean+sd).max())
c.setStrokeColor(HexColor('#222222'));c.setLineWidth(.6);c.rect(x0,y0,w,h)
c.setFillColor(HexColor('#222222'));c.drawCentredString(x0+w/2,15,'Epoch')
c.saveState();c.translate(19,174);c.rotate(90);c.drawCentredString(0,0,'Validation accuracy (%)');c.restoreState()
c.setFont('Helvetica',11);c.drawCentredString(x0+w/2,317,'Accuracy on examples with G(k) = 0')
c.save()

