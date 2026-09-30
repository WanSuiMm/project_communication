"""Seeded region identity: exact labels from graph connectivity, NOT a PDE.

Input channels: traversable mask, positive seed, negative seed.
Every connected component gets one independent binary identity and one seed.
Models receive the image only, NOT components, graph distances, or labels.
"""
from __future__ import annotations
from collections import deque
import numpy as np
import torch


def neighbors(y,x,n):
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        yy,xx=y+dy,x+dx
        if 0<=yy<n and 0<=xx<n:
            yield yy,xx


def components(mask):
    n = len(mask)
    ids = np.full((n,n),-1,dtype=np.int32)
    groups = []
    for y in range(n):
        for x in range(n):
            if not mask[y,x] or ids[y,x]>=0:
                continue
            j=len(groups); ids[y,x]=j
            queue=deque([(y,x)]); cells=[]
            while queue:
                yy,xx=queue.popleft(); cells.append((yy,xx))
                for a,b in neighbors(yy,xx,n):
                    if mask[a,b] and ids[a,b]<0:
                        ids[a,b]=j; queue.append((a,b))
            groups.append(cells)
    return ids,groups


def sample(n:int,rng:np.random.Generator):
    if n<8:
        raise ValueError('Use side length >= 8')
    rooms=min(4,max(2,n//6))
    cuts=np.round(np.linspace(0,n-1,rooms+1)).astype(int)[1:-1]
    for _ in range(200):
        # Jitter partition positions without changing room connectivity semantics.
        cv=np.clip(cuts+rng.integers(-1,2,len(cuts)),2,n-3)
        ch=np.clip(cuts+rng.integers(-1,2,len(cuts)),2,n-3)
        mask=np.ones((n,n),dtype=bool)
        mask[[0,-1],:]=False; mask[:,[0,-1]]=False
        mask[:,cv]=False; mask[ch,:]=False
        xb=[0,*cv,n-1]; yb=[0,*ch,n-1]
        opening=float(rng.uniform(.2,.5))
        for x in cv:
            for ya,yz in zip(yb[:-1],yb[1:]):
                if yz-ya>=2 and rng.random()<opening:
                    y=int(rng.integers(ya+1,yz)); mask[y,x]=True
        for y in ch:
            for xa,xz in zip(xb[:-1],xb[1:]):
                if xz-xa>=2 and rng.random()<opening:
                    x=int(rng.integers(xa+1,xz)); mask[y,x]=True
        ids,groups=components(mask)
        if len(groups)>=2 and all(len(g)>=2 for g in groups):
            break
    else:
        raise RuntimeError('Could not generate a nondegenerate region instance')
    bits=rng.integers(0,2,len(groups))
    if np.all(bits==bits[0]):
        bits[int(rng.integers(len(groups)))]=1-bits[0]
    x=np.zeros((3,n,n),np.float32); x[0]=mask
    target=np.zeros((1,n,n),np.float32)
    distance=np.full((n,n),-1,np.int32)
    seeds=[]
    for j,cells in enumerate(groups):
        sy,sx=cells[int(rng.integers(len(cells)))]; seeds.append((sy,sx))
        x[1 if bits[j] else 2,sy,sx]=1
        yy,xx=np.array(cells).T; target[0,yy,xx]=bits[j]
        distance[sy,sx]=0; queue=deque([(sy,sx)])
        while queue:
            a,b=queue.popleft()
            for yy,xx in neighbors(a,b,n):
                if ids[yy,xx]==j and distance[yy,xx]<0:
                    distance[yy,xx]=distance[a,b]+1; queue.append((yy,xx))
    # Flip one component's source value; no geometry change.
    j=int(np.argmax([len(g) for g in groups])); sy,sx=seeds[j]
    x_flip=x.copy(); x_flip[1,sy,sx],x_flip[2,sy,sx]=x[2,sy,sx],x[1,sy,sx]
    y_flip=target.copy(); changed=(ids==j)[None]
    y_flip[changed]=1-y_flip[changed]
    return {'x':x,'y':target,'mask':mask[None].astype(np.float32),
            'distance':distance[None], 'x_flip':x_flip,'y_flip':y_flip,
            'changed':changed.astype(np.float32)}


def bank(n,count,seed,device='cpu'):
    rng=np.random.default_rng(seed)
    examples=[sample(n,rng) for _ in range(count)]
    return {k:torch.from_numpy(np.stack([e[k] for e in examples])).to(device)
            for k in examples[0]}


def subset(data,idx):
    return {k:v[idx] for k,v in data.items()}


def damage(state,fraction,rng):
    h,v=state
    if not 0<=fraction<1:
        raise ValueError('damage fraction must be in [0,1)')
    keep=torch.ones_like(h[:,:1])
    n=h.shape[-1]; side=max(1,int(round(n*np.sqrt(fraction))))
    for b in range(h.shape[0]):
        y=int(rng.integers(0,n-side+1)); x=int(rng.integers(0,n-side+1))
        keep[b,:,y:y+side,x:x+side]=0
    return h*keep, None if v is None else v*keep


def balanced_loss(logits,y,mask):
    raw=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
    pos=mask*y; neg=mask*(1-y)
    dims=(1,2,3)
    lp=(raw*pos).sum(dims)/pos.sum(dims).clamp_min(1)
    ln=(raw*neg).sum(dims)/neg.sum(dims).clamp_min(1)
    present_pos=(pos.sum(dims)>0).to(raw.dtype)
    present_neg=(neg.sum(dims)>0).to(raw.dtype)
    return ((lp+ln)/(present_pos+present_neg).clamp_min(1)).mean()


def per_example_balanced_accuracy(logits,y,mask):
    """Average recall over the classes present; source flips can remove a class."""
    good=((logits>=0)==(y>=.5)).float()
    pos=mask*y; neg=mask*(1-y); dims=(1,2,3)
    np_=pos.sum(dims); nn_=neg.sum(dims)
    pr=(good*pos).sum(dims)/np_.clamp_min(1)
    nr=(good*neg).sum(dims)/nn_.clamp_min(1)
    return (pr+nr)/((np_>0).float()+(nn_>0).float()).clamp_min(1)


def metrics(logits,y,mask,distance=None):
    with torch.no_grad():
        pred=logits>=0; truth=y>=.5; good=(pred==truth).float()
        pos=mask*y; neg=mask*(1-y)
        dims=(1,2,3)
        pr=(good*pos).sum(dims)/pos.sum(dims).clamp_min(1)
        nr=(good*neg).sum(dims)/neg.sum(dims).clamp_min(1)
        # BCE and accuracy count only traversable pixels; walls cannot inflate metrics.
        out={'balanced_accuracy':float(per_example_balanced_accuracy(logits,y,mask).mean()),
             'bce':float(balanced_loss(logits,y,mask)),
             'accuracy':float((good*mask).sum()/mask.sum().clamp_min(1))}
        if distance is not None:
            for lo,hi in ((0,8),(8,16),(16,32),(32,64),(64,128),(128,100000)):
                take=mask*((distance>=lo)&(distance<hi))
                count=int(take.sum())
                out[f'distance_{lo}_{hi}']={'count':count,
                    'accuracy':float((good*take).sum()/take.sum()) if count else None}
        return out
