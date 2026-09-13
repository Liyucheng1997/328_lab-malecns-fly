"""Reproducible MNIST training; official test split is evaluated only after selection."""
import argparse, copy, hashlib, json, time
import numpy as np
import torch
from torchvision.datasets import MNIST
from sklearn.metrics import confusion_matrix
from .model import ROOT, DATA, Reservoir, Readout

OUT = ROOT / 'artifacts'

def status(**kwargs):
    OUT.mkdir(exist_ok=True)
    tmp=OUT/'status.tmp'
    tmp.write_text(json.dumps(kwargs,ensure_ascii=False),encoding='utf-8')
    tmp.replace(OUT/'status.json')
    print(json.dumps(kwargs,ensure_ascii=False),flush=True)

def fit(x,y,vx,vy,device,epochs,name):
    torch.manual_seed(328)
    model=Readout(x.shape[1]).to(device)
    mean=x.mean(0); std=x.std(0).clamp_min(.05)
    x=((x-mean)/std).to(device); vx=((vx-mean)/std).to(device)
    y=y.to(device); vy=vy.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    history=[]; best=-1; best_state=None
    for epoch in range(epochs):
        model.train(); perm=torch.randperm(len(x),device=device); total=0.
        for ix in perm.split(256):
            opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x[ix]),y[ix])
            loss.backward(); opt.step(); total+=float(loss)*len(ix)
        model.eval()
        with torch.inference_mode(): acc=float((model(vx).argmax(1)==vy).float().mean())
        history.append({'epoch':epoch+1,'loss':total/len(x),'validation_accuracy':acc})
        if acc>best: best=acc; best_state=copy.deepcopy(model.state_dict())
        status(stage='training',model=name,**history[-1])
    model.load_state_dict(best_state)
    return model,mean,std,history

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-size',type=int,default=10000)
    p.add_argument('--val-size',type=int,default=2000)
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args(); assert 0<a.train_size<=60000-a.val_size and 0<a.val_size<60000
    start=time.time(); torch.set_num_threads(8); torch.manual_seed(328)
    status(stage='loading',device=a.device)
    train=MNIST(str(ROOT/'data'),train=True,download=True)
    test=MNIST(str(ROOT/'data'),train=False,download=True)
    order=np.random.default_rng(328).permutation(60000)
    ti=order[:a.train_size]; vi=order[-a.val_size:]
    assert not np.intersect1d(ti,vi).size
    images=torch.cat([train.data[ti],train.data[vi],test.data]).reshape(-1,784).float()/255
    labels=torch.cat([train.targets[ti],train.targets[vi],test.targets])
    reservoir=Reservoir(a.device)
    status(stage='extracting',neurons=reservoir.n,edges=reservoir.manifest['edges'],total=len(images),done=0)
    cache_key=hashlib.sha256((str(vars(a))+hashlib.sha256((ROOT/'digitlab/model.py').read_bytes()).hexdigest()+hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()).encode()).hexdigest()[:16]
    cache=OUT/f'features-{cache_key}.pt'
    if cache.exists(): features=torch.load(cache,weights_only=True)
    else:
        features=torch.empty((len(images),reservoir.features))
        for i in range(0,len(images),a.batch_size):
            features[i:i+a.batch_size]=reservoir.encode(images[i:i+a.batch_size]).cpu()
            if i//a.batch_size%10==0:
                status(stage='extracting',done=min(i+a.batch_size,len(images)),total=len(images),elapsed_seconds=round(time.time()-start,1))
        torch.save(features,cache)
    t=a.train_size; v=t+a.val_size
    model,mean,std,history=fit(features[:t],labels[:t],features[t:v],labels[t:v],a.device,a.epochs,'connectome')
    model.eval()
    with torch.inference_mode():
        pred=torch.cat([model(((b-mean)/std).to(a.device)).argmax(1).cpu() for b in features[v:].split(256)])
        acc=float((pred==labels[v:]).float().mean())
        # Interventional test: same fitted readout, recurrent edges disabled.
        off=torch.cat([reservoir.encode(b,recurrent=False).cpu() for b in images[v:].split(a.batch_size)])
        offpred=torch.cat([model(((b-mean)/std).to(a.device)).argmax(1).cpu() for b in off.split(256)])
        offacc=float((offpred==labels[v:]).float().mean())
    torch.save({'state':model.cpu().state_dict(),'mean':mean,'std':std,'seed':328,'features':4096,'steps':3},OUT/'readout.tmp')
    (OUT/'readout.tmp').replace(OUT/'readout.pt')
    baseline,bmean,bstd,bhistory=fit(images[:t],labels[:t],images[t:v],labels[t:v],a.device,a.epochs,'pixel_baseline')
    baseline.eval()
    with torch.inference_mode():
        bp=torch.cat([baseline(((b-bmean)/bstd).to(a.device)).argmax(1).cpu() for b in images[v:].split(256)])
    report={'dataset':'MNIST','train_size':t,'validation_size':a.val_size,'test_size':len(test),
        'test_accuracy':acc,'recurrent_disabled_test_accuracy':offacc,
        'pixel_baseline_test_accuracy':float((bp==labels[v:]).float().mean()),
        'neurons':reservoir.n,'directed_edges':reservoir.manifest['edges'],
        'synapse_count':reservoir.manifest['synapses'],'seed':328,'device':a.device,
        'gpu':torch.cuda.get_device_name() if a.device=='cuda' else None,
        'history':history,'baseline_history':bhistory,'confusion_matrix':confusion_matrix(labels[v:],pred).tolist(),
        'elapsed_seconds':round(time.time()-start,2),'manifest_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
        'model_sha256':hashlib.sha256((ROOT/'digitlab/model.py').read_bytes()).hexdigest(),
        'train_sha256':hashlib.sha256((ROOT/'digitlab/train.py').read_bytes()).hexdigest(),
        'limitations':['Artificial image encoder, fixed rate reservoir, trained MLP readout; not the LIF movement simulator.',
        'Input is mapped across all retained neurons, not a reconstructed retina.',
        'Zero-sign outputs have no fast current, following the upstream transmitter assumptions.',
        'Pixel baseline uses the same hidden width, not a matched parameter count.',
        'Recurrent-disabled is an intervention without retraining; not proof of biological advantage.',
        'Single seed; no claim of biological learning or superiority over conventional networks.']}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'split_indices.npz',train=ti,validation=vi)
    status(stage='complete',test_accuracy=acc,baseline_accuracy=report['pixel_baseline_test_accuracy'],elapsed_seconds=report['elapsed_seconds'])

if __name__=='__main__': main()
