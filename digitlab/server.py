"""Local-only inference and experiment dashboard."""
import json, threading, time, subprocess, sys, os
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from scipy.ndimage import center_of_mass, shift
from torchvision.datasets import MNIST
from .model import ROOT, Reservoir, Readout

app=FastAPI(title='MaleCNS Digit Lab')
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])
lock=threading.Lock(); runtime=None
train_process=None
runtime_stamp=None
OUT=ROOT/'artifacts'

class Digit(BaseModel):
    pixels: list[float] = Field(min_length=784,max_length=784)
    normalize: bool = False
    recurrent: bool = True

class Lesson(Digit):
    label: int = Field(ge=0,le=9)

def checkpoint_path():
    base=OUT/'readout.pt'; personal=OUT/'personal-readout.pt'
    return personal if personal.exists() and personal.stat().st_mtime_ns>base.stat().st_mtime_ns else base

def prepare(pixels,normalize):
    a=np.asarray(pixels,dtype=np.float32).reshape(28,28)
    if not np.isfinite(a).all() or a.min()<0 or a.max()>1:
        raise HTTPException(422,'Pixels must be finite values between 0 and 1')
    if a.max()<.02: raise HTTPException(422,'请先写一个数字')
    if normalize:
        ys,xs=np.where(a>.05)
        if not len(ys): raise HTTPException(422,'笔迹太淡，请重新书写')
        cropped=a[ys.min():ys.max()+1,xs.min():xs.max()+1]
        h,w=cropped.shape; scale=20/max(h,w)
        small=np.asarray(Image.fromarray((cropped*255).astype('uint8')).resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.LANCZOS))/255
        a=np.zeros((28,28),np.float32); h,w=small.shape
        a[(28-h)//2:(28-h)//2+h,(28-w)//2:(28-w)//2+w]=small
        cy,cx=center_of_mass(a); a=shift(a,(13.5-cy,13.5-cx),order=1,mode='constant').clip(0,1)
    return a.astype('float32')

@torch.inference_mode(False)
def get_runtime():
    global runtime, runtime_stamp
    path=checkpoint_path()
    stamp=(str(path),path.stat().st_mtime_ns) if path.exists() else None
    if runtime is None or stamp != runtime_stamp:
        if not path.exists(): raise HTTPException(503,'模型正在训练，请稍后重试')
        torch.set_num_threads(8)
        ck=torch.load(path,map_location='cpu',weights_only=True)
        r=runtime[0] if runtime is not None else Reservoir('cpu',seed=ck['seed'],features=ck['features'],steps=ck['steps'])
        model=Readout(ck['features']); model.load_state_dict(ck['state']); model.eval()
        runtime=(r,model,ck['mean'],ck['std'])
        runtime_stamp=stamp
    return runtime

@app.get('/')
def home(): return FileResponse(ROOT/'digitlab/index.html')

@app.get('/api/status')
def get_status():
    path=ROOT/'artifacts/status.json'
    result=json.loads(path.read_text('utf-8')) if path.exists() else {'stage':'not_started'}
    result['training_active']=train_process is not None and train_process.poll() is None
    if train_process is not None and train_process.poll() not in (None,0):
        result.update(stage='failed',error='训练失败，查看 artifacts/train-ui.log')
    result['model_version']=checkpoint_path().name
    report=OUT/'metrics.json'
    result['base_test_accuracy']=json.loads(report.read_text('utf-8'))['test_accuracy'] if report.exists() else None
    lessons=OUT/'lessons.pt'
    result['lessons']=len(torch.load(lessons,weights_only=True)['labels']) if lessons.exists() else 0
    return result

@app.get('/api/metrics')
def metrics():
    path=ROOT/'artifacts/metrics.json'
    if not path.exists(): raise HTTPException(503,'Training in progress')
    return json.loads(path.read_text('utf-8'))

@app.get('/api/sample/{index}')
def sample(index:int):
    if not 0<=index<10000: raise HTTPException(404,'Sample index out of range')
    ds=MNIST(str(ROOT/'data'),train=False,download=False)
    return {'pixels':(ds.data[index].reshape(-1).float()/255).tolist(),'label':int(ds.targets[index]),'index':index}

@app.post('/api/predict')
def predict(digit:Digit):
    a=prepare(digit.pixels,digit.normalize)
    with lock, torch.inference_mode():
        r,model,mean,std=get_runtime(); start=time.perf_counter()
        f,trace=r.encode(a.reshape(1,-1),recurrent=digit.recurrent,trace=True)
        probs=model((f-mean)/std).softmax(1)[0].tolist()
        return {'digit':int(np.argmax(probs)),'probabilities':probs,'trace':trace,
                'elapsed_ms':round((time.perf_counter()-start)*1000,1),'pixels':a.reshape(-1).tolist(),
                'model_version':checkpoint_path().name}

@app.post('/api/train')
def train():
    global train_process
    with lock:
        if train_process is not None and train_process.poll() is None:
            raise HTTPException(409,'已经在训练，请等待完成')
        # Preserve the previous experiment before a user-requested retraining run.
        import shutil
        backup=OUT/'history'/time.strftime('%Y%m%d-%H%M%S')
        backup.mkdir(parents=True,exist_ok=True)
        for name in ['readout.pt','personal-readout.pt','metrics.json','status.json']:
            if (OUT/name).exists(): shutil.copy2(OUT/name,backup/name)
        from .train import status
        status(stage='loading',device='cpu')
        with (OUT/'train-ui.log').open('w',encoding='utf-8') as log:
            train_process=subprocess.Popen([sys.executable,'-u','-m','digitlab.train','--device','cpu'],cwd=ROOT,stdout=log,stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        return {'started':True,'pid':train_process.pid}

@app.post('/api/learn')
def learn(lesson:Lesson):
    global runtime_stamp
    if not lesson.recurrent: raise HTTPException(422,'学习时需要启用连接传播')
    a=prepare(lesson.pixels,lesson.normalize)
    with lock:
        if train_process is not None and train_process.poll() is None:
            raise HTTPException(409,'MNIST 训练进行中，完成后可继续教学')
        r,model,mean,std=get_runtime()
        with torch.inference_mode():
            feature=r.encode(a.reshape(1,-1)).clone()
            before=model((feature-mean)/std).softmax(1)[0,lesson.label].item()
        # Normal tensors outside inference mode so the readout can backpropagate.
        feature=feature.clone()
        file=OUT/'lessons.pt'
        lessons=torch.load(file,weights_only=True) if file.exists() else {'features':torch.empty(0,r.features),'labels':torch.empty(0,dtype=torch.long)}
        features=torch.cat([lessons['features'],feature])[-256:]
        labels=torch.cat([lessons['labels'],torch.tensor([lesson.label])])[-256:]
        replay_file=OUT/'lesson-replay.pt'
        if replay_file.exists(): replay=torch.load(replay_file,weights_only=True)
        else:
            ds=MNIST(str(ROOT/'data'),train=True,download=False)
            split=np.load(OUT/'split_indices.npz')['train'][:128]
            with torch.inference_mode(): rf=r.encode(ds.data[split].reshape(-1,784).float()/255).clone()
            replay={'features':rf.clone(),'labels':ds.targets[split]}
            torch.save(replay,replay_file)
        x=torch.cat([features,replay['features']]); y=torch.cat([labels,replay['labels']])
        x=((x-mean)/std).clone()
        model.eval() # Keep dropout off for short, repeatable supervised corrections.
        opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01)
        for _ in range(12):
            opt.zero_grad()
            losses=torch.nn.functional.cross_entropy(model(x),y,reduction='none')
            loss=losses[:len(features)].mean()+losses[len(features):].mean()
            loss.backward(); opt.step()
        with torch.inference_mode():
            after=model((feature-mean)/std).softmax(1)[0,lesson.label].item()
        ck={'state':model.state_dict(),'mean':mean,'std':std,'seed':r.seed,'features':r.features,'steps':r.steps}
        temp=OUT/'personal-readout.tmp'; torch.save(ck,temp); temp.replace(OUT/'personal-readout.pt')
        torch.save({'features':features,'labels':labels},file)
        runtime_stamp=None
        return {'learned':True,'label':lesson.label,'before':before,'after':after,'lessons':len(labels),
                'note':'当前样本训练分数，不是独立测试准确率；96.16% 仅适用于原始模型。'}
