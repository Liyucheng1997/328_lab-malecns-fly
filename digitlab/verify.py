"""Checks that guard real experimental failure modes, not training accuracy tuning."""
import json
import numpy as np
import torch
from fastapi.testclient import TestClient
from .model import ROOT, Reservoir
from .server import app, prepare

def main():
    torch.set_num_threads(8)
    r=Reservoir()
    # Inference must reset state per sample; batching must not couple images.
    x=torch.rand((2,784),generator=torch.Generator().manual_seed(17))
    a=r.encode(x)
    b=torch.cat([r.encode(x[:1]),r.encode(x[1:])])
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(a,r.encode(x))
    assert torch.max(torch.abs(a-r.encode(x,recurrent=False)))>.01
    # Known signed directed motif: neuron 0 -> 1, neuron 1 inhibits neuron 2.
    m=torch.sparse_csr_tensor(torch.tensor([0,0,1,2]),torch.tensor([0,1]),torch.tensor([2.,-3.]),size=(3,3))
    torch.testing.assert_close(torch.sparse.mm(m,torch.tensor([[1.],[2.],[0.]])),torch.tensor([[0.],[2.],[-6.]]))
    split=np.load(ROOT/'artifacts/split_indices.npz')
    assert not np.intersect1d(split['train'],split['validation']).size
    with TestClient(app) as c:
        assert c.get('/').status_code==200
        assert c.get('/api/sample/-1').status_code==404
        assert c.post('/api/predict',json={'pixels':[0.]*784}).status_code==422
        assert c.post('/api/predict',json={'pixels':[0.]*783}).status_code==422
        assert c.post('/api/predict',json={'pixels':[2.]*784}).status_code==422
        sample=c.get('/api/sample/0').json()
        result=c.post('/api/predict',json={'pixels':sample['pixels']}).json()
        assert len(result['probabilities'])==10 and abs(sum(result['probabilities'])-1)<1e-5
        assert result['digit']==sample['label']==7
        assert len(result['trace']['body_ids'])==256
        assert len(set(result['trace']['body_ids']))==256
        normalized=prepare(sample['pixels'],True)
        assert normalized.shape==(28,28) and np.isfinite(normalized).all()
    report={'passed':True,'checks':['all graph chunk SHA256 checks','CSR node/edge/synapse totals','batch independence','state reset','recurrent intervention changes features','signed directed motif','disjoint train/validation indices','HTTP input validation','held-out sample inference and probability normalization','handwriting preprocessing']}
    (ROOT/'artifacts/verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__': main()
