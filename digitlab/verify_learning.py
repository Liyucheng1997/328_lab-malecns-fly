"""Exercise actual learning and reload in a temporary artifact directory."""
import tempfile, shutil
from pathlib import Path
import numpy as np
import torch
from fastapi.testclient import TestClient
from . import server

def main():
    original=server.OUT
    with tempfile.TemporaryDirectory(prefix='malecns-learning-') as temp:
        server.OUT=Path(temp)
        for name in ['readout.pt','split_indices.npz']:
            shutil.copy2(original/name,server.OUT/name)
        with TestClient(server.app) as c:
            # Synthetic handwriting, never a held-out test sample.
            a=np.zeros((28,28),dtype=np.float32);a[4:24,13:16]=1
            data={'pixels':a.reshape(-1).tolist(),'normalize':True,'label':1}
            before=c.post('/api/predict',json=data).json()
            result=c.post('/api/learn',json=data)
            assert result.status_code==200,result.text
            result=result.json()
            assert result['after']>=result['before']
            assert c.post('/api/learn',json={**data,'label':10}).status_code==422
            assert c.post('/api/learn',json={**data,'recurrent':False}).status_code==422
            assert c.post('/api/predict',json={'pixels':[.03]*784,'normalize':True}).status_code==422
            after=c.post('/api/predict',json=data).json()
            assert after['model_version']=='personal-readout.pt'
            assert len(after['trace']['frames'])==4
            assert len(after['trace']['frame_body_ids'])==len(after['trace']['frames'][0])
            # Reload persisted checkpoint, not in-memory weights.
            server.runtime=None;server.runtime_stamp=None
            reloaded=c.post('/api/predict',json=data).json()
            np.testing.assert_allclose(after['probabilities'],reloaded['probabilities'],atol=1e-6)
            assert c.get('/api/status').json()['lessons']==1
            cors=c.options('/api/learn',headers={'Origin':'http://127.0.0.1:5173','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
            assert cors.status_code==200
            print({'learning':result,'checkpoint_reload':True,'frame_identity':True,'cors':True,'isolated_artifacts':True})
    server.OUT=original

if __name__=='__main__':main()
