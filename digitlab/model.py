"""Full MaleCNS rate reservoir, deliberately separate from the upstream LIF demo."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'simulation/public/data'

def load_graph():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    arrays = {}
    for spec in manifest['arrays']:
        chunks = []
        for part in spec['parts']:
            raw = (DATA / part['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != part['sha256']:
                raise ValueError('Checksum mismatch: ' + part['file'])
            chunks.append(np.frombuffer(gzip.decompress(raw), dtype='<u4'))
        arrays[spec['name']] = np.concatenate(chunks)
        assert len(arrays[spec['name']]) == spec['length']
    neurons = json.loads(gzip.decompress((DATA / manifest['metadata']).read_bytes()))
    n = manifest['neurons']
    indptr, indices, counts = arrays['offsets'], arrays['sources'], arrays['counts']
    assert len(neurons) == n and len(indptr) == n+1
    assert indptr[-1] == manifest['edges'] == len(indices) == len(counts)
    assert np.all(indptr[1:] >= indptr[:-1]) and indices.max() < n
    assert int(counts.sum(dtype=np.uint64)) == manifest['synapses']
    # Rows are POSTsynaptic targets, columns are PREsynaptic sources.
    signs = np.array([row[5] for row in neurons], dtype=np.float32)
    unsigned = sp.csr_matrix((counts.astype(np.float32), indices, indptr), shape=(n,n))
    denom = np.maximum(np.asarray(unsigned.sum(axis=1)).ravel(), 1)
    weights = counts.astype(np.float32) * signs[indices]
    weights /= np.repeat(denom, np.diff(indptr).astype(np.int64))
    graph = sp.csr_matrix((weights, indices, indptr), shape=(n,n))
    return graph, manifest, neurons

class Reservoir:
    def __init__(self, device='cpu', seed=328, features=4096, steps=3):
        self.device, self.seed, self.features, self.steps = device, seed, features, steps
        graph, self.manifest, self.neurons = load_graph()
        self.n = graph.shape[0]
        self.graph = torch.sparse_csr_tensor(
            torch.from_numpy(graph.indptr.astype(np.int64)),
            torch.from_numpy(graph.indices.astype(np.int64)),
            torch.from_numpy(graph.data), size=graph.shape, device=device)
        rng = np.random.default_rng(seed)
        # Artificial sparse image encoder. This is NOT measured fly retinal wiring.
        self.pixels = torch.tensor(rng.integers(0,784,(4,self.n)), device=device)
        self.gains = torch.tensor(rng.choice([-1.,1.],(4,self.n)).astype('float32'),device=device)
        self.bias = torch.tensor(rng.uniform(-.3,.3,self.n).astype('float32'),device=device)
        self.read_ids = torch.tensor(rng.choice(self.n,features,replace=False),device=device)
        self.trace_ids = torch.arange(0, self.n, 16, device=device)

    @torch.inference_mode()
    def encode(self, x, recurrent=True, trace=False):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(-1,784)
        drive = self.bias[:,None].expand(-1,len(x)).clone()
        for p,g in zip(self.pixels,self.gains):
            drive.add_(g[:,None] * x[:,p].T * .8)
        h = torch.tanh(drive)
        frames = [h[self.trace_ids,0].cpu().tolist()] if trace else None
        activity = [float(h.abs().mean())]
        for _ in range(self.steps):
            current = torch.sparse.mm(self.graph,h) if recurrent else torch.zeros_like(h)
            h = torch.tanh(.7*drive + 1.5*current)
            if trace: activity.append(float(h.abs().mean()))
            if trace: frames.append(h[self.trace_ids,0].cpu().tolist())
        output = h[self.read_ids].T.contiguous()
        if trace:
            ids = self.read_ids[:256].cpu().numpy()
            return output, {'mean_abs_activity':activity, 'body_ids':[self.neurons[int(i)][0] for i in ids],
                            'activation':h[self.read_ids[:256],0].cpu().tolist(),
                            'frames':frames,'frame_indices':self.trace_ids.cpu().tolist(),
                            'frame_body_ids':[self.neurons[int(i)][0] for i in self.trace_ids.cpu().tolist()],
                            'kind':'rate','steps':self.steps}
        return output

class Readout(nn.Module):
    def __init__(self, features=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features,256),nn.ReLU(),nn.Dropout(.15),nn.Linear(256,10))
    def forward(self,x): return self.net(x)
