"""Reference recipe for the experts-only compressed-tensors GPTQ export.

The 256 routed experts in each of 40 decoder layers use symmetric int4 GPTQ
with group size 128. Attention, linear attention, shared experts, router,
norms, embeddings, output head, and vision tensors remain BF16. One decoder
layer is resident at a time and quantization error is propagated layer by
layer. Run in an environment that provides the model architecture and XPU
PyTorch; calibration data is supplied by the caller and is not distributed.

Usage: python quantize_tiel.py SRC_BF16 CT_REFERENCE OUT CALIBRATION.jsonl
"""
import gc, glob, json, math, os, random, shutil, sys, time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

SRC, REF, OUT, CALIB = sys.argv[1:5]
N_SAMPLES = int(os.environ.get('N_SAMPLES', 384))
SEQ = int(os.environ.get('SEQ', 2048))
BATCH = int(os.environ.get('BATCH', 4))
EXPERT_CHUNK = int(os.environ.get('EXPERT_CHUNK', 32))
GROUP, BLOCK, DAMP = 128, 128, 0.01
CLIP = (1.0, 0.96, 0.92, 0.88, 0.84)
DEV = 'xpu'
torch.manual_seed(0); random.seed(0)
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'quantize.log'), 'a')


def log(*a):
    line = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(line, flush=True); LOG.write(line + '\n'); LOG.flush()


from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

# A descending sort is used for XPU routing selection for deterministic behavior.
_topk = torch.topk


def _safe_topk(x, k, dim=-1, largest=True, sorted=True, **kw):
    if x.device.type != 'xpu':
        return _topk(x, k, dim=dim, largest=largest, sorted=sorted, **kw)
    v, i = torch.sort(x, dim=dim, descending=largest)
    return torch.return_types.topk((v.narrow(dim, 0, k), i.narrow(dim, 0, k)))


torch.topk = _safe_topk

cfg = AutoConfig.from_pretrained(SRC)
tc = cfg.text_config
tc._attn_implementation = 'sdpa'
index = json.load(open(os.path.join(SRC, 'model.safetensors.index.json')))['weight_map']


def tensor(name, device=DEV):
    # Open and close one shard for each tensor so file mappings stay bounded.
    with safe_open(os.path.join(SRC, index[name]), framework='pt') as f:
        return f.get_tensor(name).to(device)


PREFIX = 'model.language_model.'
LAYER = PREFIX + 'layers.{}.'
I = tc.moe_intermediate_size
H = tc.hidden_size
E = tc.num_experts
log(f'config: layers={tc.num_hidden_layers} hidden={H} experts={E} top{tc.num_experts_per_tok} inter={I} types={set(tc.layer_types)}')

# ---------------------------------------------------------------- calibration data
tok = AutoTokenizer.from_pretrained(REF)
stream = []                                            # documents packed end to end, separated by end-of-text
for line in open(CALIB, encoding='utf-8'):
    stream += tok(json.loads(line)['text'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
windows = [stream[s:s + SEQ] for s in range(0, len(stream) - SEQ + 1, SEQ)]
random.shuffle(windows)
if len(windows) < N_SAMPLES:
    log(f'only {len(windows)} full windows; using all'); N_SAMPLES = len(windows)
N_SAMPLES -= N_SAMPLES % BATCH
ids = torch.tensor(windows[:N_SAMPLES], dtype=torch.long)
log(f'calibration: {N_SAMPLES} x {SEQ} tokens from {len(windows)} windows')

emb = tensor(PREFIX + 'embed_tokens.weight')
hs = torch.empty(N_SAMPLES, SEQ, H, dtype=torch.bfloat16, device=DEV)
for i in range(0, N_SAMPLES, 16):
    hs[i:i + 16] = F.embedding(ids[i:i + 16].to(DEV), emb)
del emb; torch.xpu.empty_cache()
nxt = torch.empty_like(hs)

rotary = M.Qwen3_5MoeTextRotaryEmbedding(config=tc, device=DEV)
pos = torch.arange(SEQ, device=DEV).view(1, 1, -1).expand(4, BATCH, -1)
text_pos = pos[0]
with torch.no_grad():
    pos_emb = rotary(hs[:BATCH], pos[1:])


# ---------------------------------------------------------------- GPTQ (batched over experts)
def gptq(W, Hs):
    """W [e, rows, cols] fp32, Hs [e, cols, cols] fp32 -> (q uint8 [e,rows,cols] 0..15 zp 8, scale bf16 [e,rows,cols/G])."""
    W = W.clone(); Hs = Hs.clone()
    e, rows, cols = W.shape
    ar = torch.arange(cols, device=W.device)
    diag = Hs[:, ar, ar]
    dead = diag == 0
    Hs[:, ar, ar] = torch.where(dead, torch.ones_like(diag), diag)
    W.masked_fill_(dead.unsqueeze(1), 0)
    Hs[:, ar, ar] += DAMP * Hs[:, ar, ar].mean(-1, keepdim=True)
    L, info = torch.linalg.cholesky_ex(Hs)
    for boost in (0.1, 1.0, 10.0):                    # not positive definite: damp those experts harder, then retry
        bad = (info != 0).nonzero().flatten().tolist()
        if not bad:
            break
        for b in bad:
            Hs[b, ar, ar] += boost * Hs[b, ar, ar].mean()
        L, info = torch.linalg.cholesky_ex(Hs)
    if bool((info != 0).any()):
        raise RuntimeError(f'Hessian not positive definite for {int((info != 0).sum())} experts even with damping')
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    del L, Hs
    Q = torch.empty(e, rows, cols, dtype=torch.uint8, device=W.device)
    S = torch.empty(e, rows, cols // GROUP, dtype=torch.bfloat16, device=W.device)
    for i1 in range(0, cols, BLOCK):
        i2 = i1 + BLOCK
        W1 = W[:, :, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hinv1 = Hinv[:, i1:i2, i1:i2]
        # Choose the per-row clip factor that minimizes local round-to-nearest error.
        amax = W1.abs().amax(-1).clamp(min=1e-10)
        best, best_err = None, None
        for c in CLIP:
            s = (amax * c * 2 / 15).to(torch.bfloat16).float()
            q = torch.clamp(torch.round(W1 / s.unsqueeze(-1)) + 8, 0, 15)
            err = ((q - 8) * s.unsqueeze(-1) - W1).pow(2).sum(-1)
            if best is None:
                best, best_err = s, err
            else:
                better = err < best_err
                best = torch.where(better, s, best); best_err = torch.where(better, err, best_err)
        s = best
        S[:, :, i1 // GROUP] = s.to(torch.bfloat16)
        for i in range(BLOCK):
            w = W1[:, :, i]
            d = Hinv1[:, i, i].unsqueeze(-1)
            q = torch.clamp(torch.round(w / s) + 8, 0, 15)
            Q[:, :, i1 + i] = q.to(torch.uint8)
            err = (w - (q - 8) * s) / d
            W1[:, :, i:] -= err.unsqueeze(-1) * Hinv1[:, i, i:].unsqueeze(1)
            Err[:, :, i] = err
        if i2 < cols:
            W[:, :, i2:] -= torch.bmm(Err, Hinv[:, i1:i2, i2:])
    return Q, S


def dequant(Q, S):
    return ((Q.float() - 8) * S.float().repeat_interleave(GROUP, dim=-1)).to(torch.bfloat16)


def pack_ct(q):
    """[rows, cols] uint8 0..15 -> [rows, cols//8] int32, nibble k = column j*8+k (the served layout)."""
    nib = (q.to(torch.int32) & 0xF).reshape(q.shape[0], -1, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=q.device).view(1, 1, 8)
    return (nib << shifts).sum(dim=2, dtype=torch.int32)


# ---------------------------------------------------------------- decoder layers
done = {os.path.basename(p) for p in glob.glob(os.path.join(OUT, 'model-layer-*.safetensors'))}
state_path = os.path.join(OUT, 'hidden-after.pt')
start_layer = 0
if os.path.exists(state_path) and done:
    saved = torch.load(state_path, map_location='cpu')
    start_layer = saved['next_layer']
    hs.copy_(saved['hs'].to(DEV)); del saved
    log(f'resuming at layer {start_layer}')

LAST = int(os.environ.get('LAYERS', tc.num_hidden_layers))      # a smoke test runs only the first few layers
for li in range(start_layer, LAST):
    t0 = time.time()
    prefix = LAYER.format(li)
    names = [k for k in index if k.startswith(prefix)]
    with torch.device('meta'):
        layer = M.Qwen3_5MoeDecoderLayer(tc, li)
    sd = {k[len(prefix):]: tensor(k) for k in names}
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    if unexpected:
        raise SystemExit(f'layer {li}: unexpected tensors {unexpected[:5]}')
    for n, b in list(layer.named_buffers()):
        if b.is_meta:                                   # buffers absent from the checkpoint cannot be reconstructed safely
            raise SystemExit(f'layer {li}: buffer {n} not materialised')
    if missing:
        raise SystemExit(f'layer {li}: missing tensors {missing[:5]}')
    layer.eval()
    experts = layer.mlp.experts
    gu, dn = experts.gate_up_proj.data, experts.down_proj.data   # [E, 2I, H], [E, H, I]
    Hgu = torch.zeros(E, H, H, dtype=torch.float32, device=DEV)
    Hdn = torch.zeros(E, I, I, dtype=torch.float32, device=DEV)
    counts = torch.zeros(E, dtype=torch.long, device=DEV)

    def capture(module, args):
        x = args[0].reshape(-1, H)
        _, _, idx = module.gate(x)
        for e in torch.unique(idx).tolist():
            rows = (idx == e).any(-1).nonzero().squeeze(1)
            xe = x[rows]
            counts[e] += rows.numel()
            xf = xe.float()
            Hgu[e].addmm_(xf.T, xf)
            g, u = F.linear(xe, gu[e]).chunk(2, dim=-1)
            a = (F.silu(g) * u).float()
            Hdn[e].addmm_(a.T, a)

    hook = layer.mlp.register_forward_pre_hook(capture)
    with torch.no_grad():
        for b in range(0, N_SAMPLES, BATCH):
            layer(hs[b:b + BATCH], position_embeddings=pos_emb, attention_mask=None, position_ids=text_pos)
    hook.remove()
    t1 = time.time()

    out = {}
    qerr = []
    for c0 in range(0, E, EXPERT_CHUNK):
        c1 = c0 + EXPERT_CHUNK
        Qg, Sg = gptq(gu[c0:c1].float(), Hgu[c0:c1])
        Qd, Sd = gptq(dn[c0:c1].float(), Hdn[c0:c1])
        dq_g, dq_d = dequant(Qg, Sg), dequant(Qd, Sd)
        qerr.append(((dq_g.float() - gu[c0:c1].float()).norm() / gu[c0:c1].float().norm()).item())
        for j in range(Qg.shape[0]):
            e = c0 + j
            base = f'{prefix}mlp.experts.{e}.'
            for name, q, s in (('gate_proj', Qg[j, :I], Sg[j, :I]), ('up_proj', Qg[j, I:], Sg[j, I:]), ('down_proj', Qd[j], Sd[j])):
                out[base + name + '.weight_packed'] = pack_ct(q).cpu()
                out[base + name + '.weight_scale'] = s.contiguous().cpu()
                out[base + name + '.weight_shape'] = torch.tensor(list(q.shape), dtype=torch.int64)
        gu[c0:c1] = dq_g; dn[c0:c1] = dq_d            # the quantized experts produce the next layer's inputs
        del Qg, Sg, Qd, Sd, dq_g, dq_d
    del Hgu, Hdn
    torch.xpu.empty_cache()
    t2 = time.time()

    with torch.no_grad():
        for b in range(0, N_SAMPLES, BATCH):
            nxt[b:b + BATCH] = layer(hs[b:b + BATCH], position_embeddings=pos_emb, attention_mask=None, position_ids=text_pos)
    hs, nxt = nxt, hs

    for k in names:                                   # everything but routed experts remains BF16
        if '.mlp.experts.' in k:
            continue
        out[k] = tensor(k, 'cpu')
    save_file(out, os.path.join(OUT, f'model-layer-{li:02d}.safetensors'), metadata={'format': 'pt'})
    torch.save({'next_layer': li + 1, 'hs': hs.cpu()}, state_path + '.tmp') if li % 5 == 4 else None
    if li % 5 == 4:
        os.replace(state_path + '.tmp', state_path)
    cold = int((counts == 0).sum())
    log(f'layer {li:02d} {layer.block_type}: hessians {t1 - t0:.0f}s gptq {t2 - t1:.0f}s fwd {time.time() - t2:.0f}s '
        f'tokens/expert min {int(counts.min())} median {int(counts.median())} cold {cold} gate_up rel err {sum(qerr) / len(qerr):.4f} '
        f'hidden rms {hs.float().pow(2).mean().sqrt().item():.3f}')
    del layer, sd, out, experts, gu, dn
    gc.collect(); torch.xpu.empty_cache()

if LAST < tc.num_hidden_layers:
    log(f'smoke test stopped after {LAST} layers'); sys.exit(0)

# ---------------------------------------------------------------- tensors outside decoder layers
base = {}
for k in index:
    if k.startswith(PREFIX + 'layers.') or k.startswith('mtp.'):
        continue
    base[k] = tensor(k, 'cpu')
save_file(base, os.path.join(OUT, 'model-base.safetensors'), metadata={'format': 'pt'})
log(f'base tensors: {len(base)}')

# The MTP head the served build drafts with (official BF16 weights in the fused layout vLLM's MTP loader reads).
ref_index = json.load(open(os.path.join(REF, 'model.safetensors.index.json')))['weight_map']
mtp = {}
for k, shard in ref_index.items():
    if k.startswith('mtp.'):
        with safe_open(os.path.join(REF, shard), framework='pt') as f:
            mtp[k] = f.get_tensor(k)
save_file(mtp, os.path.join(OUT, 'model-mtp.safetensors'), metadata={'format': 'pt'})
log(f'mtp tensors: {len(mtp)} (from {REF})')

weight_map = {}
for path in sorted(glob.glob(os.path.join(OUT, 'model-*.safetensors'))):
    with safe_open(path, framework='pt') as f:
        for k in f.keys():
            weight_map[k] = os.path.basename(path)
json.dump({'metadata': {'total_size': 0}, 'weight_map': weight_map}, open(os.path.join(OUT, 'model.safetensors.index.json'), 'w'), indent=1)

config = json.load(open(os.path.join(REF, 'config.json')))
q = config['quantization_config']
q['ignore'] = sorted(set(q['ignore']) | {'re:.*self_attn.*', 're:.*linear_attn.*', 're:.*shared_expert[.].*', 're:.*shared_expert_gate.*'})
q['config_groups']['config_group_0']['weights']['observer'] = 'gptq'
json.dump(config, open(os.path.join(OUT, 'config.json'), 'w'), indent=1)
for name in os.listdir(REF):
    if name.endswith('.json') and name in ('config.json', 'model.safetensors.index.json', 'quantization_config.json'):
        continue
    src = os.path.join(REF, name)
    if os.path.isfile(src) and not name.endswith('.safetensors') and not name.endswith('.md'):
        shutil.copy2(src, os.path.join(OUT, name))
log(f'DONE {OUT}: {len(weight_map)} tensors')
