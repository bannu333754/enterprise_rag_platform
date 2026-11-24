import os
import shutil
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

print("=" * 80)
print("DAY 73: DOMAIN EMBEDDING FINE-TUNING PIPELINE (NATIVE PYTORCH)")
print("=" * 80)

output_dir = "./models/day73_enterprise_embedder"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

train_pairs = [
    ("How does HyperionDB handle cross-region write consensus?",
     "HyperionDB employs a decentralized Raft variant called HyperRaft operating across multi-cloud regions with sub-5ms write acknowledgment."),
    ("What is the max partition throughput for ChronosQueue?",
     "ChronosQueue partitions achieve up to 45,000 events/second using zero-copy memory mapping on dedicated NVMe drives."),
    ("How does AetherMesh enforce service isolation?",
     "AetherMesh service mesh deploys dynamic eBPF socket filters to restrict cross-namespace pod traffic without sidecar proxies."),
    ("Where are cold audit traces stored in SentinelAudit?",
     "SentinelAudit encrypts immutable audit logs with AES-256-GCM and streams them to Glacier Deep Archive cold storage.")
]

print(f"Loaded {len(train_pairs)} proprietary enterprise training pairs.")

base_model_name = "all-MiniLM-L6-v2"
print(f"Loading base embedding model: {base_model_name}...")
model = SentenceTransformer(base_model_name)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scale = 20.0

print("Executing domain fine-tuning via Multiple Negatives Ranking (MNR) loss...")
model.train()

queries = [p[0] for p in train_pairs]
docs = [p[1] for p in train_pairs]

q_tokens = model.tokenize(queries)
d_tokens = model.tokenize(docs)

q_out = model(q_tokens)["sentence_embedding"]
d_out = model(d_tokens)["sentence_embedding"]

q_norm = F.normalize(q_out, p=2, dim=1)
d_norm = F.normalize(d_out, p=2, dim=1)

scores = torch.mm(q_norm, d_norm.transpose(0, 1)) * scale
labels = torch.arange(len(queries), dtype=torch.long)

loss = F.cross_entropy(scores, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")
print(f"Saving fine-tuned domain model to: {output_dir}")
model.save(output_dir)
print("[COMPLETED] Domain embedding model successfully fine-tuned and saved.")
