import numpy as np
import torch
from src.config import CONTEXT_TENSOR_PATH, Y_TENSOR_PATH

y_full = torch.load(Y_TENSOR_PATH, map_location="cpu").numpy()

# Standard test set split
rng = np.random.default_rng(42)
test_indices = []
for c in range(16):
    c_idxs = np.where(y_full == c)[0]
    rng.shuffle(c_idxs)
    test_indices.extend(c_idxs[int(len(c_idxs) * 0.8) :])
rng.shuffle(test_indices)

sub_test = []
for c in range(16):
    c_test_idxs = [i for i in test_indices if y_full[i] == c]
    quota = max(1, round(5000 * len(c_test_idxs) / len(test_indices)))
    sub_test.extend(c_test_idxs[:quota])
test_indices = np.array(sub_test[:5000])

nb_test = test_indices[y_full[test_indices] == 0]
print("Total test non-buyers:", len(nb_test))

s3 = int(np.sum((nb_test >= 51313) & (nb_test < 72313)))
s4 = int(np.sum((nb_test >= 72313) & (nb_test < 81313)))
s5 = int(np.sum((nb_test >= 81313) & (nb_test < 88313)))
s6 = int(np.sum((nb_test >= 88313) & (nb_test < 91313)))

total_nb = len(nb_test)
print(f"Stratum 3 (Recent Active):  {s3} ({s3 / total_nb * 100:.1f}%)")
print(f"Stratum 4 (Hist Active):    {s4} ({s4 / total_nb * 100:.1f}%)")
print(f"Stratum 5 (Recent Dormant): {s5} ({s5 / total_nb * 100:.1f}%)")
print(f"Stratum 6 (Hist Dormant):   {s6} ({s6 / total_nb * 100:.1f}%)")
print(f"Total Dormant in Test Set:  {s5 + s6} ({(s5 + s6) / total_nb * 100:.1f}%)")

# Now let's compare accuracy of Uniform vs Frankenstein ON ACTIVE vs DORMANT test non-buyers!
u = np.load("data/benchmark_output/uniform_logits_cache.npz", allow_pickle=True)
f = np.load("data/benchmark_output/frankenstein_logits.npz", allow_pickle=True)

preds_u = np.argmax(u["probs"], axis=1)
preds_f = np.argmax(f["probs"], axis=1)

is_dormant_nb_in_test = np.isin(test_indices, nb_test[(nb_test >= 81313)])
is_active_nb_in_test = np.isin(test_indices, nb_test[(nb_test < 81313)])

print("\n--- Accuracy on Test ACTIVE Non-Buyers (Strata 3 & 4) ---")
print(f"Count: {np.sum(is_active_nb_in_test)}")
print(f"Uniform Hold Acc:      {np.mean(preds_u[is_active_nb_in_test] == 0) * 100:.2f}%")
print(f"Frankenstein Hold Acc: {np.mean(preds_f[is_active_nb_in_test] == 0) * 100:.2f}%")

print("\n--- Accuracy on Test DORMANT Non-Buyers (Strata 5 & 6) ---")
print(f"Count: {np.sum(is_dormant_nb_in_test)}")
print(f"Uniform Hold Acc:      {np.mean(preds_u[is_dormant_nb_in_test] == 0) * 100:.2f}%")
print(f"Frankenstein Hold Acc: {np.mean(preds_f[is_dormant_nb_in_test] == 0) * 100:.2f}%")
