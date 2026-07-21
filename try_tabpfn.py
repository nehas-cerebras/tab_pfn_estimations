from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
import torch
import tabpfn

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = TabPFNClassifier(model_path="model/tabpfn-v3-classifier-v3_20260417_binary.ckpt", n_estimators=1, fit_mode="fit_with_cache", device="cuda" if torch.cuda.is_available() else "cpu")
clf.fit(X_train, y_train)
prediction_probabilities = clf.predict_proba(X_test)
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))

print("Model REPR:", clf.models_[0])


model = clf
print("tabpfn version:", getattr(tabpfn, "__version__", "unknown"))
print("torch version:", torch.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: none")

for name in [
    "fit_mode",
    "n_estimators",
    "n_estimators_",
    "inference_precision",
    "memory_saving_mode",
    "keep_cache_on_device",
    "device",
]:
    print(f"{name}:", getattr(model, name, "<not present>"))

print("number of loaded neural models:", len(getattr(model, "models_", [])))

try:
    print("inference config:")
    print(model.get_inference_config())
except Exception as exc:
    print("could not print inference config:", repr(exc))

if getattr(model, "models_", None):
    neural_model = model.models_[0]

    print("\narchitecture config:")
    print(getattr(neural_model, "config", "<not present>"))

    total_parameters = sum(p.numel() for p in neural_model.parameters())
    trainable_parameters = sum(
        p.numel() for p in neural_model.parameters() if p.requires_grad
    )

    print("total parameters:", f"{total_parameters:,}")
    print("trainable parameters:", f"{trainable_parameters:,}")

    print("\nparameter count by top-level component:")
    for component_name, component in neural_model.named_children():
        count = sum(p.numel() for p in component.parameters())
        dtypes = sorted({str(p.dtype) for p in component.parameters()})
        print(
            f"{component_name:32s}",
            f"{count:12,d}",
            f"dtypes={dtypes}",
        )

    # ---- ICL layer 0: head configuration (train vs test KV heads) ----
    attn0 = neural_model.icl_blocks[0].icl_attention
    print("\n[ICL layer 0] head config:")
    print("  num_heads (query heads)   :", attn0.num_heads)
    print("  num_kv_heads (train K/V)  :", attn0.num_kv_heads)
    print("  num_kv_heads_test (cached):", attn0.num_kv_heads_test)
    print("  head_dim                  :", attn0.head_dim)
    print("  config.icl_num_kv_heads_test:", neural_model.config.icl_num_kv_heads_test)
    print("  k_projection out_features :", attn0.k_projection.out_features,
          "(= num_kv_heads * head_dim =", attn0.num_kv_heads * attn0.head_dim, ")")