import torch
import torch.nn as nn
from transformers import SwinForImageClassification

arch_name = "microsoft/swin-base-patch4-window7-224-in22k"
embed_dim = 128
output_dim = 1024
img_size = 224
patch_size = 4

modelclass = SwinForImageClassification
model = modelclass.from_pretrained(arch_name)
#print(model)
old_classifier = model.classifier  # Linear(1024, 21843)
old_weights = old_classifier.weight.data  # Shape: (21843, 1024)

            # Select top 1000 weights from the most relevant classes
            # Here we take the first 1000 rows, assuming they correspond to ImageNet-1K
new_weights = old_weights[:1000, :]  # Shape: (1000, 1024)
new_bias = old_classifier.bias[:1000] if old_classifier.bias is not None else None
model.classifier = nn.Linear(output_dim, 1000)
with torch.no_grad():
    model.classifier.weight.copy_(new_weights)
    if new_bias is not None:
            model.classifier.bias.copy_(new_bias)

print(model)