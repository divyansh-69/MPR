import os
import re
from collections import defaultdict
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import numpy as np

import CNN

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(PROJECT_ROOT, 'test_images')
MODEL_PATH = os.path.join(APP_DIR, 'plant_disease_model_1_latest.pt')

# Load model
model = CNN.CNN(39)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
# Support both full state_dict and direct torch.save(model)
if isinstance(state_dict, dict) and all(k.startswith(('conv_layers', 'dense_layers')) for k in state_dict.keys()):
    model.load_state_dict(state_dict)
else:
    model = state_dict
model.eval()

# Class mapping from index to class name
idx_to_classes = getattr(CNN, 'idx_to_classes')
class_names = [idx_to_classes[i] for i in range(len(idx_to_classes))]

# Simple preprocessing matching app.py

def preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = image.resize((224, 224))
    tensor = TF.to_tensor(image).view((-1, 3, 224, 224))
    return tensor

# Helpers to normalize strings for matching
_non_alnum = re.compile(r'[^a-z0-9]+')

def normalize_label(s):
    s = s.lower()
    s = s.replace('__', '___')  # avoid collapsing the triple separators in class_names
    s = _non_alnum.sub('_', s)
    s = s.strip('_')
    return s

# Build normalized map for classes
normalized_class_to_idx = {}
for idx, name in idx_to_classes.items():
    normalized = normalize_label(name)
    normalized_class_to_idx[normalized] = idx

# Manual fixes for filename typos seen in test_images
manual_label_normalizations = {
    'starwberry_healthy': 'strawberry___healthy',
    'starwberry_leaf_scorch': 'strawberry___leaf_scorch',
    'soyaben_healthy': 'soybean___healthy',
    'corn_northen_leaf_blight': 'corn___northern_leaf_blight',
    'apple_ceder_apple_rust': 'apple___cedar_apple_rust',
    'tomato_bacterial_spot2': 'tomato___bacterial_spot',
    'tomato_leaf_curl_virus3': 'tomato___tomato_yellow_leaf_curl_virus',
    'tomato_mold': 'tomato___leaf_mold',
    'tomato-bacterial-spot2': 'tomato___bacterial_spot',
    'tomato-leaf-curl-virus3': 'tomato___tomato_yellow_leaf_curl_virus',
}

# Derive expected class idx from filename

def infer_expected_idx_from_filename(filename):
    stem = os.path.splitext(filename)[0]
    stem_norm = normalize_label(stem)
    # Apply manual fixes first
    if stem_norm in manual_label_normalizations:
        stem_norm = manual_label_normalizations[stem_norm]
    # Try full match against normalized class names
    if stem_norm in normalized_class_to_idx:
        return normalized_class_to_idx[stem_norm]
    # Try partial matching by requiring that all tokens of a candidate class are in the stem
    tokens = set(stem_norm.split('_'))
    best_idx = None
    best_score = -1
    for norm_class, idx in normalized_class_to_idx.items():
        class_tokens = set(norm_class.split('_'))
        score = len(class_tokens.intersection(tokens))
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def predict_image(img_path):
    with torch.no_grad():
        tensor = preprocess_image(img_path)
        outputs = model(tensor)
        outputs = outputs.detach().numpy()
        pred_idx = int(np.argmax(outputs, axis=1)[0])
        return pred_idx


def main():
    if not os.path.isdir(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return

    image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found in test_images.")
        return

    total = 0
    correct = 0
    per_class_counts = defaultdict(int)
    per_class_correct = defaultdict(int)
    mismatches = []

    for fname in sorted(image_files):
        expected_idx = infer_expected_idx_from_filename(fname)
        img_path = os.path.join(TEST_DIR, fname)
        pred_idx = predict_image(img_path)

        total += 1
        per_class_counts[expected_idx] += 1
        if pred_idx == expected_idx:
            correct += 1
            per_class_correct[expected_idx] += 1
        else:
            mismatches.append((fname, expected_idx, pred_idx))

    accuracy = 100.0 * correct / total if total else 0.0
    print(f"Overall accuracy: {accuracy:.2f}% ({correct}/{total})")

    # Per-class summary for classes that appeared in test set
    print("\nPer-class results (only classes present in test_images):")
    for idx in sorted(per_class_counts.keys()):
        cnt = per_class_counts[idx]
        corr = per_class_correct.get(idx, 0)
        name = idx_to_classes[idx]
        acc = 100.0 * corr / cnt if cnt else 0.0
        print(f"- {name}: {acc:.2f}% ({corr}/{cnt})")

    if mismatches:
        print("\nSome mismatches:")
        for fname, exp, pred in mismatches[:10]:
            exp_name = idx_to_classes.get(exp, str(exp))
            pred_name = idx_to_classes.get(pred, str(pred))
            print(f"  {fname}: expected -> {exp_name} | predicted -> {pred_name}")


if __name__ == '__main__':
    main()
