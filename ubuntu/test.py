from argparse import ArgumentParser
from rank import Ranker
import torch
import json
import math
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = ArgumentParser()
parser.add_argument(
    "--model_checkpoint",
    type=str,
    default="models/",
    help="Path or URL of the model",
)
parser.add_argument(
    "--pretrained", action="store_true", help="If False train from scratch"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device (cuda or cpu)",
)
parser.add_argument(
    "--parallel",
    action="store_true",
    help="Use DataParallel or not",
)
parser.add_argument(
        "--data_path", type=str, default="data", help="Dxsata path"
    )
parser.add_argument(
        "--use_post_training", type=str, default="NoPost", help="Post training"
    )
args = parser.parse_args()
ranker = Ranker(args.model_checkpoint, args) 

data_path = args.data_path
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.read())

test_data = data['test']
print('test samples:', len(test_data))
total_samples = len(test_data)
R10_1 = 0
R10_2 = 0
R10_5 = 0
MRR = 0
for i in tqdm(range(len(test_data))):
    turn = test_data[i]
    history = turn["history"]
    responses = turn["responses"]
    label = turn["label"]
    pred_label, descend_labels = ranker.rank(history, responses)
    if pred_label == label:
        R10_1 += 1
    if label in descend_labels[:2]:
        R10_2 += 1
    if label in descend_labels[:5]:
        R10_5 += 1
    MRR += 1.0 / (descend_labels.index(label) + 1)

print('R10@1:', R10_1/total_samples)
print('R10@2:', R10_2/total_samples)
print('R10@5:', R10_5/total_samples)
print('MRR:', MRR/total_samples)