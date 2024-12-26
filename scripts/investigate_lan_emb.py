import numpy as np
import torch
import os

import init_path

all_possible_lang = [
    "open the middle drawer of the cabinet",
    "put the bowl on the stove",
    "put the wine bottle on top of the cabinet",
    "open the top drawer and put the bowl inside",
    "put the bowl on top of the cabinet",
    "push the plate to the front of the stove",
    "put the cream cheese in the bowl",
    "turn on the stove",
    "put the bowl on the plate",
    "put the wine bottle on the rack",
]

CLIP_emb = np.load('output_lang_emb_9tasks-CLIP.npy')
onehot_emb = np.load('output_lang_emb_9tasks-onehot.npy')

print(CLIP_emb.shape, onehot_emb.shape)

def cal_emb_similarity(emb1, emb2):
    # emb1, emb2 are all vectors
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1, emb2)

print("CLIP_emb")
for i in range(len(all_possible_lang)):
    print(f"Similarity between {all_possible_lang[i]} and T6: {cal_emb_similarity(CLIP_emb[i], CLIP_emb[4])}")
print()
print("onehot_emb")
for i in range(len(all_possible_lang)):
    print(f"Similarity between {all_possible_lang[i]} and T6: {cal_emb_similarity(onehot_emb[i], onehot_emb[4])}")

print()
print()
print("onehot_emb")
for i in range(len(all_possible_lang)):
    for j in range(len(all_possible_lang)):
        print(f"Similarity between {all_possible_lang[i]} and {all_possible_lang[j]}: {cal_emb_similarity(CLIP_emb[i], CLIP_emb[j])}")
    print()


