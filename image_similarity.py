'''Measuring Image Similarity'''

import os
from PIL import Image
import numpy as np
import cv2
import skimage.metrics
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from tqdm.contrib.concurrent import process_map
# REQUIRES: pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

'''Lazy load images'''
# Specify image_dir when mass-loading images from a directory
def load_images(image_paths, image_dir=''):
  return [Image.open(image_dir + path) for path in image_paths]

'''Determine pairwise structural similarity index measure (SSIM)'''
def structural_similarity(image1, image2, visualize=False):
  # This method is not invariant to transformations
  # Ensure both images are grayscale
  if image1.mode != 'L' or image2.mode != 'L':
    image1 = image1.convert('L')
    image2 = image2.convert('L')
  # Resize to match dimensions
  if image1.size != image2.size:
    if image1.size > image2.size:
      image1 = image1.resize(image2.size)
    else:
      image2 = image2.resize(image1.size)
  # Calculate SSIM
  score, diff = skimage.metrics.structural_similarity(np.array(image1), np.array(image2), full=True)
  # Visualize differences if flag is true
  if visualize:
    diff = (diff * 255).astype(np.uint8)
    _, threshold = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _  = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.array(image2.convert('RGB')).copy()
    for c in contours:
      cv2.drawContours(filled, [c], 0, (0, 255, 0), -1)
    plt.figure()
    plt.imshow(filled, cmap='gray')
    plt.axis('off')
    plt.show()
  return score

def __structural_comparison_worker(image1, image2):
  score = structural_similarity(Image.open(image1), Image.open(image2))
  return [image1.split('/')[-1], score]

def structural_comparison(image, dataset, top_k=50, use_multiprocessing=True):
  if use_multiprocessing:
    max_workers = multiprocessing.cpu_count()
    matches = process_map(partial(__structural_comparison_worker, image2=image), dataset, max_workers=max_workers, chunksize=1)
  else:
    matches = []
    for i in dataset:
      score = structural_similarity(Image.open(i), Image.open(image))
      matches += [[i.split('/')[-1], score]]
  matches = np.array(matches, dtype=object)
  return matches[np.argsort(matches[:, 1])][::-1][:top_k]

'''Helper function to convert scores to matches format'''
def format(dataset, scores):
  matches = []
  for score, idx in scores:
    matches += [[dataset[int(idx)-1].filename.split('/')[-1], score]]
  return np.array(matches, dtype=object)

'''Use dense vector representations (ViT) to determine cosine similarity scores and return top K matches'''
def dense_vector_comparison(image, dataset, top_k=50, multiprocessing=True, device='cuda'):
  # This method is invariant to transformations
  model = SentenceTransformer('clip-ViT-B-32', device=device)
  if multiprocessing:
    # Use the power of multiprocessing!
    pool = model.start_multi_process_pool()
    encoded = model.encode_multi_process([image] + dataset, pool)
  else:
    encoded = model.encode([image] + dataset)
  scores = np.array(util.paraphrase_mining_embeddings(encoded, top_k=top_k), dtype=object)
  scores = (scores[np.where(scores[:,1] == 0)[0]])[:,[0,2]]
  return format(dataset, scores)

'''Combine scores from both methods'''
def combined_score(ssim, dvrs):
  # This worked well in my testing but please take this with a grain of salt!
  return (ssim**0.5) * (dvrs**2)

'''Example Usage'''
# X = load_images(os.listdir('/content/CheXViz/Chexpert/0/'), '/content/CheXViz/Chexpert/0/')
# structural_comparison(X[0], X)
# dense_vector_comparison(X[0], X)
