import json
import numpy as np
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Compute Jaccard Index using object encodings")

parser.add_argument(
    "--generated_scenes",
    required=True,
    type=str,
    help="Path to the folder containing the predicted scenes",
)

parser.add_argument(
    "--groundtruth_scenes",
    required=True,
    type=str,
    help="Path to the folder containing the ground truth scenes",
)

parser.add_argument(
    "--generated_encodings",
    required=True,
    type=str,
    help="Path to the JSON file containing predicted scene encodings",
)

parser.add_argument(
    "--groundtruth_encodings",
    required=True,
    type=str,
    help="Path to the JSON file containing ground truth scene encodings",
)

parser.add_argument(
    "--tau",
    required=True,
    type=float,
    default=0.25,
    help="Distance threshold for matching objects",
)

args = parser.parse_args()


# Function to compute Euclidean distance between two 2D coordinates
def dist(coords1, coords2):
    return np.linalg.norm(np.array(coords1) - np.array(coords2))


# Function to split encoding list into chunks of 512
def split_into_chunks(encoding_list, chunk_size=512):
    return [encoding_list[i:i+chunk_size] for i in range(0, len(encoding_list), chunk_size)]


# Function to count matching values between two 512-number chunks
def count_matching_values(chunk1, chunk2):
    if len(chunk1) != len(chunk2):
        return 0
    return sum(1 for a, b in zip(chunk1, chunk2) if a == b)


# Function to check if there are duplicate objects (identical 512 lists) in ground truth
def has_duplicate_objects(gt_chunks):
    for i in range(len(gt_chunks)):
        for j in range(i+1, len(gt_chunks)):
            if gt_chunks[i] == gt_chunks[j]:
                return True
    return False


# Function to compute the Jaccard Index using encodings
def compute_jaccard_index(ground_truth_scenes, predicted_scenes, gt_encodings, pred_encodings, tau):
    total_number_of_scenes = len(ground_truth_scenes)
    assert total_number_of_scenes == len(predicted_scenes)
    total_jaccard_index = 0
    
    duplicate_counter = 0
    object_count_mismatch_counter = 0
    processed_scenes = 0
    total_true_positives = 0
    total_false_positives = 0  
    total_false_negatives = 0

    for gt_scene, pred_scene in zip(ground_truth_scenes, predicted_scenes):
        # Get encodings for this scene
        gt_encoding_key = gt_scene["key"]
        pred_encoding_key = pred_scene["key"]
        
        if gt_encoding_key not in gt_encodings:
            print(f"Warning: Ground truth encoding key {gt_encoding_key} not found, skipping scene")
            continue
        if pred_encoding_key not in pred_encodings:
            print(f"Warning: Predicted encoding key {pred_encoding_key} not found, skipping scene")
            continue
            
        gt_encoding_list = gt_encodings[gt_encoding_key]
        pred_encoding_list = pred_encodings[pred_encoding_key]
        
        # Split into 512-number chunks
        gt_chunks = split_into_chunks(gt_encoding_list)
        pred_chunks = split_into_chunks(pred_encoding_list)
        
        # Check for duplicate objects in ground truth
        if has_duplicate_objects(gt_chunks):
            duplicate_counter += 1
            continue
            
        # Check if number of objects match
        if len(gt_chunks) != len(pred_chunks):
            object_count_mismatch_counter += 1
            continue
        
        # Get actual objects for coordinate information
        gt_objects = gt_scene["objects"]
        pred_objects = pred_scene["objects"]
        
        # Ensure object counts match the encoding chunks
        if len(gt_objects) != len(gt_chunks) or len(pred_objects) != len(pred_chunks):
            print(f"Warning: Object count doesn't match encoding chunks for scene {gt_encoding_key}, skipping")
            continue

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Create flags to track matched ground truth objects
        gt_matched = [False] * len(gt_chunks)

        # Match predictions to ground truth using encodings
        for pred_idx, pred_chunk in enumerate(pred_chunks):
            matched = False
            best_match_score = -1
            best_match_idx = -1
            
            # Find the best matching ground truth chunk
            for gt_idx, gt_chunk in enumerate(gt_chunks):
                if not gt_matched[gt_idx]:  # Check if ground truth object is unmatched
                    match_score = count_matching_values(pred_chunk, gt_chunk)
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_idx = gt_idx
            
            # Check if best match meets distance requirement
            if best_match_idx != -1:
                # Check distance condition using the corresponding objects
                pred_obj = pred_objects[pred_idx]
                gt_obj = gt_objects[best_match_idx]
                
                try:
                    if dist(pred_obj["3d_coords"][:2], gt_obj["3d_coords"][:2]) < tau:
                        gt_matched[best_match_idx] = True
                        matched = True
                        true_positives += 1
                except:
                    continue
                    
            if not matched:
                false_positives += 1

        # Count unmatched ground truths
        false_negatives += gt_matched.count(False)

        if (true_positives + false_positives + false_negatives) == 0:
            jaccard_index = -1
        else:
            jaccard_index = true_positives / (
                true_positives + false_positives + false_negatives
            )
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        total_jaccard_index += jaccard_index
        processed_scenes += 1

    # Compute average Jaccard Index
    if processed_scenes > 0:
        jaccard_index = total_jaccard_index / processed_scenes
    else:
        jaccard_index = 0
        
    print(f"Jaccard Index: {jaccard_index:.4f}")
    print(f"True Positives: {total_true_positives}")
    print(f"False Positives: {total_false_positives}")
    print(f"False Negatives: {total_false_negatives}")
    print(f"Scenes with duplicate objects (skipped): {duplicate_counter}")
    print(f"Scenes with object count mismatch (skipped): {object_count_mismatch_counter}")
    print(f"Total processed scenes: {processed_scenes}")
    
    return jaccard_index


# Load encoding files
print("Loading encoding files...")
with open(args.groundtruth_encodings, "r") as f:
    gt_encodings = json.load(f)

with open(args.generated_encodings, "r") as f:
    pred_encodings = json.load(f)

with open(f"{args.generated_scenes}", "r") as f:
    predicted_data = json.load(f)

with open(f"{args.groundtruth_scenes}", "r") as f:
    ground_truth_data = json.load(f)

ground_truth_scenes = ground_truth_data["scenes"]
predicted_scenes = predicted_data["scenes"]

print(f"Number of ground truth scenes: {len(ground_truth_scenes)}")
print(f"Number of predicted scenes: {len(predicted_scenes)}")

ground_truth_scenes = ground_truth_scenes[: len(predicted_scenes)]

jaccard_index = compute_jaccard_index(
    ground_truth_scenes, predicted_scenes, gt_encodings, pred_encodings, 
    tau=args.tau
)
