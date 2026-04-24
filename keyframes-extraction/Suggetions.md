# Dynamic Keyframe Count per Sequence

Instead of extracting a fixed number of keyframes for all sequences, the frame count can be made dynamic per sequence.

## 1. Threshold-Based Selection
Extract all frames whose score is above an adaptive threshold computed for each sequence.

**Impact:**  
Simple to implement and naturally adjusts the number of keyframes.

## 2. Peak-Width Selection
Find the strongest peak, then keep frames around it until the score drops below a chosen percentage of the peak value.

**Impact:**  
Better matches the real length of the informative segment.

## 3. Score-Distribution Budgeting
Set the number of keyframes based on sequence statistics such as peak strength, peak width, or overall high-score area.

**Impact:**  
More stable and controllable than raw thresholding.

## Recommendation
The best option is **Peak-Width Selection** with a minimum and maximum frame cap.