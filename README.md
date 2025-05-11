# E6691 Spring 2025: Final Project

## Project Title: Comparing Audio Tagging Models with fMRI Data

This project investigates how well state-of-the-art audio and multimodal models can predict human brain activity during natural audiovisual experiences. Using group-averaged fMRI data collected while subjects watched a 1-hour movie, we evaluate multiple pretrained models—including BEATs, CLIP, and CAV-MAE—as sources of feature representations for voxel-wise encoding. We aim to determine which model best explains brain responses, which brain regions align with audio vs. visual features, and how encoding accuracy varies with model layer and stimulus delay.


---

## Summary of Models Used
- **BEATs:** Transformer-based audio tagging model; produces 768-d audio embeddings and class tag probabilities.
- **CLIP:** Vision transformer (ViT-B/32) used to extract 768-d visual embeddings from movie frames.
- **CAV-MAE:** Joint audio-visual encoder producing 768-d fused representations.
- **BEATs+CLIP:** Concatenated 1536-d vector of audio and visual features for each 1-second stimulus window.

---

## Key Contributions
- Trained voxel-wise linear encoding models using audio-only, visual-only, fused, and concatenated multimodal features.
- Demonstrated that CLIP best predicts activity in visual cortex; BEATs in auditory cortex; and BEATs+CLIP best overall.
- Found that a ~4s delay between stimulus and brain response yielded peak model performance, matching hemodynamic lag.
- Performed layer-wise analysis, revealing that mid-to-late layers align better with brain activity in sensory regions.
- Compared models both globally and within auditory/visual brain masks.


---

## References
1. [BEATs GitHub](https://github.com/microsoft/BEATs)  
2. [CLIP GitHub](https://github.com/openai/CLIP)  
3. [CAV-MAE GitHub](https://github.com/YuanGongND/cav-mae)  
4. [Final Project Repository](https://github.com/ecbme6040/e6691-2025spring-project-smab-af3410-sed2195-mt3846)

---
## Team Information

- **Group ID:** SMAB  
- **Members:**
  - sed2195 — Sude Demir
  - mt3846 — Mahdi Tabesh
  - af3410 — Arsalan Firoozi

---
## Repository Organization

This repository is organized into the following folders:

- **`data/`**  
  Contains precomputed feature matrices used for fMRI encoding:
  - `cavmae_av_embeddings.mat`: Audiovisual embeddings from CAV-MAE
  - `vision_clip_features.mat`: Visual frame-level features from CLIP
  - `features_per_sec.mat`: BEATs-based audio tag probabilities per second

- **`notebooks/`**  
  Contains all major Jupyter notebooks for feature extraction and brain encoding:
  - `beats.ipynb`: Audio tag prediction using BEATs
  - `clip.ipynb`: Visual feature extraction using CLIP
  - `cavmae_av_feature_extraction.ipynb`: Audio-visual embedding extraction using CAV-MAE
  - `Brain_CLIP_BEATS.ipynb`: Linear encoding with concatenated CLIP+BEATs features
  - `Brain_CLIP_BEATS_Delay.ipynb`: Delay analysis with shifted feature windows
  - `Brain_AudioVisual.ipynb`: Encoding using fused audiovisual features (CAV-MAE)
  - `fMRI_Encoding_LinearRidge.ipynb`: General encoding evaluation and correlation analysis

- **`surfaces/`**  
  Contains cortical surface `.gii` files with voxel-wise encoding results:
  - `r_values.LH.func.gii`, `r_values.RH.func.gii`: Correlation maps for CLIP+BEATs model
  - `r_values_audioVisual.LH.func.gii`, `r_values_audioVisual.RH.func.gii`: Correlation maps for CAV-MAE model

- **`Final_Report.pdf`**  
  The final written report summarizing the full project, results, and conclusions.

- **`README.md`**  
  This file. Provides a summary of the project and explains the structure of the repository.


---





