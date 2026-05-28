# Project Progress

## Current status

- MVP napari GUI is implemented.
- New DINOv3 model is integrated.
- DINOv3 repository is connected locally.
- Crop size is updated to 7 x 128 x 128.
- New labeled time-series stacks are loaded successfully.
- Query stack and target stack are displayed in napari.
- DINOv3 query embedding is computed successfully from user clicks.
- Test embeddings were computed for 1000 candidate points from the A5 target stack.
- GUI can retrieve and display a test match using the DINOv3 embeddings.

## Current hypotheses

### H1 - Local search

A smaller search area around the selected point may find the human-labeled match faster than full candidate-point search.

### H2 - Grid search

I want to test different combinations of search area and candidate selection around the clicked point to find which setup gives the best balance between accuracy and search time.

## Current experiment setup 

- Model: DINOv3
- Crop size: 7 x 128 x 128
- Query stack: A1
- Target stack: A5
- Candidate points are currently generated from bright foreground regions.
- Full target-stack embeddings still need to be computed using HPC
- Current 1000 candidate embeddings are only for testing the pipeline.

## Next steps

- Request or use HPC for full DINOv3 embedding computation.
- Compute full target embeddings for A5 and B2.
- Implement local-search experiment script.
- Implement grid-search experiment script.
- Record all experiment runs in 'logs/experiments_log.xlsx'.
- Compare strategies using accuracy, runtime and number of candidates.