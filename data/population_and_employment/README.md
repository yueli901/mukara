# Population and Employment Data

Pipelines for transforming ONS/NOMIS demographic and employment tables into 1 km grid tensors.

## Files
- `population_and_employment_merge.ipynb`: merges population and employment tensors.
- `population/`: population-specific split, QA, and tensor-generation workflow.
- `employment/`: employment-specific split, QA, and tensor-generation workflow.

## Output Expectations
Processed file consumed by training:
- `population_and_employment.h5`
