# Model Card

## Model Details
- **Algorithm**: Random Forest Classifier  
- **Purpose**: Predicts income (>50K or <=50K) from census data  

## Intended Use
- For educational/demo purposes only  

## Training Data
- U.S. Census Bureau data (`census.csv`)  
- Processed features: workclass, education, marital-status, etc.  

## Evaluation Data
- 20% holdout test set  

## Metrics
- **Overall Performance**:  
  Precision: ~0.74 | Recall: ~0.64 | F1: ~0.68  
- **Slice Examples**:  
  - `workclass=Private`: F1=0.6856  
  - `education=Masters`: F1=0.8409  
  - Full metrics in `slice_output.txt`  

## Ethical Considerations
- Potential bias in occupation/race slices (e.g., `race=White` F1=0.685 vs `race=Other` F1=0.800)  

## Caveats and Recommendations
- Not production-ready. Monitor bias in sensitive features.  