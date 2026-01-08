# MovieLens 1M Splits

## Split Strategy

**Two-stage split to balance temporal realism with evaluation interpretability:**

1. **Time-based split** (80/10/10):
   - Train: Earliest 80% of interactions by timestamp
   - Validation: Next 10% of interactions
   - Test: Latest 10% of interactions

2. **Minimum history filter**:
   - Validation and test sets filtered to only include users with ≥5 ratings in training
   - Reduces cold-start users from ~54% to near 0%
   - Makes ranking metrics more interpretable (focuses on warm-start performance)
   - Maintains temporal ordering within retained ratings

## Files

- `train_ratings.parquet`: Training ratings (user_id, movie_id, rating, timestamp)
- `val_ratings.parquet`: Validation ratings (filtered for users with history)
- `test_ratings.parquet`: Test ratings (filtered for users with history)
- `movies.parquet`: Movie metadata (movie_id, title, genres)
- `users.parquet`: User metadata (user_id, gender, age, occupation, zip_code)

## Notes

- **Cold-start handling**: Val/test focus on warm users. Cold-start can be evaluated separately if needed.
- **Coverage**: ~70-80% of original val/test ratings retained after filtering
- Timestamps represent when the rating was made
- Ratings are on a 1-5 scale (integer values)
- Movies: Very few new movies in val/test (<1%)
