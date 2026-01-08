# MovieLens 1M Splits

## Split Strategy

Time-based split to simulate real-world recommendation scenarios:
- **Train**: Earliest 80% of interactions by timestamp
- **Validation**: Next 10% of interactions
- **Test**: Latest 10% of interactions

## Files

- `train_ratings.parquet`: Training ratings (user_id, movie_id, rating, timestamp)
- `val_ratings.parquet`: Validation ratings
- `test_ratings.parquet`: Test ratings
- `movies.parquet`: Movie metadata (movie_id, title, genres)
- `users.parquet`: User metadata (user_id, gender, age, occupation, zip_code)

## Notes

- Some users/movies in validation/test may not appear in training (cold-start scenarios)
- Timestamps represent when the rating was made
- Ratings are on a 1-5 scale (integer values)
