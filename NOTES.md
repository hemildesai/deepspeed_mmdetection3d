## Data cleaning
Corrupt file - data/lyft/v1.01-train/lidar/host-a011_lidar1_1233090652702363606.bin

```python
# proces corrupt files in sweeps
with open("corruptfile", "rb") as f:
    data = f.read()

pts = np.frombuffer(data, dtype=np.float32)
pts_processed = np.delete(pts, [-1,-2,-3])
pts_bytes = pts_processed.tobytes()
with open("processedfile", "wb") as f:
    f.write(pts_bytes)
```

## Tensorboard files
- https://tensorboard.dev/experiment/LI0FZbcnT0OX3HKKnVOjuQ/#scalars
- https://tensorboard.dev/experiment/VSoPJVXNQcaUO5wgsKVAHQ/