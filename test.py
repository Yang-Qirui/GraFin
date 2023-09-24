import numpy as np

# np_positions = np.array(heat_positions)
np_positions = np.array([[1, 2], [4, 6], [7, 10]])
diff = np_positions[:, np.newaxis, :] - np_positions[np.newaxis, :, :]
euclidean_dist = np.sqrt(np.sum(diff ** 2, axis=-1))
print(euclidean_dist)