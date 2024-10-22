import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import os
import sounddevice as sd
from qndiag import qndiag
import time

# List of input files
input_files = [
    "Bach.mp3",
    "Bengio.mp3",
    "LeCun.mp3",
    "Schmidhuber.mp3",
    "Yudkowski.mp3",
    "Hinton.m4a",
    "Jazz.m4a",
]

num_sources = len(input_files)


def n_batch_covariances(X, winlen, n_covariances):
    """
    Compute n_covariances covariance matrices, each based on a unique random batch of length winlen from the data X.

    Parameters:
    - X: A 2D NumPy array of shape (m, k).
    - winlen: The length of the window (number of samples in each batch).
    - n_covariances: The number of covariance matrices to generate.

    Returns:
    - A 3D NumPy array of shape [n_covariances, k, k] containing the computed covariance matrices.
    """
    m, k = X.shape
    if winlen > m:
        raise ValueError(
            "winlen cannot be larger than the number of samples in X."
        )

    covariance_matrices = np.zeros((n_covariances, k, k))

    for i in range(n_covariances):
        # Randomly sample indices without replacement to ensure uniqueness within a batch
        indices = np.random.choice(m, winlen, replace=False)
        # Extract the random batch
        batch = X[indices, :]
        # Calculate the covariance matrix for this batch
        cov_matrix = np.cov(batch, rowvar=False)
        covariance_matrices[i, :, :] = cov_matrix

    return covariance_matrices


def filter_full_rank_matrices(cov_matrices, tol=None):
    """
    Filters out matrices that are not full rank from a list of
    covariance matrices, with an option to adjust the threshold for
    considering a matrix to be full rank.

    Parameters:
    - cov_matrices: A numpy array of shape [num_samples, k, k] containing covariance matrices.
    - tol: Tolerance for rank determination. If None, the function automatically determines a suitable tolerance based on the matrix size and the precision of its elements.

    Returns:
    - A filtered numpy array containing only full rank covariance matrices, based on the given tolerance.
    """
    full_rank_matrices = []

    for cov_matrix in cov_matrices:
        # Compute the rank of the covariance matrix with the specified tolerance
        rank = np.linalg.matrix_rank(cov_matrix, tol=tol)

        # Check if the matrix is full rank (rank == k)
        if rank == cov_matrix.shape[0]:
            full_rank_matrices.append(cov_matrix)

    # Convert the list of matrices back into a numpy array
    return np.array(full_rank_matrices)


# Function to load audio file and return as mono
def load_audio(file_path):
    y, sr = librosa.load(file_path, mono=True)
    return y, sr


# Load all files and find the shortest duration
durations = []
for file in input_files:
    y, sr = load_audio(file)
    durations.append(len(y))
min_duration = min(durations)

# Truncate and save as single channel MP3
for file in input_files:
    y, sr = load_audio(file)
    y_truncated = y[:min_duration] / np.max(np.abs(y[:min_duration]))
    output_file = f"single_channel_{os.path.splitext(file)[0]}.mp3"
    sf.write(output_file, y_truncated, sr, format="mp3")

# Load truncated files and arrange into matrix S
S = np.zeros((num_sources, min_duration))
for i, file in enumerate(input_files):
    y, _ = librosa.load(
        f"single_channel_{os.path.splitext(file)[0]}.mp3", mono=True
    )
    S[i, :] = y

# Generate random Gaussian 6x6 matrix A
A = np.random.randn(num_sources, num_sources)

# Mix sources
X = np.dot(A, S)

# Normalize mixed signals
X = X / np.max(np.abs(X))

# Save mixtures
for i in range(num_sources):
    output_file = f"mixture{i+1:02d}.mp3"
    sf.write(output_file, X[i] / np.max(np.abs(X[i])), sr, format="mp3")

print("Processing complete. Check the current directory for output files.")

# Start the timer
start_time = time.time()
winlen = 100
cs = n_batch_covariances(X.T, winlen, 200)
B, _ = qndiag(filter_full_rank_matrices(cs, tol=1e-10))
Y = (X.T @ B.T).T
# End the timer
end_time = time.time()
# Calculate elapsed time
elapsed_time = end_time - start_time

X_norm = np.percentile(np.abs(X), 99)
Y = Y / X_norm
Y = Y / np.max(np.abs(Y), axis=1)[:, np.newaxis]

# Print the elapsed time
print(f"Elapsed time: {elapsed_time:.4f} seconds")

# Save sources
for i in range(num_sources):
    output_file = f"sources{i+1:02d}.mp3"
    sf.write(output_file, Y[i], sr, format="mp3")
