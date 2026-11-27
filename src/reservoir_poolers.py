"""
reservoir_poolers.py

General-purpose, scikit-learn compatible Echo State Network (ESN) feature extractors
for univariate and multivariate time-series classification and regression.
"""

from typing import Sequence, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _validate_and_standardize_ts(X: np.ndarray) -> np.ndarray:
    """
    Validates time-series input and standardizes shape to 3D:
    (n_samples, n_timesteps, n_features).

    Accepts:
      - 2D array: (n_samples, n_timesteps) -> univariate, reshaped to (n_samples, n_timesteps, 1)
      - 3D array: (n_samples, n_timesteps, n_features) -> multivariate
    """
    X_arr = np.asarray(X)
    if X_arr.ndim == 2:
        return X_arr[:, :, np.newaxis]
    if X_arr.ndim == 3:
        return X_arr
    raise ValueError(
        f"Expected 2D (n_samples, n_timesteps) or 3D (n_samples, n_timesteps, n_features) "
        f"input array, got shape {X_arr.shape}."
    )


class _BaseESNPooler(BaseEstimator, TransformerMixin):
    """
    Base class implementing shared Echo State Network weight initialization,
    multivariate input handling, and configurable temporal pooling.
    """

    VALID_POOLING = ("mean", "std", "min", "max", "last")

    def __init__(
        self,
        n_reservoir: int = 300,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.5,
        leaking_rate: float = 0.25,
        sparsity: float = 0.1,
        washout: int = 50,
        pooling: Sequence[str] = ("mean", "std", "min", "max", "last"),
        dtype: Union[str, np.dtype] = np.float32,
        random_state: Union[int, None] = 42,
    ):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.sparsity = sparsity
        self.washout = washout
        self.pooling = pooling
        self.dtype = dtype
        self.random_state = random_state

    def _validate_params(self, n_timesteps: int):
        if not (0.0 < self.leaking_rate <= 1.0):
            raise ValueError(f"leaking_rate must be in (0, 1], got {self.leaking_rate}.")
        if not (0.0 < self.sparsity <= 1.0):
            raise ValueError(f"sparsity must be in (0, 1], got {self.sparsity}.")
        if self.washout < 0 or self.washout >= n_timesteps:
            raise ValueError(
                f"washout ({self.washout}) must be >= 0 and < n_timesteps ({n_timesteps})."
            )
        invalid = set(self.pooling) - set(self.VALID_POOLING)
        if invalid:
            raise ValueError(
                f"Invalid pooling operations {invalid}. Valid options: {self.VALID_POOLING}."
            )
        if len(self.pooling) == 0:
            raise ValueError("pooling must contain at least one statistic.")

    def fit(self, X: np.ndarray, y=None):
        """
        Initializes fixed random input and recurrent reservoir weights based on
        the input dimensionality of X.
        """
        X_3d = _validate_and_standardize_ts(X)
        _, n_timesteps, n_features = X_3d.shape
        self._validate_params(n_timesteps)

        target_dtype = np.dtype(self.dtype)
        rng = np.random.default_rng(self.random_state)

        self.n_features_in_ = n_features

        # Input projection matrix: (n_features, n_reservoir)
        self.W_in_ = rng.uniform(
            -self.input_scaling,
            self.input_scaling,
            size=(n_features, self.n_reservoir),
        ).astype(target_dtype)

        # Recurrent reservoir matrix: (n_reservoir, n_reservoir)
        W = rng.normal(0.0, 1.0, size=(self.n_reservoir, self.n_reservoir))
        mask = rng.random(W.shape) < self.sparsity
        W *= mask

        # Scale spectral radius to control Echo State Property
        eigvals = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigvals))
        if max_eig > 0:
            W *= self.spectral_radius / max_eig

        self.W_ = W.astype(target_dtype)
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, ["W_", "W_in_"])
        names = [
            f"esn_{stat}_u{unit}"
            for stat in self.pooling
            for unit in range(self.n_reservoir)
        ]
        return np.array(names, dtype=object)


class ESNTemporalPooler(_BaseESNPooler):
    """
    Full-trajectory Echo State Network temporal pooler.

    Stores the complete state tensor (n_samples, n_timesteps, n_reservoir) in memory
    before pooling. Best suited for smaller datasets or when inspecting full
    trajectory dynamics via `transform_states()`.
    """

    def transform_states(self, X: np.ndarray) -> np.ndarray:
        """
        Runs the reservoir over X and returns the full unpooled state tensor
        of shape (n_samples, n_timesteps, n_reservoir).
        """
        check_is_fitted(self, ["W_", "W_in_"])
        X_3d = _validate_and_standardize_ts(X)
        n_samples, n_timesteps, n_features = X_3d.shape

        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} input features, got {n_features}."
            )
        self._validate_params(n_timesteps)

        target_dtype = np.dtype(self.dtype)
        X_cast = X_3d.astype(target_dtype, copy=False)

        states = np.zeros((n_samples, n_timesteps, self.n_reservoir), dtype=target_dtype)
        x_t = np.zeros((n_samples, self.n_reservoir), dtype=target_dtype)

        alpha = target_dtype.type(self.leaking_rate)
        one_minus_alpha = target_dtype.type(1.0 - self.leaking_rate)

        for t in range(n_timesteps):
            u_t = X_cast[:, t, :]  # (n_samples, n_features)
            pre_act = u_t @ self.W_in_ + x_t @ self.W_
            x_t = one_minus_alpha * x_t + alpha * np.tanh(pre_act)
            states[:, t, :] = x_t

        return states

    def transform(self, X: np.ndarray) -> np.ndarray:
        states = self.transform_states(X)
        valid_states = states[:, self.washout :, :]

        pooled_blocks = []
        for stat in self.pooling:
            if stat == "mean":
                pooled_blocks.append(valid_states.mean(axis=1))
            elif stat == "std":
                pooled_blocks.append(valid_states.std(axis=1))
            elif stat == "min":
                pooled_blocks.append(valid_states.min(axis=1))
            elif stat == "max":
                pooled_blocks.append(valid_states.max(axis=1))
            elif stat == "last":
                pooled_blocks.append(valid_states[:, -1, :])

        return np.hstack(pooled_blocks)


class MemorySafeESNPooler(_BaseESNPooler):
    """
    Streaming, batch-processed Echo State Network temporal pooler.

    Computes temporal pooling statistics online without allocating the 3D
    (n_samples, n_timesteps, n_reservoir) state tensor, reducing RAM usage
    from O(N * T * R) to O(batch_size * R).
    """

    def __init__(
        self,
        n_reservoir: int = 300,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.5,
        leaking_rate: float = 0.25,
        sparsity: float = 0.1,
        washout: int = 50,
        pooling: Sequence[str] = ("mean", "std", "min", "max", "last"),
        batch_size: int = 512,
        dtype: Union[str, np.dtype] = np.float32,
        random_state: Union[int, None] = 42,
    ):
        super().__init__(
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leaking_rate=leaking_rate,
            sparsity=sparsity,
            washout=washout,
            pooling=pooling,
            dtype=dtype,
            random_state=random_state,
        )
        self.batch_size = batch_size

    def _transform_batch(self, X_batch: np.ndarray) -> np.ndarray:
        n_samples, n_timesteps, _ = X_batch.shape
        n_res = self.n_reservoir
        target_dtype = np.dtype(self.dtype)

        alpha = target_dtype.type(self.leaking_rate)
        one_minus_alpha = target_dtype.type(1.0 - self.leaking_rate)

        x_t = np.zeros((n_samples, n_res), dtype=target_dtype)

        need_mean = "mean" in self.pooling or "std" in self.pooling
        need_std = "std" in self.pooling
        need_min = "min" in self.pooling
        need_max = "max" in self.pooling

        state_sum = np.zeros((n_samples, n_res), dtype=target_dtype) if need_mean else None
        state_sq_sum = np.zeros((n_samples, n_res), dtype=target_dtype) if need_std else None
        state_min = np.full((n_samples, n_res), np.inf, dtype=target_dtype) if need_min else None
        state_max = np.full((n_samples, n_res), -np.inf, dtype=target_dtype) if need_max else None

        for t in range(n_timesteps):
            u_t = X_batch[:, t, :]  # (n_samples, n_features)
            pre_act = u_t @ self.W_in_ + x_t @ self.W_
            x_t = one_minus_alpha * x_t + alpha * np.tanh(pre_act)

            if t >= self.washout:
                if need_mean:
                    state_sum += x_t
                if need_std:
                    state_sq_sum += x_t * x_t
                if need_min:
                    np.minimum(state_min, x_t, out=state_min)
                if need_max:
                    np.maximum(state_max, x_t, out=state_max)

        valid_steps = target_dtype.type(n_timesteps - self.washout)
        if need_mean:
            feat_mean = state_sum / valid_steps
        if need_std:
            feat_var = np.maximum(state_sq_sum / valid_steps - feat_mean * feat_mean, 0.0)
            feat_std = np.sqrt(feat_var)

        pooled_blocks = []
        for stat in self.pooling:
            if stat == "mean":
                pooled_blocks.append(feat_mean)
            elif stat == "std":
                pooled_blocks.append(feat_std)
            elif stat == "min":
                pooled_blocks.append(state_min)
            elif stat == "max":
                pooled_blocks.append(state_max)
            elif stat == "last":
                pooled_blocks.append(x_t)

        return np.hstack(pooled_blocks)

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["W_", "W_in_"])
        X_3d = _validate_and_standardize_ts(X)
        n_samples, n_timesteps, n_features = X_3d.shape

        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} input features, got {n_features}."
            )
        self._validate_params(n_timesteps)

        target_dtype = np.dtype(self.dtype)
        X_cast = X_3d.astype(target_dtype, copy=False)

        batches = [
            self._transform_batch(X_cast[i : i + self.batch_size])
            for i in range(0, n_samples, self.batch_size)
        ]
        return np.vstack(batches)
