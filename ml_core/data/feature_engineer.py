"""
Feature engineering module for Student Academic Risk Prediction System.

Handles feature creation, transformation, clustering-based feature generation,
and feature selection for the ML pipeline.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dataclasses import dataclass
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# Feature column definitions (Chinese names from source data)
SCL90_FACTORS: List[str] = [
    "躯体化",       # Somatization
    "强迫症状",     # Obsessive-Compulsive
    "人际关系敏感", # Interpersonal Sensitivity
    "抑郁",         # Depression
    "焦虑",         # Anxiety
    "敌对",         # Hostility
    "恐怖",         # Phobic Anxiety
    "偏执",         # Paranoid Ideation
    "精神病性",     # Psychoticism
    "其他",         # Additional Items
]

EPQ_FACTORS: List[str] = [
    "内外向E",  # Extraversion
    "神经质N",  # Neuroticism
    "精神质P",  # Psychoticism (EPQ)
    "掩饰性L",  # Lie Scale
]

FEATURE_COLUMNS: List[str] = EPQ_FACTORS + SCL90_FACTORS

TARGET_COLUMN: str = "挂科数目"

# English mappings for API/export compatibility
FEATURE_COLUMN_ENGLISH: Dict[str, str] = {
    "内外向E": "extraversion_e",
    "神经质N": "neuroticism_n",
    "精神质P": "psychoticism_p",
    "掩饰性L": "lie_scale_l",
    "躯体化": "somatization",
    "强迫症状": "obsessive_compulsive",
    "人际关系敏感": "interpersonal_sensitivity",
    "抑郁": "depression",
    "焦虑": "anxiety",
    "敌对": "hostility",
    "恐怖": "phobic_anxiety",
    "偏执": "paranoid_ideation",
    "精神病性": "psychoticism",
    "其他": "additional_items",
}


@dataclass
class FeatureStats:
    """Container for feature statistics."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q1: float
    q3: float


class ClusterFeatureGenerator:
    """
    Generates cluster-based features using K-Means clustering.
    
    This class implements the unsupervised component of the hybrid model,
    clustering students based on psychological profiles to create an
    additional feature for the supervised classifier.
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        scaling_features: Optional[List[str]] = None
    ):
        """
        Initialize cluster feature generator.
        
        Args:
            n_clusters: Number of clusters for K-Means (default: 3).
            random_state: Random seed for reproducibility.
            scaling_features: Features to use for clustering. Defaults to SCL90_FACTORS.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaling_features = scaling_features or SCL90_FACTORS
        
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        self._is_fitted: bool = False
        self._cluster_centers: Optional[np.ndarray] = None
        self._cluster_descriptions: Dict[int, str] = {}
    
    def fit(self, data: pd.DataFrame) -> "ClusterFeatureGenerator":
        """
        Fit the scaler and K-Means model on training data.
        
        Args:
            data: Training DataFrame containing scaling_features columns.
        
        Returns:
            Self for method chaining.
        
        Raises:
            ValueError: If required features are missing.
        """
        missing_features = set(self.scaling_features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features for clustering: {missing_features}")
        
        # Extract features for clustering
        X = data[self.scaling_features].values
        
        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(X_scaled)
        
        # Store cluster centers for interpretation
        self._cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self._generate_cluster_descriptions()
        
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate cluster labels for new data.
        
        Args:
            data: DataFrame containing scaling_features columns.
        
        Returns:
            Array of cluster labels.
        
        Raises:
            RuntimeError: If not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("ClusterFeatureGenerator must be fitted before transform")
        
        X = data[self.scaling_features].values
        X_scaled = self.scaler.transform(X)
        
        return self.kmeans.predict(X_scaled)
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            data: Training DataFrame.
        
        Returns:
            Array of cluster labels.
        """
        return self.fit(data).transform(data)
    
    def get_cluster_distances(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get distances to all cluster centers for each sample.
        
        Args:
            data: DataFrame containing scaling_features columns.
        
        Returns:
            Array of shape (n_samples, n_clusters) with distances.
        """
        if not self._is_fitted:
            raise RuntimeError("ClusterFeatureGenerator must be fitted before use")
        
        X = data[self.scaling_features].values
        X_scaled = self.scaler.transform(X)
        
        return self.kmeans.transform(X_scaled)
    
    def get_cluster_probabilities(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get soft cluster assignments based on inverse distances.
        
        Args:
            data: DataFrame containing scaling_features columns.
        
        Returns:
            Array of shape (n_samples, n_clusters) with probabilities.
        """
        distances = self.get_cluster_distances(data)
        # Convert distances to probabilities using softmax on negative distances
        inv_distances = 1 / (distances + 1e-10)
        probabilities = inv_distances / inv_distances.sum(axis=1, keepdims=True)
        return probabilities
    
    def _generate_cluster_descriptions(self) -> None:
        """Generate human-readable descriptions for each cluster."""
        if self._cluster_centers is None:
            return
        
        for cluster_idx in range(self.n_clusters):
            center = self._cluster_centers[cluster_idx]
            # Find the most prominent features (highest z-scores)
            overall_mean = np.mean(self._cluster_centers, axis=0)
            overall_std = np.std(self._cluster_centers, axis=0) + 1e-10
            z_scores = (center - overall_mean) / overall_std
            
            # Get top 3 distinctive features
            top_indices = np.argsort(np.abs(z_scores))[-3:][::-1]
            
            descriptions = []
            for idx in top_indices:
                feature_name = self.scaling_features[idx]
                if z_scores[idx] > 0.5:
                    descriptions.append(f"High {feature_name}")
                elif z_scores[idx] < -0.5:
                    descriptions.append(f"Low {feature_name}")
            
            if descriptions:
                self._cluster_descriptions[cluster_idx] = ", ".join(descriptions)
            else:
                self._cluster_descriptions[cluster_idx] = f"Cluster {cluster_idx} (Average)"
    
    def get_cluster_description(self, cluster_label: int) -> str:
        """
        Get description for a specific cluster.
        
        Args:
            cluster_label: Cluster index.
        
        Returns:
            Human-readable cluster description.
        """
        return self._cluster_descriptions.get(
            cluster_label,
            f"Unknown Cluster {cluster_label}"
        )
    
    def get_cluster_profile(self, cluster_label: int) -> Dict[str, float]:
        """
        Get the feature profile (center values) for a cluster.
        
        Args:
            cluster_label: Cluster index.
        
        Returns:
            Dictionary mapping feature names to center values.
        """
        if self._cluster_centers is None or cluster_label >= self.n_clusters:
            return {}
        
        center = self._cluster_centers[cluster_label]
        return {
            feature: round(value, 4)
            for feature, value in zip(self.scaling_features, center)
        }
    
    def save(self, filepath: str) -> None:
        """
        Save fitted model to file.
        
        Args:
            filepath: Path to save the model.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        artifacts = {
            "scaler": self.scaler,
            "kmeans": self.kmeans,
            "n_clusters": self.n_clusters,
            "scaling_features": self.scaling_features,
            "cluster_centers": self._cluster_centers,
            "cluster_descriptions": self._cluster_descriptions,
        }
        joblib.dump(artifacts, filepath)
    
    def load(self, filepath: str) -> "ClusterFeatureGenerator":
        """
        Load fitted model from file.
        
        Args:
            filepath: Path to the saved model.
        
        Returns:
            Self for method chaining.
        """
        artifacts = joblib.load(filepath)
        
        self.scaler = artifacts["scaler"]
        self.kmeans = artifacts["kmeans"]
        self.n_clusters = artifacts["n_clusters"]
        self.scaling_features = artifacts["scaling_features"]
        self._cluster_centers = artifacts["cluster_centers"]
        self._cluster_descriptions = artifacts["cluster_descriptions"]
        self._is_fitted = True
        
        return self


class FeatureEngineer:
    """
    Main feature engineering class for the risk prediction pipeline.
    
    Orchestrates feature creation, transformation, and selection for
    both training and inference.
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        create_interaction_features: bool = True,
        create_aggregate_features: bool = True,
        random_state: int = 42
    ):
        """
        Initialize feature engineer.
        
        Args:
            n_clusters: Number of clusters for K-Means.
            create_interaction_features: Whether to create interaction terms.
            create_aggregate_features: Whether to create aggregate statistics.
            random_state: Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.create_interaction_features = create_interaction_features
        self.create_aggregate_features = create_aggregate_features
        self.random_state = random_state
        
        self.cluster_generator = ClusterFeatureGenerator(
            n_clusters=n_clusters,
            random_state=random_state
        )
        
        self.scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False
        self._feature_names: List[str] = []
    
    def fit(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> "FeatureEngineer":
        """
        Fit the feature engineer on training data.
        
        Args:
            data: Training DataFrame with feature columns.
            target: Optional target series (for potential feature selection).
        
        Returns:
            Self for method chaining.
        """
        # Fit cluster generator
        self.cluster_generator.fit(data)
        
        # Fit scaler on all numeric features
        feature_cols = [c for c in FEATURE_COLUMNS if c in data.columns]
        self.scaler = StandardScaler()
        self.scaler.fit(data[feature_cols])
        
        # Record feature names that will be generated
        self._feature_names = self._get_feature_names(data)
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        data: pd.DataFrame,
        include_scaled: bool = True
    ) -> pd.DataFrame:
        """
        Transform data by adding engineered features.
        
        Args:
            data: Input DataFrame.
            include_scaled: Whether to include scaled versions of features.
        
        Returns:
            DataFrame with original and engineered features.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        
        result = data.copy()
        
        # Add cluster label
        result["Cluster_Label"] = self.cluster_generator.transform(data)
        
        # Add aggregate features
        if self.create_aggregate_features:
            result = self._add_aggregate_features(result)
        
        # Add interaction features
        if self.create_interaction_features:
            result = self._add_interaction_features(result)
        
        # Add scaled features if requested
        if include_scaled:
            result = self._add_scaled_features(result)
        
        return result
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        include_scaled: bool = True
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            data: Training DataFrame.
            target: Optional target series.
            include_scaled: Whether to include scaled versions.
        
        Returns:
            Transformed DataFrame.
        """
        return self.fit(data, target).transform(data, include_scaled)
    
    def get_feature_matrix(
        self,
        data: pd.DataFrame,
        feature_subset: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Get feature matrix ready for model input.
        
        Args:
            data: Input DataFrame (already transformed or raw).
            feature_subset: Specific features to include. Defaults to all.
        
        Returns:
            NumPy array of features.
        """
        if "Cluster_Label" not in data.columns:
            # Data needs transformation
            data = self.transform(data, include_scaled=False)
        
        if feature_subset:
            return data[feature_subset].values
        
        # Return all feature columns plus cluster label
        feature_cols = FEATURE_COLUMNS + ["Cluster_Label"]
        if self.create_aggregate_features:
            feature_cols.extend([
                "SCL90_Mean", "SCL90_Max", "SCL90_Std",
                "EPQ_Neuroticism_x_Anxiety"
            ])
        
        available_cols = [c for c in feature_cols if c in data.columns]
        return data[available_cols].values
    
    def get_model_features(self) -> List[str]:
        """
        Get list of feature names used by the model.
        
        Returns:
            List of feature column names.
        """
        features = FEATURE_COLUMNS.copy()
        features.append("Cluster_Label")
        
        if self.create_aggregate_features:
            features.extend([
                "SCL90_Mean", "SCL90_Max", "SCL90_Std",
                "EPQ_Neuroticism_x_Anxiety"
            ])
        
        return features
    
    def _add_aggregate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add aggregate statistical features.
        
        Args:
            data: Input DataFrame.
        
        Returns:
            DataFrame with aggregate features added.
        """
        result = data.copy()
        
        # SCL-90 aggregates
        scl90_cols = [c for c in SCL90_FACTORS if c in data.columns]
        if scl90_cols:
            result["SCL90_Mean"] = data[scl90_cols].mean(axis=1)
            result["SCL90_Max"] = data[scl90_cols].max(axis=1)
            result["SCL90_Std"] = data[scl90_cols].std(axis=1)
        
        return result
    
    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between key variables.
        
        Args:
            data: Input DataFrame.
        
        Returns:
            DataFrame with interaction features added.
        """
        result = data.copy()
        
        # Key interactions based on psychological theory
        # Neuroticism x Anxiety interaction (high risk indicator)
        if "神经质N" in data.columns and "焦虑" in data.columns:
            result["EPQ_Neuroticism_x_Anxiety"] = (
                data["神经质N"] / 100 * data["焦虑"]
            )
        
        return result
    
    def _add_scaled_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add standardized versions of features.
        
        Args:
            data: Input DataFrame.
        
        Returns:
            DataFrame with scaled features added.
        """
        result = data.copy()
        
        feature_cols = [c for c in FEATURE_COLUMNS if c in data.columns]
        scaled_values = self.scaler.transform(data[feature_cols])
        
        for idx, col in enumerate(feature_cols):
            result[f"{col}_scaled"] = scaled_values[:, idx]
        
        return result
    
    def _get_feature_names(self, data: pd.DataFrame) -> List[str]:
        """Get all feature names that will be created."""
        names = [c for c in FEATURE_COLUMNS if c in data.columns]
        names.append("Cluster_Label")
        
        if self.create_aggregate_features:
            names.extend(["SCL90_Mean", "SCL90_Max", "SCL90_Std"])
        
        if self.create_interaction_features:
            names.append("EPQ_Neuroticism_x_Anxiety")
        
        return names
    
    def get_feature_statistics(
        self,
        data: pd.DataFrame
    ) -> Dict[str, FeatureStats]:
        """
        Calculate descriptive statistics for all features.
        
        Args:
            data: Input DataFrame.
        
        Returns:
            Dictionary mapping feature names to FeatureStats.
        """
        stats = {}
        
        for col in FEATURE_COLUMNS:
            if col in data.columns:
                series = data[col]
                stats[col] = FeatureStats(
                    mean=float(series.mean()),
                    std=float(series.std()),
                    min=float(series.min()),
                    max=float(series.max()),
                    median=float(series.median()),
                    q1=float(series.quantile(0.25)),
                    q3=float(series.quantile(0.75))
                )
        
        return stats
    
    def create_binary_target(
        self,
        failed_subjects: pd.Series,
        threshold: int = 0
    ) -> pd.Series:
        """
        Create binary risk target from number of failed subjects.
        
        Args:
            failed_subjects: Series with number of failed subjects.
            threshold: Minimum failed subjects to be considered at risk.
        
        Returns:
            Binary series (1 = at risk, 0 = not at risk).
        """
        return (failed_subjects > threshold).astype(int)
    
    def save(self, filepath: str) -> None:
        """
        Save fitted feature engineer to file.
        
        Args:
            filepath: Path to save the model.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted FeatureEngineer")
        
        artifacts = {
            "cluster_generator": {
                "scaler": self.cluster_generator.scaler,
                "kmeans": self.cluster_generator.kmeans,
                "n_clusters": self.cluster_generator.n_clusters,
                "scaling_features": self.cluster_generator.scaling_features,
                "cluster_centers": self.cluster_generator._cluster_centers,
                "cluster_descriptions": self.cluster_generator._cluster_descriptions,
            },
            "scaler": self.scaler,
            "n_clusters": self.n_clusters,
            "create_interaction_features": self.create_interaction_features,
            "create_aggregate_features": self.create_aggregate_features,
            "feature_names": self._feature_names,
        }
        joblib.dump(artifacts, filepath)
    
    def load(self, filepath: str) -> "FeatureEngineer":
        """
        Load fitted feature engineer from file.
        
        Args:
            filepath: Path to the saved model.
        
        Returns:
            Self for method chaining.
        """
        artifacts = joblib.load(filepath)
        
        # Restore cluster generator
        cg_artifacts = artifacts["cluster_generator"]
        self.cluster_generator.scaler = cg_artifacts["scaler"]
        self.cluster_generator.kmeans = cg_artifacts["kmeans"]
        self.cluster_generator.n_clusters = cg_artifacts["n_clusters"]
        self.cluster_generator.scaling_features = cg_artifacts["scaling_features"]
        self.cluster_generator._cluster_centers = cg_artifacts["cluster_centers"]
        self.cluster_generator._cluster_descriptions = cg_artifacts["cluster_descriptions"]
        self.cluster_generator._is_fitted = True
        
        # Restore main attributes
        self.scaler = artifacts["scaler"]
        self.n_clusters = artifacts["n_clusters"]
        self.create_interaction_features = artifacts["create_interaction_features"]
        self.create_aggregate_features = artifacts["create_aggregate_features"]
        self._feature_names = artifacts["feature_names"]
        self._is_fitted = True
        
        return self


def prepare_features_for_prediction(
    epq_scores: Dict[str, float],
    scl90_scores: Dict[str, float],
    feature_engineer: FeatureEngineer
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Convenience function to prepare input data for model prediction.
    
    Args:
        epq_scores: Dictionary with EPQ factor scores.
        scl90_scores: Dictionary with SCL-90 factor scores.
        feature_engineer: Fitted FeatureEngineer instance.
    
    Returns:
        Tuple of (raw DataFrame, feature matrix for model).
    """
    # Build input record
    record = {}
    
    # Map EPQ scores
    epq_mapping = {
        "E": "内外向E", "extraversion": "内外向E", "extraversion_e": "内外向E",
        "N": "神经质N", "neuroticism": "神经质N", "neuroticism_n": "神经质N",
        "P": "精神质P", "psychoticism": "精神质P", "psychoticism_p": "精神质P",
        "L": "掩饰性L", "lie": "掩饰性L", "lie_scale_l": "掩饰性L",
    }
    
    for key, value in epq_scores.items():
        mapped_key = epq_mapping.get(key.lower(), key)
        if mapped_key in FEATURE_COLUMNS or key in FEATURE_COLUMNS:
            record[mapped_key if mapped_key in FEATURE_COLUMNS else key] = value
    
    # Map SCL-90 scores
    scl90_mapping = {
        "somatization": "躯体化",
        "obsessive_compulsive": "强迫症状",
        "interpersonal_sensitivity": "人际关系敏感",
        "depression": "抑郁",
        "anxiety": "焦虑",
        "hostility": "敌对",
        "phobic_anxiety": "恐怖",
        "paranoid_ideation": "偏执",
        "psychoticism": "精神病性",
        "additional_items": "其他",
    }
    
    for key, value in scl90_scores.items():
        mapped_key = scl90_mapping.get(key.lower(), key)
        if mapped_key in FEATURE_COLUMNS or key in FEATURE_COLUMNS:
            record[mapped_key if mapped_key in FEATURE_COLUMNS else key] = value
    
    # Create DataFrame and transform
    df = pd.DataFrame([record])
    transformed_df = feature_engineer.transform(df, include_scaled=False)
    feature_matrix = feature_engineer.get_feature_matrix(transformed_df)
    
    return df, feature_matrix