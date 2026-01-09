"""
Data preprocessing module for Student Academic Risk Prediction System.

Handles data cleaning, validation, SCL-90 score calculation from raw questionnaire
responses, and standardization of input data formats.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# SCL-90 Question to Factor Mapping
# Each factor maps to specific question indices (1-based indexing from questionnaire)
SCL90_QUESTION_MAPPING: Dict[str, List[int]] = {
    "躯体化": [1, 4, 12, 27, 40, 42, 48, 49, 52, 53, 56, 58],
    "强迫症状": [3, 9, 10, 28, 38, 45, 46, 51, 55, 65],
    "人际关系敏感": [6, 21, 34, 36, 37, 41, 61, 69, 73],
    "抑郁": [5, 14, 15, 20, 22, 26, 29, 30, 31, 32, 54, 71, 79],
    "焦虑": [2, 17, 23, 33, 39, 57, 72, 78, 80, 86],
    "敌对": [11, 24, 63, 67, 74, 81],
    "恐怖": [13, 25, 47, 50, 70, 75, 82],
    "偏执": [8, 18, 43, 68, 76, 83],
    "精神病性": [7, 16, 35, 62, 77, 84, 85, 87, 88, 90],
    "其他": [19, 44, 59, 60, 64, 66, 89],
}

# Column name mappings for standardization
COLUMN_NAME_MAPPING: Dict[str, str] = {
    "学号": "student_id",
    "姓名": "name",
    "班级": "class",
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
    "挂科数目": "failed_subjects",
}

# Reverse mapping for output
COLUMN_NAME_REVERSE_MAPPING: Dict[str, str] = {v: k for k, v in COLUMN_NAME_MAPPING.items()}


@dataclass
class ValidationResult:
    """Container for data validation results."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_data: Optional[pd.DataFrame] = None


class SCL90Calculator:
    """
    Calculator for SCL-90 psychological assessment scores.
    
    Converts raw questionnaire responses (90 questions, 1-5 scale) into
    10 factor scores used for risk prediction.
    """
    
    VALID_RESPONSE_RANGE: Tuple[int, int] = (1, 5)
    NUM_QUESTIONS: int = 90
    
    def __init__(self):
        self.question_mapping = SCL90_QUESTION_MAPPING
    
    def calculate_factor_scores(
        self,
        responses: Union[List[int], np.ndarray, Dict[int, int]]
    ) -> Dict[str, float]:
        """
        Calculate all 10 SCL-90 factor scores from raw responses.
        
        Args:
            responses: Raw questionnaire responses. Can be:
                - List/array of 90 integers (1-5 scale)
                - Dict mapping question number (1-90) to response value
        
        Returns:
            Dictionary mapping factor names to calculated scores.
        
        Raises:
            ValueError: If responses are invalid or incomplete.
        """
        validated_responses = self._validate_and_normalize_responses(responses)
        
        factor_scores = {}
        for factor_name, question_indices in self.question_mapping.items():
            # Convert 1-based indices to 0-based for array access
            factor_responses = [validated_responses[idx - 1] for idx in question_indices]
            # Factor score = mean of all responses for that factor
            factor_scores[factor_name] = round(np.mean(factor_responses), 4)
        
        return factor_scores
    
    def calculate_single_factor(
        self,
        responses: Union[List[int], np.ndarray],
        factor_name: str
    ) -> float:
        """
        Calculate a single factor score.
        
        Args:
            responses: Raw questionnaire responses.
            factor_name: Name of the factor to calculate.
        
        Returns:
            Calculated factor score.
        
        Raises:
            ValueError: If factor name is invalid.
        """
        if factor_name not in self.question_mapping:
            valid_factors = list(self.question_mapping.keys())
            raise ValueError(
                f"Invalid factor name: {factor_name}. Valid factors: {valid_factors}"
            )
        
        validated_responses = self._validate_and_normalize_responses(responses)
        question_indices = self.question_mapping[factor_name]
        factor_responses = [validated_responses[idx - 1] for idx in question_indices]
        
        return round(np.mean(factor_responses), 4)
    
    def calculate_total_score(
        self,
        responses: Union[List[int], np.ndarray]
    ) -> float:
        """
        Calculate total SCL-90 score (mean of all 90 responses).
        
        Args:
            responses: Raw questionnaire responses.
        
        Returns:
            Total score as mean of all responses.
        """
        validated_responses = self._validate_and_normalize_responses(responses)
        return round(np.mean(validated_responses), 4)
    
    def calculate_positive_item_count(
        self,
        responses: Union[List[int], np.ndarray],
        threshold: int = 2
    ) -> int:
        """
        Count positive items (responses >= threshold).
        
        Args:
            responses: Raw questionnaire responses.
            threshold: Minimum value to be considered positive (default: 2).
        
        Returns:
            Count of positive items.
        """
        validated_responses = self._validate_and_normalize_responses(responses)
        return int(np.sum(np.array(validated_responses) >= threshold))
    
    def _validate_and_normalize_responses(
        self,
        responses: Union[List[int], np.ndarray, Dict[int, int]]
    ) -> List[int]:
        """
        Validate and normalize response input to a list format.
        
        Args:
            responses: Input responses in various formats.
        
        Returns:
            Normalized list of 90 integer responses.
        
        Raises:
            ValueError: If responses are invalid.
        """
        # Handle dictionary input
        if isinstance(responses, dict):
            if len(responses) != self.NUM_QUESTIONS:
                raise ValueError(
                    f"Expected {self.NUM_QUESTIONS} responses, got {len(responses)}"
                )
            # Convert dict to ordered list
            normalized = [responses.get(i, 0) for i in range(1, self.NUM_QUESTIONS + 1)]
        else:
            normalized = list(responses)
        
        # Validate length
        if len(normalized) != self.NUM_QUESTIONS:
            raise ValueError(
                f"Expected {self.NUM_QUESTIONS} responses, got {len(normalized)}"
            )
        
        # Validate value range
        min_val, max_val = self.VALID_RESPONSE_RANGE
        for idx, val in enumerate(normalized):
            if not isinstance(val, (int, float)) or not (min_val <= val <= max_val):
                raise ValueError(
                    f"Response at index {idx + 1} is invalid: {val}. "
                    f"Expected integer in range [{min_val}, {max_val}]"
                )
        
        return [int(v) for v in normalized]
    
    @staticmethod
    def get_factor_interpretation(factor_name: str, score: float) -> str:
        """
        Get clinical interpretation for a factor score.
        
        Args:
            factor_name: Name of the SCL-90 factor.
            score: Calculated factor score.
        
        Returns:
            Clinical interpretation string.
        """
        if score < 1.5:
            level = "normal"
        elif score < 2.0:
            level = "mild"
        elif score < 2.5:
            level = "moderate"
        elif score < 3.0:
            level = "moderately_severe"
        else:
            level = "severe"
        
        return level


class DataValidator:
    """
    Validator for input data integrity and format compliance.
    
    Ensures data meets requirements for the risk prediction pipeline.
    """
    
    # Required columns for prediction (using Chinese names from original data)
    REQUIRED_FEATURE_COLUMNS_CN: List[str] = [
        "内外向E", "神经质N", "精神质P", "掩饰性L",
        "躯体化", "强迫症状", "人际关系敏感", "抑郁",
        "焦虑", "敌对", "恐怖", "偏执", "精神病性", "其他"
    ]
    
    # Required columns for training (includes target)
    REQUIRED_TRAINING_COLUMNS_CN: List[str] = REQUIRED_FEATURE_COLUMNS_CN + ["挂科数目"]
    
    # Optional identifier columns
    IDENTIFIER_COLUMNS_CN: List[str] = ["学号", "姓名", "班级"]
    
    # Value constraints
    EPQ_SCORE_RANGE: Tuple[float, float] = (0.0, 100.0)
    SCL90_SCORE_RANGE: Tuple[float, float] = (1.0, 5.0)
    FAILED_SUBJECTS_RANGE: Tuple[int, int] = (0, 50)
    
    def validate_prediction_input(
        self,
        data: Union[pd.DataFrame, Dict]
    ) -> ValidationResult:
        """
        Validate data for prediction (features only, no target needed).
        
        Args:
            data: Input data as DataFrame or dictionary.
        
        Returns:
            ValidationResult with validation status and any errors.
        """
        errors = []
        warnings = []
        
        # Convert dict to DataFrame if necessary
        if isinstance(data, dict):
            try:
                data = pd.DataFrame([data])
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Failed to convert dict to DataFrame: {str(e)}"]
                )
        
        # Check required columns
        missing_cols = set(self.REQUIRED_FEATURE_COLUMNS_CN) - set(data.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if errors:
            return ValidationResult(is_valid=False, errors=errors)
        
        # Validate value ranges
        cleaned_data = data.copy()
        
        # EPQ factors validation
        epq_cols = ["内外向E", "神经质N", "精神质P", "掩饰性L"]
        for col in epq_cols:
            if col in cleaned_data.columns:
                out_of_range = cleaned_data[
                    (cleaned_data[col] < self.EPQ_SCORE_RANGE[0]) |
                    (cleaned_data[col] > self.EPQ_SCORE_RANGE[1])
                ]
                if len(out_of_range) > 0:
                    warnings.append(
                        f"Column '{col}' has {len(out_of_range)} values outside "
                        f"expected range {self.EPQ_SCORE_RANGE}"
                    )
        
        # SCL-90 factors validation
        scl90_cols = [
            "躯体化", "强迫症状", "人际关系敏感", "抑郁",
            "焦虑", "敌对", "恐怖", "偏执", "精神病性", "其他"
        ]
        for col in scl90_cols:
            if col in cleaned_data.columns:
                out_of_range = cleaned_data[
                    (cleaned_data[col] < self.SCL90_SCORE_RANGE[0]) |
                    (cleaned_data[col] > self.SCL90_SCORE_RANGE[1])
                ]
                if len(out_of_range) > 0:
                    warnings.append(
                        f"Column '{col}' has {len(out_of_range)} values outside "
                        f"expected range {self.SCL90_SCORE_RANGE}"
                    )
        
        # Check for missing values
        null_counts = cleaned_data[self.REQUIRED_FEATURE_COLUMNS_CN].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if len(cols_with_nulls) > 0:
            errors.append(
                f"Missing values found in columns: {dict(cols_with_nulls)}"
            )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_data if len(errors) == 0 else None
        )
    
    def validate_training_input(
        self,
        data: pd.DataFrame
    ) -> ValidationResult:
        """
        Validate data for model training (includes target column).
        
        Args:
            data: Training data DataFrame.
        
        Returns:
            ValidationResult with validation status and any errors.
        """
        errors = []
        warnings = []
        
        # First run prediction validation
        pred_result = self.validate_prediction_input(data)
        errors.extend(pred_result.errors)
        warnings.extend(pred_result.warnings)
        
        # Check target column
        if "挂科数目" not in data.columns:
            errors.append("Missing target column: 挂科数目")
        else:
            # Validate target values
            invalid_targets = data[
                (data["挂科数目"] < self.FAILED_SUBJECTS_RANGE[0]) |
                (data["挂科数目"] > self.FAILED_SUBJECTS_RANGE[1])
            ]
            if len(invalid_targets) > 0:
                warnings.append(
                    f"Target column has {len(invalid_targets)} values outside "
                    f"expected range {self.FAILED_SUBJECTS_RANGE}"
                )
            
            # Check for null targets
            null_targets = data["挂科数目"].isnull().sum()
            if null_targets > 0:
                errors.append(f"Target column has {null_targets} missing values")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=data.copy() if len(errors) == 0 else None
        )
    
    @staticmethod
    def validate_single_record(record: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single prediction record (for API input).
        
        Args:
            record: Dictionary containing feature values.
        
        Returns:
            Tuple of (is_valid, error_messages).
        """
        validator = DataValidator()
        result = validator.validate_prediction_input(record)
        return result.is_valid, result.errors


class DataPreprocessor:
    """
    Main preprocessor class for data cleaning and transformation.
    
    Handles column renaming, missing value imputation, outlier handling,
    and format standardization for the risk prediction pipeline.
    """
    
    def __init__(
        self,
        use_english_columns: bool = False,
        handle_missing: str = "mean",
        clip_outliers: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            use_english_columns: Whether to convert column names to English.
            handle_missing: Strategy for missing values ('mean', 'median', 'drop').
            clip_outliers: Whether to clip outliers to expected ranges.
        """
        self.use_english_columns = use_english_columns
        self.handle_missing = handle_missing
        self.clip_outliers = clip_outliers
        self.validator = DataValidator()
        self.scl90_calculator = SCL90Calculator()
        
        # Store fitted statistics for imputation
        self._fitted_stats: Dict[str, float] = {}
    
    def fit(self, data: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit preprocessor on training data to learn imputation statistics.
        
        Args:
            data: Training DataFrame.
        
        Returns:
            Self for method chaining.
        """
        feature_cols = self.validator.REQUIRED_FEATURE_COLUMNS_CN
        
        for col in feature_cols:
            if col in data.columns:
                if self.handle_missing == "mean":
                    self._fitted_stats[col] = data[col].mean()
                elif self.handle_missing == "median":
                    self._fitted_stats[col] = data[col].median()
        
        return self
    
    def transform(
        self,
        data: pd.DataFrame,
        include_target: bool = False
    ) -> pd.DataFrame:
        """
        Transform data using fitted statistics.
        
        Args:
            data: Input DataFrame.
            include_target: Whether to include target column in output.
        
        Returns:
            Transformed DataFrame.
        """
        df = data.copy()
        
        # Handle missing values
        if self.handle_missing == "drop":
            df = df.dropna()
        else:
            for col, stat in self._fitted_stats.items():
                if col in df.columns:
                    df[col] = df[col].fillna(stat)
        
        # Clip outliers if enabled
        if self.clip_outliers:
            df = self._clip_outliers(df)
        
        # Convert column names if requested
        if self.use_english_columns:
            df = df.rename(columns=COLUMN_NAME_MAPPING)
        
        # Select output columns
        if include_target:
            output_cols = self.validator.REQUIRED_TRAINING_COLUMNS_CN
        else:
            output_cols = self.validator.REQUIRED_FEATURE_COLUMNS_CN
        
        if self.use_english_columns:
            output_cols = [COLUMN_NAME_MAPPING.get(c, c) for c in output_cols]
        
        # Only select columns that exist
        available_cols = [c for c in output_cols if c in df.columns]
        
        return df[available_cols]
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        include_target: bool = False
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            data: Input DataFrame.
            include_target: Whether to include target column.
        
        Returns:
            Transformed DataFrame.
        """
        return self.fit(data).transform(data, include_target)
    
    def preprocess_raw_scl90(
        self,
        responses: Union[List[int], np.ndarray, Dict[int, int]]
    ) -> Dict[str, float]:
        """
        Convert raw SCL-90 responses to factor scores.
        
        Args:
            responses: Raw questionnaire responses (90 items, 1-5 scale).
        
        Returns:
            Dictionary of factor scores.
        """
        return self.scl90_calculator.calculate_factor_scores(responses)
    
    def prepare_batch_data(
        self,
        file_path: str,
        file_type: str = "csv"
    ) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        Load and prepare batch data from file.
        
        Args:
            file_path: Path to input file.
            file_type: Type of file ('csv', 'excel').
        
        Returns:
            Tuple of (prepared DataFrame, validation result).
        """
        # Load data
        if file_type == "csv":
            df = pd.read_csv(file_path)
        elif file_type == "excel":
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Validate
        validation_result = self.validator.validate_prediction_input(df)
        
        if validation_result.is_valid:
            processed_df = self.transform(df, include_target=False)
            validation_result.cleaned_data = processed_df
        
        return df, validation_result
    
    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip values to expected ranges.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            DataFrame with clipped values.
        """
        result = df.copy()
        
        # EPQ factors: 0-100
        epq_cols = ["内外向E", "神经质N", "精神质P", "掩饰性L"]
        for col in epq_cols:
            if col in result.columns:
                result[col] = result[col].clip(0, 100)
        
        # SCL-90 factors: 1-5
        scl90_cols = [
            "躯体化", "强迫症状", "人际关系敏感", "抑郁",
            "焦虑", "敌对", "恐怖", "偏执", "精神病性", "其他"
        ]
        for col in scl90_cols:
            if col in result.columns:
                result[col] = result[col].clip(1, 5)
        
        # Target column: 0-50
        if "挂科数目" in result.columns:
            result["挂科数目"] = result["挂科数目"].clip(0, 50)
        
        return result
    
    @staticmethod
    def create_prediction_input(
        epq_scores: Dict[str, float],
        scl90_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Create a properly formatted prediction input from scores.
        
        Args:
            epq_scores: EPQ factor scores (E, N, P, L).
            scl90_scores: SCL-90 factor scores (10 factors).
        
        Returns:
            DataFrame ready for prediction.
        """
        # Map English keys to Chinese if needed
        epq_mapping = {
            "E": "内外向E", "extraversion": "内外向E",
            "N": "神经质N", "neuroticism": "神经质N",
            "P": "精神质P", "psychoticism": "精神质P",
            "L": "掩饰性L", "lie": "掩饰性L"
        }
        
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
            "additional_items": "其他"
        }
        
        record = {}
        
        # Process EPQ scores
        for key, value in epq_scores.items():
            mapped_key = epq_mapping.get(key.lower(), key)
            record[mapped_key] = value
        
        # Process SCL-90 scores
        for key, value in scl90_scores.items():
            mapped_key = scl90_mapping.get(key.lower(), key)
            record[mapped_key] = value
        
        return pd.DataFrame([record])