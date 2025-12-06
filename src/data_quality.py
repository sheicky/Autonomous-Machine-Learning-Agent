import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class DataQualityAnalyzer:
    """Comprehensive data quality analysis and recommendations."""
    
    def __init__(self, df, target_column=None):
        self.df = df
        self.target_column = target_column
        self.issues = []
        self.recommendations = []
        
    def analyze(self):
        """Run all quality checks."""
        self.check_class_imbalance()
        self.check_missing_patterns()
        self.check_outliers()
        self.check_data_leakage()
        self.check_sample_size()
        self.check_cardinality()
        self.check_constant_features()
        self.check_duplicate_features()
        
        return {
            "issues": self.issues,
            "recommendations": self.recommendations,
            "severity_score": self._calculate_severity()
        }
    
    def check_class_imbalance(self):
        """Detect class imbalance in target variable."""
        if self.target_column and self.target_column in self.df.columns:
            value_counts = self.df[self.target_column].value_counts()
            if len(value_counts) > 1:
                ratio = value_counts.max() / value_counts.min()
                if ratio > 3:
                    self.issues.append({
                        "type": "class_imbalance",
                        "severity": "high" if ratio > 10 else "medium",
                        "message": f"Class imbalance detected: {ratio:.1f}:1 ratio",
                        "details": value_counts.to_dict()
                    })
                    self.recommendations.append({
                        "issue": "class_imbalance",
                        "action": "Use SMOTE, class_weight='balanced', or stratified sampling",
                        "code_hint": "from imblearn.over_sampling import SMOTE"
                    })
    
    def check_missing_patterns(self):
        """Analyze missing value patterns."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            self.issues.append({
                "type": "high_missing",
                "severity": "high",
                "message": f"{len(high_missing)} columns have >50% missing values",
                "details": high_missing.to_dict()
            })
            self.recommendations.append({
                "issue": "high_missing",
                "action": "Consider dropping these columns or using advanced imputation",
                "columns": high_missing.index.tolist()
            })
        
        # Check for MCAR vs MAR patterns
        if missing.sum() > 0:
            # Simple heuristic: if missing values correlate with target, it's MAR
            if self.target_column and self.target_column in self.df.columns:
                for col in missing[missing > 0].index:
                    if col != self.target_column:
                        missing_indicator = self.df[col].isnull().astype(int)
                        if len(self.df[self.target_column].unique()) < 20:  # Categorical target
                            # Chi-square test
                            from scipy.stats import chi2_contingency
                            contingency = pd.crosstab(missing_indicator, self.df[self.target_column])
                            chi2, p_value, _, _ = chi2_contingency(contingency)
                            if p_value < 0.05:
                                self.recommendations.append({
                                    "issue": "missing_not_random",
                                    "action": f"Missing values in '{col}' correlate with target - use indicator features",
                                    "column": col
                                })
    
    def check_outliers(self):
        """Detect outliers using IQR method."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 3 * IQR)) | (self.df[col] > (Q3 + 3 * IQR))).sum()
            
            if outliers > 0:
                outlier_pct = (outliers / len(self.df)) * 100
                if outlier_pct > 5:
                    outlier_summary[col] = {"count": int(outliers), "percentage": round(outlier_pct, 2)}
        
        if outlier_summary:
            self.issues.append({
                "type": "outliers",
                "severity": "medium",
                "message": f"{len(outlier_summary)} columns have significant outliers",
                "details": outlier_summary
            })
            self.recommendations.append({
                "issue": "outliers",
                "action": "Consider RobustScaler, winsorization, or tree-based models",
                "affected_columns": list(outlier_summary.keys())
            })
    
    def check_data_leakage(self):
        """Check for potential data leakage."""
        if not self.target_column or self.target_column not in self.df.columns:
            return
        
        # Check for perfect correlations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.target_column]
        
        if len(numeric_cols) > 0 and self.df[self.target_column].dtype in [np.int64, np.float64]:
            correlations = self.df[numeric_cols].corrwith(self.df[self.target_column]).abs()
            high_corr = correlations[correlations > 0.95]
            
            if len(high_corr) > 0:
                self.issues.append({
                    "type": "potential_leakage",
                    "severity": "critical",
                    "message": f"{len(high_corr)} features have suspiciously high correlation with target",
                    "details": high_corr.to_dict()
                })
                self.recommendations.append({
                    "issue": "potential_leakage",
                    "action": "Investigate these features - they may contain future information",
                    "columns": high_corr.index.tolist()
                })
    
    def check_sample_size(self):
        """Validate sample size adequacy."""
        n_samples = len(self.df)
        n_features = len(self.df.columns) - (1 if self.target_column else 0)
        
        # Rule of thumb: need at least 10 samples per feature
        min_recommended = n_features * 10
        
        if n_samples < min_recommended:
            self.issues.append({
                "type": "small_sample",
                "severity": "high",
                "message": f"Sample size ({n_samples}) may be too small for {n_features} features",
                "details": {"samples": n_samples, "features": n_features, "recommended": min_recommended}
            })
            self.recommendations.append({
                "issue": "small_sample",
                "action": "Use regularization, feature selection, or collect more data",
                "suggestion": "Consider L1/L2 regularization or dimensionality reduction"
            })
    
    def check_cardinality(self):
        """Check for high-cardinality categorical features."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = {}
        
        for col in categorical_cols:
            n_unique = self.df[col].nunique()
            if n_unique > 50:
                high_cardinality[col] = n_unique
        
        if high_cardinality:
            self.issues.append({
                "type": "high_cardinality",
                "severity": "medium",
                "message": f"{len(high_cardinality)} categorical columns have >50 unique values",
                "details": high_cardinality
            })
            self.recommendations.append({
                "issue": "high_cardinality",
                "action": "Use target encoding, frequency encoding, or embedding layers",
                "columns": list(high_cardinality.keys())
            })
    
    def check_constant_features(self):
        """Identify constant or near-constant features."""
        constant_cols = []
        for col in self.df.columns:
            if col != self.target_column:
                if self.df[col].nunique() == 1:
                    constant_cols.append(col)
                elif self.df[col].dtype in [np.int64, np.float64]:
                    # Near-constant: >95% same value
                    value_counts = self.df[col].value_counts()
                    if len(value_counts) > 0 and (value_counts.iloc[0] / len(self.df)) > 0.95:
                        constant_cols.append(col)
        
        if constant_cols:
            self.issues.append({
                "type": "constant_features",
                "severity": "low",
                "message": f"{len(constant_cols)} features are constant or near-constant",
                "details": constant_cols
            })
            self.recommendations.append({
                "issue": "constant_features",
                "action": "Drop these features - they provide no information",
                "columns": constant_cols
            })
    
    def check_duplicate_features(self):
        """Find duplicate or highly correlated features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return
        
        corr_matrix = self.df[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        duplicates = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.98)]
        
        if duplicates:
            self.issues.append({
                "type": "duplicate_features",
                "severity": "low",
                "message": f"{len(duplicates)} features are near-duplicates of others",
                "details": duplicates
            })
            self.recommendations.append({
                "issue": "duplicate_features",
                "action": "Remove redundant features to reduce multicollinearity",
                "columns": duplicates
            })
    
    def _calculate_severity(self):
        """Calculate overall severity score (0-100)."""
        severity_weights = {"critical": 30, "high": 20, "medium": 10, "low": 5}
        score = sum(severity_weights.get(issue["severity"], 0) for issue in self.issues)
        return min(score, 100)
    
    def get_report(self):
        """Generate a formatted report."""
        analysis = self.analyze()
        
        report = {
            "summary": {
                "total_issues": len(analysis["issues"]),
                "severity_score": analysis["severity_score"],
                "status": self._get_status(analysis["severity_score"])
            },
            "issues": analysis["issues"],
            "recommendations": analysis["recommendations"]
        }
        
        return report
    
    def _get_status(self, score):
        """Get status based on severity score."""
        if score >= 50:
            return "critical"
        elif score >= 30:
            return "warning"
        elif score >= 10:
            return "caution"
        else:
            return "good"
