"""
Test suite for enhanced AutoML features
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_quality import DataQualityAnalyzer
from src.experiment_tracker import ExperimentTracker
from src.cache_manager import CacheManager


class TestDataQualityAnalyzer:
    """Test data quality analysis features."""
    
    def test_class_imbalance_detection(self):
        """Test detection of class imbalance."""
        # Create imbalanced dataset
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': [0] * 90 + [1] * 10  # 9:1 ratio
        })
        
        analyzer = DataQualityAnalyzer(df, target_column='target')
        report = analyzer.get_report()
        
        # Should detect imbalance
        issues = [i for i in report['issues'] if i['type'] == 'class_imbalance']
        assert len(issues) > 0
        assert issues[0]['severity'] in ['high', 'medium']
    
    def test_missing_value_detection(self):
        """Test detection of missing values."""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [np.nan] * 3 + [4, 5],  # 60% missing
            'target': [0, 1, 0, 1, 0]
        })
        
        analyzer = DataQualityAnalyzer(df, target_column='target')
        report = analyzer.get_report()
        
        # Should detect high missing values
        issues = [i for i in report['issues'] if i['type'] == 'high_missing']
        assert len(issues) > 0
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        # Create dataset with outliers
        normal_data = np.random.normal(0, 1, 95)
        outliers = np.array([10, -10, 15, -15, 20])
        
        df = pd.DataFrame({
            'feature1': np.concatenate([normal_data, outliers]),
            'target': [0] * 50 + [1] * 50
        })
        
        analyzer = DataQualityAnalyzer(df, target_column='target')
        report = analyzer.get_report()
        
        # Should detect outliers
        issues = [i for i in report['issues'] if i['type'] == 'outliers']
        assert len(issues) > 0
    
    def test_constant_features(self):
        """Test detection of constant features."""
        df = pd.DataFrame({
            'constant_col': [1] * 100,
            'near_constant': [1] * 96 + [2] * 4,  # 96% same value
            'normal_col': np.random.rand(100),
            'target': [0] * 50 + [1] * 50
        })
        
        analyzer = DataQualityAnalyzer(df, target_column='target')
        report = analyzer.get_report()
        
        # Should detect constant features
        issues = [i for i in report['issues'] if i['type'] == 'constant_features']
        assert len(issues) > 0
        assert 'constant_col' in issues[0]['details']
    
    def test_severity_score(self):
        """Test severity score calculation."""
        # Perfect dataset
        df_good = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': [0] * 50 + [1] * 50
        })
        
        analyzer_good = DataQualityAnalyzer(df_good, target_column='target')
        report_good = analyzer_good.get_report()
        
        # Bad dataset
        df_bad = pd.DataFrame({
            'feature1': [1] * 100,  # Constant
            'feature2': [np.nan] * 60 + list(range(40)),  # 60% missing
            'target': [0] * 95 + [1] * 5  # Imbalanced
        })
        
        analyzer_bad = DataQualityAnalyzer(df_bad, target_column='target')
        report_bad = analyzer_bad.get_report()
        
        # Bad dataset should have higher severity
        assert report_bad['summary']['severity_score'] > report_good['summary']['severity_score']


class TestExperimentTracker:
    """Test experiment tracking features."""
    
    def test_start_experiment(self, tmp_path):
        """Test starting a new experiment."""
        tracker = ExperimentTracker(experiments_dir=str(tmp_path))
        
        exp_id = tracker.start_experiment(
            name="test_experiment",
            dataset_info={"rows": 100, "columns": 5}
        )
        
        assert exp_id is not None
        assert tracker.current_experiment is not None
        assert tracker.current_experiment['name'] == "test_experiment"
    
    def test_log_stage(self, tmp_path):
        """Test logging pipeline stages."""
        tracker = ExperimentTracker(experiments_dir=str(tmp_path))
        tracker.start_experiment(name="test")
        
        tracker.log_stage("preprocessing", {"status": "complete"})
        
        assert "preprocessing" in tracker.current_experiment['stages']
        assert tracker.current_experiment['stages']['preprocessing']['data']['status'] == "complete"
    
    def test_log_model(self, tmp_path):
        """Test logging model results."""
        tracker = ExperimentTracker(experiments_dir=str(tmp_path))
        tracker.start_experiment(name="test")
        
        tracker.log_model(
            "Random Forest",
            {"accuracy": 0.85, "f1": 0.83},
            {"n_estimators": 100, "max_depth": 5}
        )
        
        assert len(tracker.current_experiment['models']) == 1
        assert tracker.current_experiment['models'][0]['name'] == "Random Forest"
        assert tracker.current_experiment['models'][0]['metrics']['accuracy'] == 0.85
    
    def test_end_experiment(self, tmp_path):
        """Test ending and saving experiment."""
        tracker = ExperimentTracker(experiments_dir=str(tmp_path))
        exp_id = tracker.start_experiment(name="test")
        
        tracker.log_model("Model1", {"accuracy": 0.8}, {})
        result = tracker.end_experiment()
        
        assert result is not None
        assert tracker.current_experiment is None
        
        # Check file was saved
        exp_file = tmp_path / f"{exp_id}.json"
        assert exp_file.exists()
    
    def test_list_experiments(self, tmp_path):
        """Test listing all experiments."""
        tracker = ExperimentTracker(experiments_dir=str(tmp_path))
        
        # Create multiple experiments
        for i in range(3):
            tracker.start_experiment(name=f"exp_{i}")
            tracker.log_model(f"Model_{i}", {"accuracy": 0.8 + i * 0.05}, {})
            tracker.end_experiment()
        
        experiments = tracker.list_experiments()
        assert len(experiments) == 3
    
    def test_get_leaderboard(self, tmp_path):
        """Test global leaderboard."""
        tracker = ExperimentTracker(experiments_dir=str(tmp_path))
        
        # Create experiments with different models
        for i in range(2):
            tracker.start_experiment(name=f"exp_{i}")
            tracker.log_model(f"Model_{i}", {"accuracy": 0.7 + i * 0.1, "f1": 0.65 + i * 0.1}, {})
            tracker.end_experiment()
        
        leaderboard = tracker.get_leaderboard(metric='accuracy')
        
        assert len(leaderboard) == 2
        assert leaderboard.iloc[0]['accuracy'] > leaderboard.iloc[1]['accuracy']


class TestCacheManager:
    """Test caching functionality."""
    
    def test_cache_set_and_get(self, tmp_path):
        """Test basic cache operations."""
        cache = CacheManager(cache_dir=str(tmp_path))
        
        key = cache._generate_key(pd.DataFrame({'a': [1, 2, 3]}))
        data = {"result": "test_data"}
        
        # Set cache
        success = cache.set(key, data, operation_type="preprocessing")
        assert success
        
        # Get cache
        retrieved = cache.get(key, operation_type="preprocessing")
        assert retrieved == data
    
    def test_cache_has(self, tmp_path):
        """Test cache existence check."""
        cache = CacheManager(cache_dir=str(tmp_path))
        
        key = cache._generate_key(pd.DataFrame({'a': [1, 2, 3]}))
        
        # Should not exist initially
        assert not cache.has(key, operation_type="preprocessing")
        
        # Set cache
        cache.set(key, {"data": "test"}, operation_type="preprocessing")
        
        # Should exist now
        assert cache.has(key, operation_type="preprocessing")
    
    def test_cache_invalidate(self, tmp_path):
        """Test cache invalidation."""
        cache = CacheManager(cache_dir=str(tmp_path))
        
        key = cache._generate_key(pd.DataFrame({'a': [1, 2, 3]}))
        cache.set(key, {"data": "test"}, operation_type="preprocessing")
        
        # Invalidate
        cache.invalidate(key, operation_type="preprocessing")
        
        # Should not exist
        assert not cache.has(key, operation_type="preprocessing")
    
    def test_cache_stats(self, tmp_path):
        """Test cache statistics."""
        cache = CacheManager(cache_dir=str(tmp_path))
        
        # Add multiple cache entries
        for i in range(3):
            key = cache._generate_key(pd.DataFrame({'a': [i]}))
            cache.set(key, {"data": f"test_{i}"}, operation_type="preprocessing")
        
        stats = cache.get_stats()
        
        assert stats['total_entries'] == 3
        assert 'preprocessing' in stats['by_operation']
        assert stats['by_operation']['preprocessing']['count'] == 3
    
    def test_cache_key_generation(self, tmp_path):
        """Test cache key generation consistency."""
        cache = CacheManager(cache_dir=str(tmp_path))
        
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df3 = pd.DataFrame({'a': [1, 2, 4], 'b': [4, 5, 6]})
        
        key1 = cache._generate_key(df1)
        key2 = cache._generate_key(df2)
        key3 = cache._generate_key(df3)
        
        # Same data should generate same key
        assert key1 == key2
        
        # Different data should generate different key
        assert key1 != key3


class TestIntegration:
    """Integration tests for combined features."""
    
    def test_full_pipeline_with_tracking(self, tmp_path):
        """Test full pipeline with experiment tracking."""
        # Create sample dataset
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': [0] * 50 + [1] * 50
        })
        
        # Initialize components
        tracker = ExperimentTracker(experiments_dir=str(tmp_path))
        analyzer = DataQualityAnalyzer(df, target_column='target')
        
        # Start experiment
        exp_id = tracker.start_experiment(
            name="integration_test",
            dataset_info={"rows": len(df), "columns": len(df.columns)}
        )
        
        # Run quality analysis
        quality_report = analyzer.get_report()
        tracker.log_stage("quality_analysis", quality_report)
        
        # Simulate model training
        tracker.log_model(
            "Test Model",
            {"accuracy": 0.85, "f1": 0.83},
            {"param1": "value1"}
        )
        
        # End experiment
        result = tracker.end_experiment()
        
        # Verify
        assert result is not None
        assert 'quality_analysis' in result['stages']
        assert len(result['models']) == 1
        
        # Verify file saved
        exp_file = tmp_path / f"{exp_id}.json"
        assert exp_file.exists()


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
