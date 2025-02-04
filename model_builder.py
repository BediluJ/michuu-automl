import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.inspection import permutation_importance
import warnings

class ModelBuilder:
    def __init__(self, df, target, use_cv=False, cv_folds=5, 
                 tune_params=False, tune_method='GridSearch'):
        self.df = df
        self.target = target
        self.X = df.drop(columns=[target])
        self.y = df[target]
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.tune_params = tune_params
        self.tune_method = tune_method
        self.problem_type = self._detect_problem_type()
        self.class_ratio = self._get_class_ratio() if self.problem_type == 'classification' else None
        self.models = []
        self.results = []
        self.best_model = None
        self._split_data()

    def _detect_problem_type(self):
        if pd.api.types.is_numeric_dtype(self.y):
            unique_values = self.y.nunique()
            if unique_values / len(self.y) < 0.05 and unique_values > 1:
                return 'classification'
            return 'regression' if unique_values > 10 else 'classification'
        return 'classification'

    def _get_class_ratio(self):
        class_counts = self.y.value_counts()
        if len(class_counts) < 2:
            return None
        return class_counts.min() / class_counts.max()

    def _split_data(self):
        stratify = self.y if self.problem_type == 'classification' else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=stratify
        )

    def compare_models(self):
        model_configs = self._get_model_configs()
        
        for name, model, params in model_configs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    if self.tune_params:
                        model = self._hyperparameter_tune(model, params)
                    
                    scores = self._evaluate_model(model)
                    self.results.append({'Model': name, **scores})
                    self.models.append(model)
                    
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
        
        self._select_best_model()
        return pd.DataFrame(self.results)

    def _get_model_configs(self):
        common_reg_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10]
        }
        
        if self.problem_type == 'classification':
            return [
                ('Logistic Regression', 
                 LogisticRegression(max_iter=1000, class_weight='balanced'),
                 {'C': [0.1, 1, 10]}),
                
                ('Random Forest', 
                 RandomForestClassifier(class_weight='balanced'),
                 {**common_reg_params, 'max_features': ['sqrt', 'log2']}),
                
                ('XGBoost', 
                 XGBClassifier(scale_pos_weight=self.class_ratio, eval_metric='logloss'),
                 {'learning_rate': [0.01, 0.1, 0.3], **common_reg_params}),
                
                ('SVM', 
                 SVC(probability=True, class_weight='balanced'),
                 {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
                
                ('Gradient Boosting',
                 GradientBoostingClassifier(),
                 {'learning_rate': [0.01, 0.1], **common_reg_params})
            ]
        else:
            return [
                ('Linear Regression', 
                 LinearRegression(),
                 {'fit_intercept': [True, False]}),
                
                ('Random Forest', 
                 RandomForestRegressor(),
                 {**common_reg_params, 'max_features': [0.5, 'sqrt']}),
                
                ('XGBoost', 
                 XGBRegressor(),
                 {'learning_rate': [0.01, 0.1, 0.3], **common_reg_params}),
                
                ('SVM', 
                 SVR(),
                 {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
                
                ('Gradient Boosting',
                 GradientBoostingRegressor(),
                 {'learning_rate': [0.01, 0.1], **common_reg_params})
            ]

    def _hyperparameter_tune(self, model, param_grid):
        search_class = GridSearchCV if self.tune_method == 'GridSearch' else RandomizedSearchCV
        search = search_class(
            model,
            param_grid,
            cv=self.cv_folds,
            n_jobs=-1,
            scoring=self._get_scoring_metric()
        )
        search.fit(self.X_train, self.y_train)
        return search.best_estimator_

    def _evaluate_model(self, model):
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        probas = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
        
        metrics = self._get_metrics(model, preds, probas)
        
        if self.use_cv:
            cv_scores = cross_val_score(
                model, self.X, self.y, cv=self.cv_folds,
                scoring=self._get_scoring_metric()
            )
            metrics['CV_Score'] = f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            
        return metrics

    def _get_scoring_metric(self):
        return 'roc_auc' if self.problem_type == 'classification' else 'r2'

    def _get_metrics(self, model, preds, probas):
        metrics = {}
        if self.problem_type == 'classification':
            metrics.update({
                'Accuracy': accuracy_score(self.y_test, preds),
                'Balanced_Accuracy': balanced_accuracy_score(self.y_test, preds),
                'F1_Score': f1_score(self.y_test, preds, average='weighted', zero_division=0)
            })
            
            if probas is not None:
                try:
                    if len(np.unique(self.y)) > 2:
                        metrics['AUC_ROC'] = roc_auc_score(self.y_test, probas, multi_class='ovr')
                    else:
                        metrics['AUC_ROC'] = roc_auc_score(self.y_test, probas[:, 1])
                except Exception as e:
                    metrics['AUC_ROC'] = np.nan
        else:
            metrics.update({
                'R2': r2_score(self.y_test, preds),
                'MAE': mean_absolute_error(self.y_test, preds),
                'RMSE': np.sqrt(mean_squared_error(self.y_test, preds))
            })
        return metrics

    def _select_best_model(self):
        metric = 'AUC_ROC' if self.problem_type == 'classification' else 'R2'
        valid_results = pd.DataFrame(self.results).dropna(subset=[metric])
        
        if valid_results.empty:
            raise ValueError("No valid models with complete metrics")
            
        self.best_model = self.models[valid_results[metric].idxmax()]
        return self.best_model

    def get_feature_importance(self, n_features=15):
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        else:
            result = permutation_importance(
                self.best_model, self.X_test, self.y_test,
                n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = result.importances_mean
            
        return pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(n_features)

    def get_champion_model(self):
        """Correctly identify the best model based on metrics"""
        metric = 'AUC_ROC' if self.problem_type == 'classification' else 'R2'
        
        # Convert results to DataFrame for easy manipulation
        results_df = pd.DataFrame(self.results)
        
        # Ensure there are valid results
        if results_df.empty or metric not in results_df.columns:
            raise ValueError("No valid models or metrics available")
        
        # Find the index of the best model
        best_idx = results_df[metric].idxmax()
        
        # Verify model list alignment
        if best_idx >= len(self.models):
            raise IndexError("Model index out of range")
        
        return {
            'Model': self.models[best_idx].__class__.__name__,
            'Score': results_df.loc[best_idx, metric],
            'All Metrics': results_df.loc[best_idx].to_dict(),
            'Model Object': self.models[best_idx]
        }