import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from imblearn.over_sampling import SMOTE


class DataProcessor:
    def __init__(self, df, target_column):
        self.original_df = df.copy()
        self.df = df.copy()
        self.target_column = target_column  # Store target column
        self.text_cols = []
        self.cat_cols = []
        self.num_cols = []
        
    def process_data(self, balance_method="None"):
        self._detect_column_types()
        self._clean_data()
        self._handle_imbalance(method=balance_method)
        return self.df 
       
    def _detect_column_types(self):
        """Detect numerical, categorical, and text columns"""
        for col in self.df.columns:
            if col == self.target_column:
                continue
            dtype = self.df[col].dtype
            if dtype == 'object':
                if self.df[col].str.len().mean() > 30:
                    self.text_cols.append(col)
                else:
                    self.cat_cols.append(col)
            elif np.issubdtype(dtype, np.number):
                self.num_cols.append(col)
                
    def _clean_data(self):
        """Enhanced cleaning with proper target handling"""
        # Handle missing values
        self._handle_missing_values()
        
        # Process text data
        self._process_text_columns()
            
        # Encode categorical data
        self._encode_categorical_data()
        
        # Handle class imbalance
        if self.target_column:
            self._handle_imbalance()
            
    def _handle_missing_values(self):
        """Fixed inplace operations"""
        # Numerical columns
        for col in self.num_cols:
            if col == self.target_column:
                continue
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Categorical columns
        for col in self.cat_cols:
            if col == self.target_column:
                continue
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        # Text columns
        for col in self.text_cols:
            self.df[col] = self.df[col].fillna('')
        
        # Target column
        if self.target_column:
            if self.df[self.target_column].dtype == 'object':
                self.df[self.target_column] = self.df[self.target_column].fillna(
                    self.df[self.target_column].mode()[0]
                )
            else:
                self.df[self.target_column] = self.df[self.target_column].fillna(
                    self.df[self.target_column].median()
                )
    
    def _process_text_columns(self):
        """Improved text processing"""
        for col in self.text_cols:
            self.df[col] = self.df[col].apply(self._clean_text)
    
    def _encode_categorical_data(self):
        """Safer categorical encoding"""
        if self.cat_cols:
            self.df = pd.get_dummies(
                self.df, 
                columns=self.cat_cols, 
                drop_first=True,
                dtype=int
            )
    
    def get_descriptive_stats(self):
        """Get descriptive statistics of numerical columns"""
        if not self.num_cols:
            raise ValueError("No numerical columns to describe.")
        
        # Descriptive statistics for numerical columns
        return self.df[self.num_cols].describe()

    
    def _handle_imbalance(self, method="None"):
        """Handle class imbalance"""
        if method == "SMOTE" and self.target_column:
            self._apply_smote_balancing()
    
    def _apply_smote_balancing(self):
        """SMOTE application with error handling"""
        try:
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
            
            smote = SMOTE()
            X_res, y_res = smote.fit_resample(X, y)
            self.df = pd.concat([X_res, y_res], axis=1)
        except Exception as e:
            print(f"SMOTE failed: {str(e)}")
            raise
    
    def get_class_weights(self):
        """Improved class weight calculation"""
        if self.target_column and self.df[self.target_column].dtype == 'object':
            class_counts = self.df[self.target_column].value_counts()
            total = class_counts.sum()
            return {cls: total/(len(class_counts)*count) for cls, count in class_counts.items()}
        return None
    
    def _get_class_ratio(self):
        if self.target_column and self.df[self.target_column].nunique() == 2:
            class_counts = self.df[self.target_column].value_counts()
            return class_counts.min() / class_counts.max()
        return None  # For multi-class or regression
        
    def _clean_text(self, text):
        """More robust text cleaning"""
        try:
            text = str(text).lower()
            text = ''.join([char for char in text if char not in string.punctuation])
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            return ' '.join(tokens)
        except:
            return ''
    
    def get_data_info(self):
        """Enhanced data information"""
        return {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'column_types': {
                'numerical': self.num_cols,
                'categorical': self.cat_cols,
                'text': self.text_cols
            },
            'class_balance': self.get_class_weights() if self.target_column else None
        }
