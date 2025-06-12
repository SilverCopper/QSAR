"""
机器学习模型工具模块
支持多种机器学习算法：LightGBM、RandomForest、XGBoost、SVM等
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error
)
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
import streamlit as st
from datetime import datetime

class ModelTrainer:
    """机器学习模型训练器"""
    
    def __init__(self, task_type='classification'):
        """
        初始化模型训练器
        
        Args:
            task_type: 任务类型 ('classification' 或 'regression')
        """
        self.task_type = task_type
        self.model = None
        self.model_name = None
        self.feature_scaler = None
        self.training_metadata = None
        
    def get_available_models(self):
        """获取可用的模型列表"""
        if self.task_type == 'classification':
            return {
                'LightGBM': lgb.LGBMClassifier,
                'XGBoost': xgb.XGBClassifier,
                'Random Forest': RandomForestClassifier,
                'SVM': SVC
            }
        else:
            return {
                'LightGBM': lgb.LGBMRegressor,
                'XGBoost': xgb.XGBRegressor,
                'Random Forest': RandomForestRegressor,
                'SVM': SVR
            }
    
    def get_default_params(self, model_name):
        """获取模型的默认参数"""
        params = {
            'LightGBM': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'XGBoost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'Random Forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'SVM': {
                'C': 1.0,
                'kernel': 'rbf',
                'random_state': 42
            }
        }
        return params.get(model_name, {})
    
    def train_model(self, X, y, model_name, params=None, test_size=0.2):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
            model_name: 模型名称
            params: 模型参数
            test_size: 测试集比例
        
        Returns:
            dict: 包含模型、评估结果等的字典
        """
        # 数据分割
        if len(np.unique(y)) > 1:  # 检查是否有多个类别
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except ValueError:
                # 如果无法分层，使用常规分割
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
        else:
            st.error("标签只有一个类别，无法进行训练")
            return None
        
        # 获取模型类和参数
        model_class = self.get_available_models()[model_name]
        if params is None:
            params = self.get_default_params(model_name)
        
        # 特殊处理SVM的概率预测
        if model_name == 'SVM' and self.task_type == 'classification':
            params['probability'] = True
        
        # 训练模型
        self.model = model_class(**params)
        self.model_name = model_name
        
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"模型训练失败: {e}")
            return None
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 评估
        results = {
            'model': self.model,
            'model_name': model_name,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'params': params
        }
        
        if self.task_type == 'classification':
            results.update(self._evaluate_classification(y_test, y_pred))
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_test)[:, 1]
                results['y_proba'] = y_proba
                results.update(self._evaluate_classification_proba(y_test, y_proba))
        else:
            results.update(self._evaluate_regression(y_test, y_pred))
        
        return results
    
    def _evaluate_classification(self, y_true, y_pred):
        """分类任务评估"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }
    
    def _evaluate_classification_proba(self, y_true, y_proba):
        """分类任务概率评估"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        return {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
    
    def _evaluate_regression(self, y_true, y_pred):
        """回归任务评估"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def predict_single(self, X):
        """
        使用训练好的模型进行单个预测
        
        Args:
            X: 特征向量或特征矩阵
            
        Returns:
            预测结果和概率（如果是分类任务）
        """
        if self.model is None:
            raise ValueError("模型尚未训练或加载")
            
        # 确保输入格式正确
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # 应用特征缩放（如果有）
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
            
        prediction = self.model.predict(X)
        
        result = {
            'prediction': prediction[0] if len(prediction) == 1 else prediction,
            'model_name': self.model_name,
            'task_type': self.task_type
        }
        
        # 如果是分类任务，添加概率预测
        if self.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            result['probabilities'] = probabilities[0] if len(probabilities) == 1 else probabilities
            result['confidence'] = np.max(probabilities[0]) if len(probabilities) == 1 else np.max(probabilities, axis=1)
            
        return result
    
    def predict_batch(self, X):
        """
        批量预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            批量预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练或加载")
            
        # 应用特征缩放（如果有）
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
            
        predictions = self.model.predict(X)
        
        result = {
            'predictions': predictions,
            'model_name': self.model_name,
            'task_type': self.task_type,
            'num_samples': len(predictions)
        }
        
        # 如果是分类任务，添加概率预测
        if self.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            result['probabilities'] = probabilities
            result['confidence'] = np.max(probabilities, axis=1)
            
        return result
    
    def save_model_with_metadata(self, filepath, metadata=None):
        """
        保存模型及其元数据
        
        Args:
            filepath: 保存路径
            metadata: 额外的元数据
            
        Returns:
            bool: 保存是否成功
        """
        if self.model is None:
            return False
            
        try:
            # 创建完整的模型数据
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'task_type': self.task_type,
                'feature_scaler': self.feature_scaler,
                'training_metadata': self.training_metadata,
                'saved_at': datetime.now().isoformat()
            }
            
            if metadata:
                model_data.update(metadata)
                
            # 保存模型
            joblib.dump(model_data, filepath)
            return True
            
        except Exception as e:
            print(f"保存模型失败: {e}")
            return False
    
    def load_model_with_metadata(self, filepath):
        """
        加载模型及其元数据
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            bool: 加载是否成功
        """
        if not os.path.exists(filepath):
            return False
            
        try:
            # 加载模型数据
            model_data = joblib.load(filepath)
            
            # 兼容旧格式（直接保存的模型）
            if hasattr(model_data, 'predict'):
                self.model = model_data
                self.model_name = "Unknown"
                self.task_type = "classification"
                self.feature_scaler = None
                self.training_metadata = None
            else:
                # 新格式（包含元数据）
                self.model = model_data.get('model')
                self.model_name = model_data.get('model_name', "Unknown")
                self.task_type = model_data.get('task_type', "classification")
                self.feature_scaler = model_data.get('feature_scaler')
                self.training_metadata = model_data.get('training_metadata')
                
            return True
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', figsize=(8, 6)):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=figsize)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    return fig

def plot_roc_curve(fpr, tpr, roc_auc, title='ROC Curve', figsize=(8, 6)):
    """绘制ROC曲线"""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='blue', lw=2, 
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_feature_importance(model, feature_names=None, top_n=20, figsize=(10, 8)):
    """绘制特征重要性"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    # 获取前N个重要特征
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.invert_yaxis()
    
    return fig

def plot_prediction_scatter(y_true, y_pred, title='Prediction vs True Values', figsize=(8, 6)):
    """绘制预测值vs真实值散点图（回归任务）"""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    # 绘制理想线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # 添加R²值
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig

def train_model(X, y, algorithm, task_type, test_size=0.2):
    """
    简化的模型训练函数，供streamlit应用调用
    
    Args:
        X: 特征矩阵
        y: 目标变量
        algorithm: 算法名称
        task_type: 任务类型（"分类"或"回归"）
        test_size: 测试集比例
    
    Returns:
        dict: 训练结果
    """
    # 转换任务类型
    task_type_en = 'classification' if task_type == '分类' else 'regression'
    
    # 转换算法名称
    algorithm_mapping = {
        'LightGBM': 'LightGBM',
        'XGBoost': 'XGBoost', 
        '随机森林': 'Random Forest',
        '支持向量机': 'SVM'
    }
    
    model_name = algorithm_mapping.get(algorithm, algorithm)
    
    # 创建训练器
    trainer = ModelTrainer(task_type=task_type_en)
    
    # 训练模型
    results = trainer.train_model(X, y, model_name, test_size=test_size)
    
    if results:
        # 格式化结果用于显示
        formatted_results = {
            'model': results['model'],
            'metrics': {},
            'task_type': task_type
        }
        
        if task_type_en == 'classification':
            formatted_results['metrics'] = {
                '准确率': results['accuracy'],
                '精确率': results['precision'],
                '召回率': results['recall'],
                'F1得分': results['f1']
            }
            if 'confusion_matrix' in results:
                formatted_results['confusion_matrix'] = pd.DataFrame(
                    results['confusion_matrix'],
                    columns=[f'预测{i}' for i in range(results['confusion_matrix'].shape[1])],
                    index=[f'实际{i}' for i in range(results['confusion_matrix'].shape[0])]
                )
            # 添加ROC曲线数据
            if 'fpr' in results and 'tpr' in results and 'roc_auc' in results:
                formatted_results['roc_data'] = {
                    'fpr': results['fpr'],
                    'tpr': results['tpr'],
                    'roc_auc': results['roc_auc']
                }
                formatted_results['metrics']['AUC'] = results['roc_auc']
        else:
            formatted_results['metrics'] = {
                '均方误差': results['mse'],
                '均方根误差': results['rmse'],
                '平均绝对误差': results['mae'],
                'R²得分': results['r2']
            }
        
        return formatted_results
    
    return None

def hyperparameter_tuning(X, y, model_name, param_grid, cv=5, scoring='accuracy'):
    """超参数调优"""
    model_class = ModelTrainer().get_available_models()[model_name]
    model = model_class()
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    } 