"""
用户数据管理工具模块
管理个人文件夹、项目数据、嵌入结果等
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import uuid

class UserDataManager:
    """用户数据管理器"""
    
    def __init__(self, username):
        self.username = username
        self.base_path = f"users/{username}"
        self.ensure_user_folders()
    
    def ensure_user_folders(self):
        """确保用户文件夹存在"""
        folders = ['projects', 'embeddings', 'models', 'uploads']
        for folder in folders:
            os.makedirs(os.path.join(self.base_path, folder), exist_ok=True)
    
    def get_user_path(self, subfolder=""):
        """获取用户路径"""
        if subfolder:
            return os.path.join(self.base_path, subfolder)
        return self.base_path
    
    def save_uploaded_file(self, uploaded_file):
        """保存上传的文件，避免重复"""
        uploads_dir = os.path.join(self.base_path, "uploads")
        
        # 检查是否已存在同名文件
        original_filename = uploaded_file.name
        filepath = os.path.join(uploads_dir, original_filename)
        
        # 如果文件已存在，检查内容是否相同
        if os.path.exists(filepath):
            # 读取现有文件内容
            with open(filepath, "rb") as f:
                existing_content = f.read()
            
            # 比较文件内容
            new_content = uploaded_file.getbuffer()
            if existing_content == new_content:
                # 内容相同，返回现有文件
                return filepath, original_filename
            else:
                # 内容不同，添加时间戳
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{timestamp}{ext}"
                filepath = os.path.join(uploads_dir, filename)
        else:
            filename = original_filename
        
        # 保存文件
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return filepath, filename
    
    def save_embedding_result(self, smiles_data, embeddings, method, labels=None, metadata=None, project_name=None):
        """保存嵌入提取结果"""
        project_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建项目文件夹名称
        if project_name and project_name.strip():
            # 清理项目名称，移除特殊字符
            clean_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
            clean_name = clean_name.replace(' ', '_')
            folder_name = f"{timestamp}_{clean_name}_{project_id}"
        else:
            folder_name = f"{timestamp}_{project_id}"
        
        project_folder = os.path.join(self.base_path, "embeddings", folder_name)
        os.makedirs(project_folder, exist_ok=True)
        
        # 保存数据
        data = {
            'smiles': smiles_data,
            'embeddings': embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
            'labels': labels.tolist() if isinstance(labels, np.ndarray) else labels,
            'method': method,
            'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0]),
            'num_molecules': len(smiles_data),
            'created_at': timestamp,
            'project_id': project_id,
            'project_name': project_name if project_name and project_name.strip() else None
        }
        
        if metadata:
            data.update(metadata)
        
        # 保存为JSON和npy格式
        with open(os.path.join(project_folder, "embedding_data.json"), 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in data.items() if k != 'embeddings'}, f, ensure_ascii=False, indent=2)
        
        np.save(os.path.join(project_folder, "embeddings.npy"), embeddings)
        
        # 如果有标签，创建用于建模的数据文件
        if labels is not None:
            df = pd.DataFrame(embeddings)
            df.insert(0, 'SMILES', smiles_data)
            df['Label'] = labels
            df.to_csv(os.path.join(project_folder, "modeling_data.csv"), index=False)
        
        return project_id, project_folder
    
    def load_embedding_result(self, project_id):
        """加载嵌入提取结果"""
        # 查找项目文件夹
        embeddings_folder = os.path.join(self.base_path, "embeddings")
        project_folder = None
        
        for folder in os.listdir(embeddings_folder):
            if project_id in folder:
                project_folder = os.path.join(embeddings_folder, folder)
                break
        
        if not project_folder or not os.path.exists(project_folder):
            return None
        
        # 加载数据
        json_path = os.path.join(project_folder, "embedding_data.json")
        npy_path = os.path.join(project_folder, "embeddings.npy")
        
        if not os.path.exists(json_path) or not os.path.exists(npy_path):
            return None
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['embeddings'] = np.load(npy_path)
        
        return data
    
    def save_model_result(self, model_data, model_name, metrics, metadata=None):
        """保存模型训练结果"""
        project_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建项目文件夹
        project_folder = os.path.join(self.base_path, "models", f"{timestamp}_{model_name}_{project_id}")
        os.makedirs(project_folder, exist_ok=True)
        
        # 保存模型（使用新的带元数据的保存方法）
        model_path = os.path.join(project_folder, "model.pkl")
        
        # 准备模型保存的完整元数据
        model_save_metadata = {
            'model_name': model_name,
            'metrics': metrics,
            'created_at': timestamp,
            'project_id': project_id,
            'training_features': getattr(model_data.get('model'), 'n_features_in_', None),
            'model_params': getattr(model_data.get('model'), 'get_params', lambda: {})()
        }
        
        if metadata:
            model_save_metadata.update(metadata)
        
        # 使用ModelTrainer的保存方法
        from utils.model_utils import ModelTrainer
        trainer = ModelTrainer()
        trainer.model = model_data['model']
        trainer.model_name = model_name
        trainer.task_type = metadata.get('task_type', 'classification') if metadata else 'classification'
        trainer.training_metadata = model_save_metadata
        
        success = trainer.save_model_with_metadata(model_path, model_save_metadata)
        
        if not success:
            # 备用保存方法
            with open(model_path, 'wb') as f:
                pickle.dump(model_data['model'], f)
        
        # 保存详细的模型信息
        model_info = {
            'model_name': model_name,
            'metrics': metrics,
            'created_at': timestamp,
            'project_id': project_id,
            'model_path': model_path,
            'task_type': metadata.get('task_type', 'unknown') if metadata else 'unknown',
            'embedding_method': metadata.get('embedding_method', 'unknown') if metadata else 'unknown',
            'num_samples': metadata.get('num_samples', 0) if metadata else 0,
            'num_features': metadata.get('num_features', 0) if metadata else 0,
            'test_size': metadata.get('test_size', 0.2) if metadata else 0.2,
            'version': '2.0'  # 版本标识
        }
        
        if metadata:
            model_info.update({k: v for k, v in metadata.items() if k not in model_info})
        
        # 保存训练数据信息（如果有）
        if 'X_train' in model_data and 'y_train' in model_data:
            training_data_path = os.path.join(project_folder, "training_data.npz")
            np.savez(training_data_path, 
                    X_train=model_data['X_train'], 
                    y_train=model_data['y_train'],
                    X_test=model_data.get('X_test'),
                    y_test=model_data.get('y_test'))
            model_info['training_data_path'] = training_data_path
        
        # 保存预测结果（如果有）
        if 'y_pred' in model_data:
            predictions_path = os.path.join(project_folder, "predictions.npy")
            np.save(predictions_path, model_data['y_pred'])
            model_info['predictions_path'] = predictions_path
        
        with open(os.path.join(project_folder, "model_info.json"), 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        return project_id, project_folder
    
    def load_model_for_prediction(self, project_id):
        """
        加载已保存的模型用于预测
        
        Args:
            project_id: 项目ID
            
        Returns:
            ModelTrainer实例或None
        """
        # 查找项目文件夹
        models_folder = os.path.join(self.base_path, "models")
        project_folder = None
        
        for folder in os.listdir(models_folder):
            if project_id in folder:
                project_folder = os.path.join(models_folder, folder)
                break
        
        if not project_folder or not os.path.exists(project_folder):
            return None
        
        # 加载模型信息
        model_info_path = os.path.join(project_folder, "model_info.json")
        if not os.path.exists(model_info_path):
            return None
            
        with open(model_info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # 加载模型
        model_path = os.path.join(project_folder, "model.pkl")
        if not os.path.exists(model_path):
            return None
        
        from utils.model_utils import ModelTrainer
        trainer = ModelTrainer()
        
        # 尝试加载新格式的模型
        if trainer.load_model_with_metadata(model_path):
            return trainer, model_info
        else:
            # 备用：加载旧格式模型
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                trainer.model = model
                trainer.model_name = model_info.get('model_name', 'Unknown')
                trainer.task_type = model_info.get('task_type', 'classification')
                return trainer, model_info
            except:
                return None
    
    def get_available_models_for_prediction(self):
        """
        获取可用于预测的模型列表
        
        Returns:
            list: 模型信息列表
        """
        models = []
        models_folder = os.path.join(self.base_path, "models")
        
        if not os.path.exists(models_folder):
            return models
        
        for folder in os.listdir(models_folder):
            folder_path = os.path.join(models_folder, folder)
            if os.path.isdir(folder_path):
                model_info_path = os.path.join(folder_path, "model_info.json")
                if os.path.exists(model_info_path):
                    try:
                        with open(model_info_path, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                        
                        # 添加模型状态检查
                        model_path = os.path.join(folder_path, "model.pkl")
                        model_info['available'] = os.path.exists(model_path)
                        model_info['folder_name'] = folder
                        
                        models.append(model_info)
                    except:
                        continue
        
        # 按创建时间排序
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return models
    
    def get_user_projects(self):
        """获取用户的所有项目"""
        projects = {
            'embeddings': [],
            'models': [],
            'uploads': []
        }
        
        # 获取嵌入项目
        embeddings_path = os.path.join(self.base_path, "embeddings")
        if os.path.exists(embeddings_path):
            for folder in os.listdir(embeddings_path):
                folder_path = os.path.join(embeddings_path, folder)
                if os.path.isdir(folder_path):
                    json_path = os.path.join(folder_path, "embedding_data.json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            project_name = data.get('project_name')
                            display_name = project_name if project_name else data.get('project_id', folder)
                            projects['embeddings'].append({
                                'project_id': data.get('project_id', folder),
                                'project_name': display_name,
                                'method': data.get('method', '未知'),
                                'num_molecules': data.get('num_molecules', 0),
                                'created_at': data.get('created_at', ''),
                                'folder': folder
                            })
        
        # 获取模型项目
        models_path = os.path.join(self.base_path, "models")
        if os.path.exists(models_path):
            for folder in os.listdir(models_path):
                folder_path = os.path.join(models_path, folder)
                if os.path.isdir(folder_path):
                    json_path = os.path.join(folder_path, "model_info.json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            projects['models'].append({
                                'project_id': data.get('project_id', folder),
                                'model_name': data.get('model_name', '未知'),
                                'metrics': data.get('metrics', {}),
                                'created_at': data.get('created_at', ''),
                                'folder': folder
                            })
        
        # 获取上传文件
        uploads_path = os.path.join(self.base_path, "uploads")
        if os.path.exists(uploads_path):
            for filename in os.listdir(uploads_path):
                filepath = os.path.join(uploads_path, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    projects['uploads'].append({
                        'filename': filename,
                        'size': stat.st_size,
                        'created_at': datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                        'filepath': filepath
                    })
        
        return projects
    
    def delete_project(self, project_type, project_id):
        """删除项目"""
        if project_type == 'embeddings':
            folder_path = os.path.join(self.base_path, "embeddings")
        elif project_type == 'models':
            folder_path = os.path.join(self.base_path, "models")
        else:
            return False
        
        # 查找并删除项目文件夹
        for folder in os.listdir(folder_path):
            if project_id in folder:
                project_folder = os.path.join(folder_path, folder)
                import shutil
                shutil.rmtree(project_folder)
                return True
        
        return False
    
    def get_embedding_projects_for_modeling(self):
        """获取可用于建模的嵌入项目"""
        projects = []
        embeddings_path = os.path.join(self.base_path, "embeddings")
        
        if os.path.exists(embeddings_path):
            for folder in os.listdir(embeddings_path):
                folder_path = os.path.join(embeddings_path, folder)
                if os.path.isdir(folder_path):
                    # 检查是否有建模数据文件
                    modeling_data_path = os.path.join(folder_path, "modeling_data.csv")
                    if os.path.exists(modeling_data_path):
                        json_path = os.path.join(folder_path, "embedding_data.json")
                        if os.path.exists(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                projects.append({
                                    'project_id': data.get('project_id', folder),
                                    'method': data.get('method', '未知'),
                                    'num_molecules': data.get('num_molecules', 0),
                                    'created_at': data.get('created_at', ''),
                                    'modeling_data_path': modeling_data_path,
                                    'folder': folder
                                })
        
        return projects
    
    def load_modeling_data(self, project_id):
        """加载建模数据"""
        embeddings_path = os.path.join(self.base_path, "embeddings")
        
        for folder in os.listdir(embeddings_path):
            if project_id in folder:
                modeling_data_path = os.path.join(embeddings_path, folder, "modeling_data.csv")
                if os.path.exists(modeling_data_path):
                    return pd.read_csv(modeling_data_path)
        
        return None
    
    def export_project_summary(self):
        """导出项目摘要"""
        projects = self.get_user_projects()
        
        summary = {
            'username': self.username,
            'export_time': datetime.now().isoformat(),
            'total_embeddings': len(projects['embeddings']),
            'total_models': len(projects['models']),
            'total_uploads': len(projects['uploads']),
            'projects': projects
        }
        
        return summary 