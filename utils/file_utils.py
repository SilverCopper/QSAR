"""
文件管理工具模块
"""

import os
import glob
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import random
import string

def create_project_directory(project_name=None):
    """
    创建项目目录
    
    Args:
        project_name: 项目名称（可选）
    
    Returns:
        str: 项目目录路径
    """
    if project_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        project_name = f"{timestamp}_{random_id}"
    
    project_dir = os.path.join("./projects", project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    return project_dir

def save_project_metadata(project_dir, metadata):
    """
    保存项目元数据
    
    Args:
        project_dir: 项目目录
        metadata: 元数据字典
    """
    metadata_file = os.path.join(project_dir, "metadata.json")
    
    # 添加时间戳
    metadata['created_time'] = datetime.now().isoformat()
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_project_metadata(project_dir):
    """
    加载项目元数据
    
    Args:
        project_dir: 项目目录
    
    Returns:
        dict: 元数据字典
    """
    metadata_file = os.path.join(project_dir, "metadata.json")
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return {}

def get_existing_projects():
    """
    获取现有项目列表
    
    Returns:
        list: 项目目录列表
    """
    projects_dir = "./projects"
    if not os.path.exists(projects_dir):
        os.makedirs(projects_dir)
        return []
    
    projects = []
    for item in os.listdir(projects_dir):
        item_path = os.path.join(projects_dir, item)
        if os.path.isdir(item_path):
            projects.append(item)
    
    return sorted(projects, reverse=True)  # 按时间倒序

def save_embeddings(embeddings, project_dir, method_name, metadata=None):
    """
    保存嵌入向量
    
    Args:
        embeddings: 嵌入矩阵
        project_dir: 项目目录
        method_name: 方法名称
        metadata: 额外的元数据
    """
    # 保存嵌入向量
    embeddings_file = os.path.join(project_dir, f"embeddings_{method_name}.npy")
    np.save(embeddings_file, embeddings)
    
    # 保存嵌入元数据
    embedding_metadata = {
        'method': method_name,
        'shape': embeddings.shape,
        'file': f"embeddings_{method_name}.npy",
        'created_time': datetime.now().isoformat()
    }
    
    if metadata:
        embedding_metadata.update(metadata)
    
    metadata_file = os.path.join(project_dir, f"embeddings_{method_name}_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(embedding_metadata, f, ensure_ascii=False, indent=2)

def load_embeddings(project_dir, method_name):
    """
    加载嵌入向量
    
    Args:
        project_dir: 项目目录
        method_name: 方法名称
    
    Returns:
        tuple: (嵌入矩阵, 元数据)
    """
    embeddings_file = os.path.join(project_dir, f"embeddings_{method_name}.npy")
    metadata_file = os.path.join(project_dir, f"embeddings_{method_name}_metadata.json")
    
    embeddings = None
    metadata = {}
    
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return embeddings, metadata

def save_model_results(results, project_dir):
    """
    保存模型训练结果
    
    Args:
        results: 训练结果字典
        project_dir: 项目目录
    """
    # 保存模型
    model_file = os.path.join(project_dir, "model.pkl")
    joblib.dump(results['model'], model_file)
    
    # 保存训练数据
    train_data_file = os.path.join(project_dir, "train_data.npz")
    np.savez(train_data_file,
             X_train=results['X_train'],
             X_test=results['X_test'],
             y_train=results['y_train'],
             y_test=results['y_test'],
             y_pred=results['y_pred'])
    
    # 保存模型元数据（排除不能序列化的对象）
    model_metadata = {
        'model_name': results['model_name'],
        'params': results['params'],
        'created_time': datetime.now().isoformat()
    }
    
    # 添加评估指标
    for key, value in results.items():
        if key not in ['model', 'X_train', 'X_test', 'y_train', 'y_test', 'y_pred', 'confusion_matrix']:
            if isinstance(value, (int, float, str, list)):
                model_metadata[key] = value
            elif isinstance(value, np.ndarray):
                model_metadata[key] = value.tolist()
    
    metadata_file = os.path.join(project_dir, "model_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(model_metadata, f, ensure_ascii=False, indent=2)

def load_model_results(project_dir):
    """
    加载模型训练结果
    
    Args:
        project_dir: 项目目录
    
    Returns:
        dict: 训练结果字典
    """
    results = {}
    
    # 加载模型
    model_file = os.path.join(project_dir, "model.pkl")
    if os.path.exists(model_file):
        results['model'] = joblib.load(model_file)
    
    # 加载训练数据
    train_data_file = os.path.join(project_dir, "train_data.npz")
    if os.path.exists(train_data_file):
        data = np.load(train_data_file)
        results.update({
            'X_train': data['X_train'],
            'X_test': data['X_test'],
            'y_train': data['y_train'],
            'y_test': data['y_test'],
            'y_pred': data['y_pred']
        })
    
    # 加载元数据
    metadata_file = os.path.join(project_dir, "model_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            results.update(metadata)
    
    return results

def get_available_datasets():
    """
    获取可用的数据集列表
    
    Returns:
        list: 数据集文件路径列表
    """
    data_dir = "./data"
    if not os.path.exists(data_dir):
        return []
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    excel_files.extend(glob.glob(os.path.join(data_dir, "*.xls")))
    
    return csv_files + excel_files

def export_results_to_csv(project_dir, include_embeddings=False):
    """
    导出项目结果到CSV文件
    
    Args:
        project_dir: 项目目录
        include_embeddings: 是否包含嵌入向量
    
    Returns:
        str: 导出文件路径
    """
    export_data = {}
    
    # 加载原始数据
    input_file = os.path.join(project_dir, "input_data.csv")
    if os.path.exists(input_file):
        input_data = pd.read_csv(input_file)
        export_data.update(input_data.to_dict('series'))
    
    # 加载嵌入向量（可选）
    if include_embeddings:
        for method in ['rdkit', 'chembert', 'smiles_transformer']:
            embeddings, _ = load_embeddings(project_dir, method)
            if embeddings is not None:
                embedding_df = pd.DataFrame(
                    embeddings, 
                    columns=[f'{method}_dim_{i}' for i in range(embeddings.shape[1])]
                )
                export_data.update(embedding_df.to_dict('series'))
    
    # 加载预测结果
    model_results = load_model_results(project_dir)
    if 'y_pred' in model_results:
        export_data['predictions'] = model_results['y_pred']
    
    # 创建DataFrame并导出
    if export_data:
        export_df = pd.DataFrame(export_data)
        export_file = os.path.join(project_dir, "export_results.csv")
        export_df.to_csv(export_file, index=False)
        return export_file
    
    return None

def cleanup_old_projects(keep_recent=10):
    """
    清理旧项目文件
    
    Args:
        keep_recent: 保留最近的项目数量
    """
    projects = get_existing_projects()
    
    if len(projects) > keep_recent:
        projects_to_remove = projects[keep_recent:]
        
        for project in projects_to_remove:
            project_path = os.path.join("./projects", project)
            if os.path.exists(project_path):
                import shutil
                shutil.rmtree(project_path) 