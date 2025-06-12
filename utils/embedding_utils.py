"""
嵌入提取工具模块
支持三种嵌入提取方法：RDKit分子指纹、ChemBERTa、SMILES Transformer
"""

import os
import sys
import numpy as np
import pandas as pd
import subprocess
import tempfile
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

try:
    import streamlit as st
except ImportError:
    # 如果不在streamlit环境中，创建一个mock对象
    class MockStreamlit:
        def write(self, *args, **kwargs):
            print(*args)
        def error(self, *args, **kwargs):
            print("ERROR:", *args)
        def warning(self, *args, **kwargs):
            print("WARNING:", *args)
    st = MockStreamlit()

def extract_rdkit_fingerprints(smiles_list, fp_type='morgan', radius=2, n_bits=2048):
    """
    使用RDKit提取分子指纹
    
    Args:
        smiles_list: SMILES字符串列表
        fp_type: 指纹类型 ('morgan', 'maccs', 'topological')
        radius: Morgan指纹半径
        n_bits: 指纹位数
    
    Returns:
        numpy数组: 指纹矩阵
    """
    fingerprints = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 如果SMILES无效，使用零向量
            fingerprints.append(np.zeros(n_bits))
            continue
            
        if fp_type == 'morgan':
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = fpgen.GetFingerprint(mol)
        elif fp_type == 'maccs':
            fp = Chem.MACCSkeys.GenMACCSKeys(mol)
            n_bits = 167  # MACCS keys固定长度
        elif fp_type == 'topological':
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        else:
            raise ValueError(f"不支持的指纹类型: {fp_type}")
        
        # 转换为numpy数组
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
    
    return np.array(fingerprints)

def extract_chembert_embeddings(smiles_list, batch_size=32):
    """
    使用ChemBERTa模型提取分子嵌入
    
    Args:
        smiles_list: SMILES字符串列表
        batch_size: 批处理大小
    
    Returns:
        numpy数组: 768维嵌入矩阵
    """
    # 创建临时CSV文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_input = f.name
        df = pd.DataFrame({'smiles': smiles_list})
        df.to_csv(temp_input, index=False)
    
    # 创建临时输出文件
    temp_output = tempfile.mktemp(suffix='.npy')
    
    try:
        # 调用ChemBERTa脚本
        script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'chembert.py')
        cmd = [
            sys.executable, script_path,
            '--input', temp_input,
            '--output', temp_output,
            '--batch_size', str(batch_size)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script_path))
        
        if result.returncode != 0:
            st.error(f"ChemBERTa嵌入提取失败: {result.stderr}")
            return None
        
        # 加载结果
        embeddings = np.load(temp_output)
        return embeddings
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_input):
            os.unlink(temp_input)
        if os.path.exists(temp_output):
            os.unlink(temp_output)

def extract_smiles_transformer_embeddings(smiles_list, batch_size=16):
    """
    使用SMILES Transformer模型提取分子嵌入
    
    Args:
        smiles_list: SMILES字符串列表
        batch_size: 批处理大小
    
    Returns:
        numpy数组: 1024维嵌入矩阵
    """
    # 检查模型文件是否存在
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model', 'smiles_transformer')
    vocab_path = os.path.join(model_dir, 'vocab.pkl')
    model_path = os.path.join(model_dir, 'trfm_12_23000.pkl')
    
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        st.error("SMILES Transformer模型文件不存在，请确保vocab.pkl和trfm_12_23000.pkl在model/smiles_transformer目录下")
        return None
    
    # 创建临时CSV文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_input = f.name
        df = pd.DataFrame({'smiles': smiles_list})
        df.to_csv(temp_input, index=False)
    
    # 创建临时输出文件
    temp_output = tempfile.mktemp(suffix='.npy')
    
    try:
        # 调用SMILES Transformer脚本
        script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'smiles_transformer.py')
        cmd = [
            sys.executable, script_path,
            '--input', temp_input,
            '--output', temp_output,
            '--vocab_path', vocab_path,
            '--model_path', model_path,
            '--batch_size', str(batch_size)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script_path))
        
        if result.returncode != 0:
            st.error(f"SMILES Transformer嵌入提取失败: {result.stderr}")
            return None
        
        # 加载结果
        embeddings = np.load(temp_output)
        return embeddings
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_input):
            os.unlink(temp_input)
        if os.path.exists(temp_output):
            os.unlink(temp_output)

def extract_embeddings(smiles_list, method_name):
    """
    统一的嵌入提取接口
    
    Args:
        smiles_list: SMILES字符串列表
        method_name: 提取方法名称
    
    Returns:
        numpy数组: 嵌入矩阵
    """
    if method_name == "RDKit 指纹":
        return extract_rdkit_fingerprints(smiles_list, fp_type='morgan', radius=2, n_bits=2048)
    elif method_name == "ChemBERTa 嵌入":
        return extract_chembert_embeddings(smiles_list, batch_size=32)
    elif method_name == "SMILES Transformer 嵌入":
        return extract_smiles_transformer_embeddings(smiles_list, batch_size=16)
    else:
        raise ValueError(f"不支持的嵌入方法: {method_name}")

def get_embedding_info(method):
    """
    获取嵌入方法的信息
    
    Args:
        method: 嵌入方法名称
    
    Returns:
        dict: 包含维度、描述等信息的字典
    """
    info = {
        'rdkit': {
            'dimensions': 2048,
            'description': 'RDKit分子指纹，基于分子结构的二进制特征',
            'speed': '快',
            'memory': '低'
        },
        'chembert': {
            'dimensions': 768,
            'description': 'ChemBERTa预训练模型，基于Transformer的分子表示',
            'speed': '中等',
            'memory': '中等'
        },
        'smiles_transformer': {
            'dimensions': 1024,
            'description': '自定义SMILES Transformer，专门针对SMILES序列训练',
            'speed': '慢',
            'memory': '高'
        }
    }
    return info.get(method, {}) 