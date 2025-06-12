"""
数据处理和可视化工具模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import streamlit as st

# 设置matplotlib字体，确保英文字符正常显示
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """
    加载数据文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        pandas.DataFrame: 加载的数据
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        else:
            st.error("不支持的文件格式，请使用CSV或Excel文件")
            return None
    except Exception as e:
        st.error(f"文件加载失败: {e}")
        return None

def validate_smiles(smiles_list):
    """
    验证SMILES字符串的有效性
    
    Args:
        smiles_list: SMILES字符串列表
    
    Returns:
        dict: 包含有效和无效SMILES的统计信息
    """
    valid_count = 0
    invalid_smiles = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_count += 1
        else:
            invalid_smiles.append((i, smiles))
    
    return {
        'total': len(smiles_list),
        'valid': valid_count,
        'invalid': len(invalid_smiles),
        'invalid_smiles': invalid_smiles,
        'valid_ratio': valid_count / len(smiles_list) if smiles_list else 0
    }

def calculate_molecular_descriptors(smiles_list):
    """
    计算分子描述符
    
    Args:
        smiles_list: SMILES字符串列表
    
    Returns:
        pandas.DataFrame: 分子描述符数据框
    """
    descriptors = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            desc = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'NumHDonors': Lipinski.NumHDonors(mol),
                'NumHAcceptors': Lipinski.NumHAcceptors(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'RingCount': Descriptors.RingCount(mol)
            }
        else:
            desc = {key: np.nan for key in [
                'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
                'TPSA', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings', 'RingCount'
            ]}
        descriptors.append(desc)
    
    return pd.DataFrame(descriptors)

def plot_data_distribution(data, columns=None, figsize=(15, 10)):
    """
    绘制数据分布图
    
    Args:
        data: 数据框
        columns: 要绘制的列名列表
        figsize: 图像大小
    
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns[:6]
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            if data[col].dtype in ['int64', 'float64']:
                sns.histplot(data[col].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
    
    # 隐藏多余的子图
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(data, figsize=(12, 10)):
    """
    绘制相关性矩阵热图
    
    Args:
        data: 数据框
        figsize: 图像大小
    
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return None
    
    correlation_matrix = numeric_data.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    
    return fig

def plot_molecular_property_distribution(descriptors_df, figsize=(15, 12)):
    """
    绘制分子性质分布图
    
    Args:
        descriptors_df: 分子描述符数据框
        figsize: 图像大小
    
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    properties = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 
                 'NumRotatableBonds', 'TPSA']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, prop in enumerate(properties):
        if prop in descriptors_df.columns:
            sns.histplot(descriptors_df[prop].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'{prop} Distribution')
            axes[i].set_xlabel(prop)
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig

def plot_lipinski_analysis(descriptors_df, figsize=(12, 8)):
    """
    绘制Lipinski五规则分析
    
    Args:
        descriptors_df: 分子描述符数据框
        figsize: 图像大小
    
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    # 计算Lipinski规则违反情况
    violations = pd.DataFrame({
        'MolWt > 500': descriptors_df['MolWt'] > 500,
        'LogP > 5': descriptors_df['LogP'] > 5,
        'HBD > 5': descriptors_df['NumHDonors'] > 5,
        'HBA > 10': descriptors_df['NumHAcceptors'] > 10
    })
    
    violation_counts = violations.sum()
    total_violations = violations.sum(axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 每个规则的违反频次
    violation_counts.plot(kind='bar', ax=ax1, color='lightcoral')
    ax1.set_title('Lipinski Rule Violations')
    ax1.set_xlabel('Rules')
    ax1.set_ylabel('Number of Violations')
    ax1.tick_params(axis='x', rotation=45)
    
    # 总违反数分布
    total_violations.value_counts().sort_index().plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Distribution of Total Rule Violations')
    ax2.set_xlabel('Number of Violations')
    ax2.set_ylabel('Number of Molecules')
    
    plt.tight_layout()
    return fig

def display_data_summary(data):
    """
    显示数据摘要信息
    
    Args:
        data: 数据框
    """
    st.subheader("📊 数据集基本信息")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("样本数量", len(data))
    
    with col2:
        st.metric("特征数量", len(data.columns))
    
    with col3:
        missing_ratio = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("缺失值比例", f"{missing_ratio:.1f}%")
    
    with col4:
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric("数值型特征", numeric_cols)
    
    # 数据类型信息
    st.subheader("🔍 数据类型分布")
    dtype_counts = data.dtypes.value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**各列数据类型:**")
        st.dataframe(pd.DataFrame({
            '列名': data.columns,
            '数据类型': data.dtypes.values,
            '非空值数量': data.count().values,
            '缺失值数量': data.isnull().sum().values
        }))
    
    with col2:
        st.write("**数据类型统计:**")
        fig, ax = plt.subplots(figsize=(8, 6))
        dtype_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title('Data Type Distribution')
        st.pyplot(fig)

def preprocess_data(data, smiles_col, label_col, remove_invalid=True):
    """
    数据预处理
    
    Args:
        data: 数据框
        smiles_col: SMILES列名
        label_col: 标签列名
        remove_invalid: 是否移除无效的SMILES
    
    Returns:
        pandas.DataFrame: 预处理后的数据
    """
    # 复制数据
    processed_data = data.copy()
    
    # 移除缺失值
    processed_data = processed_data.dropna(subset=[smiles_col, label_col])
    
    # 验证SMILES
    if remove_invalid:
        valid_mask = processed_data[smiles_col].apply(
            lambda x: Chem.MolFromSmiles(x) is not None
        )
        processed_data = processed_data[valid_mask]
    
    return processed_data 