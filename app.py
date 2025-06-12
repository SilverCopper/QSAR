"""
基于预训练深度学习模型的定量构效关系预测平台

一个集成多种分子嵌入方法和机器学习算法的QSAR预测平台
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import requests
import json

# 设置页面配置
st.set_page_config(
    page_title="QSAR深度学习预测平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入工具模块
from utils.embedding_utils import extract_embeddings, get_embedding_info
from utils.model_utils import ModelTrainer, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from utils.data_utils import (
    load_data, display_data_summary, validate_smiles, calculate_molecular_descriptors,
    plot_data_distribution, plot_correlation_matrix, plot_molecular_property_distribution,
    plot_lipinski_analysis, preprocess_data
)
from utils.file_utils import (
    create_project_directory, save_project_metadata, load_project_metadata,
    get_existing_projects, save_embeddings, load_embeddings, save_model_results,
    load_model_results, get_available_datasets
)

warnings.filterwarnings("ignore")

def main():
    """主函数"""
    
    st.markdown("""
    <style>
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .user-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 初始化 session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login' if not st.session_state.logged_in else 'home'
    
    # 检查登录状态
    if not st.session_state.logged_in:
        from utils.auth_utils import render_login_page
        render_login_page()
        return
    
    # 顶部用户信息和导航栏 - 调整对齐
    col_user, col1, col2, col3, col4, col5, col6, col_logout = st.columns([1.2, 1, 1, 1, 1, 1, 1, 1])
    
    with col_user:
        # 使用与按钮完全相同的样式和高度
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            margin: 0.2rem;
            text-align: center;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 400;
            line-height: 1.6;
            border: 1px solid transparent;
            box-sizing: border-box;
        ">👤 {st.session_state.username}</div>
        """, unsafe_allow_html=True)
    
    with col1:
        if st.button("🏠 首页", key="nav_home"):
            st.session_state.current_page = 'home'
            st.rerun()
    with col2:
        if st.button("📊 数据探索", key="nav_data"):
            st.session_state.current_page = 'data_exploration'
            st.rerun()
    with col3:
        if st.button("🧠 智能建模", key="nav_smart_modeling"):
            st.session_state.current_page = 'smart_modeling'
            st.rerun()
    with col4:
        if st.button("📁 项目管理", key="nav_projects"):
            st.session_state.current_page = 'project_viewer'
            st.rerun()
    with col5:
        if st.button("📚 文献挖掘", key="nav_literature"):
            st.session_state.current_page = 'literature_mining'
            st.rerun()
    
    with col6:
        if st.button("🤖 AI摘要", key="nav_ai_summary"):
            st.session_state.current_page = 'ai_summary'
            st.rerun()
    
    with col_logout:
        if st.button("🚪 登出", key="nav_logout"):
            from utils.auth_utils import logout_user
            logout_user()
    
    st.markdown("---")
    
    # 路由到对应页面
    if st.session_state.current_page == 'home':
        render_homepage()
    elif st.session_state.current_page == 'data_exploration':
        render_data_exploration()
    elif st.session_state.current_page == 'smart_modeling':
        render_smart_modeling()
    elif st.session_state.current_page == 'project_viewer':
        render_project_viewer()
    elif st.session_state.current_page == 'literature_mining':
        render_literature_mining()
    elif st.session_state.current_page == 'ai_summary':
        render_ai_summary()

def render_homepage():
    """渲染首页"""
    st.markdown('<h1 class="main-title">🧬 QSAR深度学习预测平台</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d; margin-bottom: 3rem;">基于预训练深度学习模型的定量构效关系预测</p>', unsafe_allow_html=True)
    
    # 平台特性展示
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">📊</div>
            <div style="color: #667eea; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; text-align: center;">数据探索分析</div>
            <div style="text-align: center; line-height: 1.6;">
                智能数据预处理、可视化分析<br/>
                分子描述符计算、SMILES验证<br/>
                Lipinski规则分析
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🔬</div>
            <div style="color: #667eea; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; text-align: center;">多模态嵌入提取</div>
            <div style="text-align: center; line-height: 1.6;">
                RDKit分子指纹、ChemBERTa<br/>
                SMILES Transformer<br/>
                多维度分子表示学习
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🧪</div>
            <div style="color: #667eea; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; text-align: center;">实时预测建模</div>
            <div style="text-align: center; line-height: 1.6;">
                LightGBM、XGBoost、随机森林<br/>
                训练完成立即预测<br/>
                支持单个和批量预测
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🧠</div>
            <div style="color: #667eea; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; text-align: center;">AI智能摘要</div>
            <div style="text-align: center; line-height: 1.6;">
                文献摘要智能分析<br/>
                核心要点自动提炼<br/>
                中文输出关键信息
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 个人统计信息
    st.markdown("### 📈 个人统计")
    
    # 获取用户数据
    from utils.user_data_utils import UserDataManager
    user_data = UserDataManager(st.session_state.username)
    user_projects = user_data.get_user_projects()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{len(user_projects['embeddings'])}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">嵌入项目</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{len(user_projects['models'])}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">模型项目</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{len(user_projects['uploads'])}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">上传文件</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_size = sum(f['size'] for f in user_projects['uploads'])
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{total_size/1024/1024:.1f}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">存储(MB)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 快速开始指南
    st.markdown("### 🚀 快速开始")
    st.info("""
    **智能建模流程：**
    1. **数据探索** → 上传数据集，进行分子性质分析和可视化
    2. **智能建模** → 一站式完成分子嵌入提取和机器学习建模
    3. **项目管理** → 查看和管理个人的嵌入项目、模型和文件
    4. **文献挖掘** → 搜索相关研究文献，支持多种检索策略
    5. **AI摘要** → 智能分析文献摘要，提炼核心要点和创新点
    
    💡 **提示**: 新用户可以从"智能建模"开始，体验完整的QSAR建模流程
    """)

def render_data_exploration():
    """渲染数据探索页面"""
    st.title("📊 数据探索分析")
    
    # 数据源选择
    data_source = st.radio(
        "选择数据源",
        ["使用现有数据集", "上传新数据"],
        horizontal=True
    )
    
    data = None
    
    if data_source == "使用现有数据集":
        datasets = get_available_datasets()
        if datasets:
            dataset_names = [os.path.basename(f) for f in datasets]
            selected_dataset = st.selectbox("选择数据集", dataset_names)
            
            if selected_dataset:
                dataset_path = next(f for f in datasets if os.path.basename(f) == selected_dataset)
                data = load_data(dataset_path)
        else:
            st.warning("data目录下没有找到数据集文件")
            
    else:
        uploaded_file = st.file_uploader(
            "上传数据文件", 
            type=['csv', 'xlsx', 'xls'],
            help="支持CSV和Excel格式"
        )
        
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
    
    # 数据分析
    if data is not None:
        # 基本信息展示
        display_data_summary(data)
        
        # 数据预览
        st.subheader("🔍 数据预览")
        st.dataframe(data.head(10), use_container_width=True)
        
        # SMILES列检测和验证
        st.subheader("🧪 SMILES分析")
        
        smiles_columns = [col for col in data.columns if 'smiles' in col.lower()]
        if smiles_columns:
            smiles_col = st.selectbox("选择SMILES列", smiles_columns)
            
            if smiles_col:
                # SMILES验证
                smiles_list = data[smiles_col].dropna().tolist()
                validation_result = validate_smiles(smiles_list)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总分子数", validation_result['total'])
                with col2:
                    st.metric("有效分子数", validation_result['valid'])
                with col3:
                    st.metric("有效率", f"{validation_result['valid_ratio']:.1%}")
                
                if validation_result['invalid'] > 0:
                    with st.expander(f"查看无效SMILES ({validation_result['invalid']}个)"):
                        invalid_df = pd.DataFrame(validation_result['invalid_smiles'], 
                                                columns=['索引', 'SMILES'])
                        st.dataframe(invalid_df)
                
                # 分子描述符计算
                if st.button("计算分子描述符"):
                    with st.spinner("正在计算分子描述符..."):
                        descriptors_df = calculate_molecular_descriptors(smiles_list)
                        
                        st.subheader("📈 分子性质分布")
                        fig = plot_molecular_property_distribution(descriptors_df)
                        st.pyplot(fig)
                        
                        st.subheader("⚖️ Lipinski规则分析")
                        fig = plot_lipinski_analysis(descriptors_df)
                        st.pyplot(fig)
                        
                        # 描述符统计
                        st.subheader("📋 描述符统计")
                        st.dataframe(descriptors_df.describe(), use_container_width=True)
        else:
            st.warning("未找到SMILES列，请确保数据中包含名为'smiles'或类似的列")

def render_smart_modeling():
    """渲染智能建模页面（合并嵌入提取和模型训练）"""
    st.title("🧠 智能建模平台")
    st.markdown("**一站式分子嵌入提取与机器学习建模**")
    
    # 获取用户数据管理器
    from utils.user_data_utils import UserDataManager
    user_data = UserDataManager(st.session_state.username)
    
    # 建模流程选择
    modeling_mode = st.radio(
        "选择建模方式",
        ["新建项目", "使用已有嵌入"],
        horizontal=True
    )
    
    if modeling_mode == "新建项目":
        render_new_modeling_project(user_data)
    else:
        render_existing_embedding_modeling(user_data)

def render_new_modeling_project(user_data):
    """渲染新建建模项目"""
    st.subheader("📊 步骤1: 数据准备")
    
    # 数据输入方式
    input_method = st.radio("数据输入方式", ["上传文件", "手动输入", "使用已上传文件"])
    
    data_df = None
    smiles_data = []
    labels = None
    
    if input_method == "上传文件":
        uploaded_file = st.file_uploader("上传包含SMILES和标签的CSV文件", type=['csv'])
        if uploaded_file:
            # 保存文件到用户目录
            filepath, filename = user_data.save_uploaded_file(uploaded_file)
            
            data_df = pd.read_csv(uploaded_file)
            st.write("文件预览：")
            st.dataframe(data_df.head())
            
    elif input_method == "使用已上传文件":
        projects = user_data.get_user_projects()
        if projects['uploads']:
            upload_files = [f['filename'] for f in projects['uploads']]
            selected_file = st.selectbox("选择文件", upload_files)
            
            if selected_file:
                # 找到文件路径
                file_info = next(f for f in projects['uploads'] if f['filename'] == selected_file)
                data_df = pd.read_csv(file_info['filepath'])
                st.write("文件预览：")
                st.dataframe(data_df.head())
        else:
            st.info("没有已上传的文件")
            
    else:  # 手动输入
        col1, col2 = st.columns(2)
        with col1:
            smiles_input = st.text_area("输入SMILES（每行一个）", height=150)
            if smiles_input:
                smiles_data = [s.strip() for s in smiles_input.split('\n') if s.strip()]
        
        with col2:
            labels_input = st.text_area("输入对应标签（每行一个）", height=150)
            if labels_input:
                labels = [float(l.strip()) if l.strip().replace('.','').replace('-','').isdigit() 
                         else l.strip() for l in labels_input.split('\n') if l.strip()]
        
        if smiles_data and labels and len(smiles_data) == len(labels):
            data_df = pd.DataFrame({'SMILES': smiles_data, 'Label': labels})
            st.write("数据预览：")
            st.dataframe(data_df.head())
    
    if data_df is not None:
        # 列选择
        st.subheader("🎯 步骤2: 选择列")
        
        smiles_col = st.selectbox("选择SMILES列", data_df.columns.tolist())
        label_col = st.selectbox("选择标签列", [col for col in data_df.columns if col != smiles_col])
        
        if smiles_col and label_col:
            smiles_data = data_df[smiles_col].dropna().tolist()
            labels = data_df[label_col].dropna().tolist()
            
            # 确保数据长度一致
            min_len = min(len(smiles_data), len(labels))
            smiles_data = smiles_data[:min_len]
            labels = labels[:min_len]
            
            st.success(f"准备了 {len(smiles_data)} 个样本用于建模")
            
            # 嵌入提取
            st.subheader("🔬 步骤3: 分子嵌入提取")
            
            embedding_method = st.selectbox(
                "选择嵌入方法",
                ["RDKit 指纹", "ChemBERTa 嵌入", "SMILES Transformer 嵌入"]
            )
            
            # 项目命名
            project_name = st.text_input(
                "项目名称（可选）",
                placeholder="为您的嵌入项目命名，留空将使用默认名称",
                help="自定义项目名称便于后续管理和识别"
            )
            
            if st.button("提取嵌入向量"):
                with st.spinner(f"使用 {embedding_method} 提取嵌入向量..."):
                    from utils.embedding_utils import extract_embeddings
                    
                    try:
                        embeddings = extract_embeddings(smiles_data, embedding_method)
                        
                        if embeddings is not None:
                            st.success(f"成功提取了 {len(embeddings)} 个分子的嵌入向量")
                            st.write(f"嵌入向量维度: {embeddings.shape}")
                            
                            # 保存嵌入结果
                            project_id, project_folder = user_data.save_embedding_result(
                                smiles_data, embeddings, embedding_method, labels,
                                metadata={'original_data_cols': [smiles_col, label_col]},
                                project_name=project_name
                            )
                            
                            st.success(f"嵌入结果已保存，项目ID: {project_id}")
                            
                            # 保存到session state以便训练使用
                            st.session_state.current_embeddings = embeddings
                            st.session_state.current_labels = labels
                            st.session_state.current_method = embedding_method
                            st.session_state.embedding_ready = True
                            
                            st.info("✅ 嵌入提取完成！现在可以进行模型训练。")
                            
                    except Exception as e:
                        st.error(f"嵌入提取失败: {str(e)}")
            
            # 检查是否有准备好的嵌入数据，如果有则显示训练界面
            if st.session_state.get('embedding_ready', False):
                embeddings = st.session_state.current_embeddings
                labels = st.session_state.current_labels
                method_name = st.session_state.current_method
                
                st.markdown("---")
                render_model_training_section(embeddings, labels, user_data, method_name)

            # 如果训练结果存在，则显示它
            if 'last_training_result' in st.session_state:
                st.markdown("---")
                render_training_results(st.session_state.last_training_result)

            # 模型训练完成后，独立渲染预测界面
            if st.session_state.get('model_trained', False):
                render_realtime_prediction_interface()

def render_existing_embedding_modeling(user_data):
    """使用已有嵌入进行建模"""
    st.subheader("📁 选择已有嵌入项目")
    
    embedding_projects = user_data.get_embedding_projects_for_modeling()
    
    if not embedding_projects:
        st.info("没有可用于建模的嵌入项目。请先创建包含标签的嵌入项目。")
        return
    
    # 项目选择
    project_options = [f"{p['method']} - {p['num_molecules']}分子 - {p['created_at']}" 
                      for p in embedding_projects]
    selected_project_idx = st.selectbox("选择嵌入项目", range(len(project_options)), 
                                       format_func=lambda x: project_options[x])
    
    if selected_project_idx is not None:
        selected_project = embedding_projects[selected_project_idx]
        
        # 加载建模数据
        modeling_data = user_data.load_modeling_data(selected_project['project_id'])
        
        if modeling_data is not None:
            st.write("数据预览：")
            st.dataframe(modeling_data.head())
            
            # 提取特征和标签
            feature_cols = [col for col in modeling_data.columns if col not in ['SMILES', 'Label']]
            X = modeling_data[feature_cols].values
            y = modeling_data['Label'].values
            
            st.success(f"加载了 {len(X)} 个样本，特征维度: {X.shape[1]}")
            
            # 进入模型训练
            render_model_training_section(X, y, user_data, selected_project['method'])

            # 如果训练结果存在，则显示它
            if 'last_training_result' in st.session_state:
                st.markdown("---")
                render_training_results(st.session_state.last_training_result)

            # 模型训练完成后，独立渲染预测界面
            if st.session_state.get('model_trained', False):
                render_realtime_prediction_interface()

def render_model_training_section(X, y, user_data, method_name):
    """渲染模型训练部分"""
    st.subheader("🤖 步骤4: 机器学习建模")
    
    # 任务类型检测
    unique_labels = len(np.unique(y))
    if unique_labels <= 10:
        task_type = st.selectbox("任务类型", ["分类", "回归"], index=0)
    else:
        task_type = st.selectbox("任务类型", ["回归", "分类"], index=0)
    
    # 算法选择
    algorithm = st.selectbox(
        "选择算法",
        ["LightGBM", "XGBoost", "随机森林", "支持向量机"]
    )
    
    # 训练参数
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
    with col2:
        random_state = st.number_input("随机种子", value=42, min_value=0)
    
    # 开始训练
    if st.button("🚀 开始训练", use_container_width=True):
        with st.spinner("正在训练模型..."):
            from utils.model_utils import train_model
            
            try:
                # 训练模型
                result = train_model(X, y, algorithm, task_type, test_size)
                
                if result:
                    st.success("🎉 模型训练完成！")
                    
                    # 将训练好的模型和完整结果保存到session state
                    st.session_state.last_training_result = result
                    st.session_state.trained_model = result['model']
                    st.session_state.model_algorithm = algorithm
                    st.session_state.model_task_type = task_type
                    st.session_state.embedding_method = method_name
                    st.session_state.model_metrics = result['metrics']
                    st.session_state.model_trained = True
                    
                    # 重新运行脚本以显示结果和预测界面
                    st.rerun()
                    
            except Exception as e:
                st.error(f"训练失败: {str(e)}")
                st.error("请检查数据格式和参数设置")
    
    # 方法说明
    with st.expander("💡 方法说明"):
        st.info(f"""
        **嵌入方法**: {method_name}
        **算法说明**:
        - **LightGBM**: 基于梯度提升的高效算法，适合大规模数据
        - **XGBoost**: 经典的梯度提升算法，性能优秀
        - **随机森林**: 集成学习算法，对过拟合不敏感
        - **支持向量机**: 适合小样本数据，有良好的泛化能力
        """)

def render_training_results(result):
    """渲染模型训练的评估结果"""
    st.subheader("📈 模型评估结果")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**关键指标**:")
        for metric, value in result['metrics'].items():
            st.metric(metric, f"{value:.4f}")
    
    with col2:
        if 'confusion_matrix' in result:
            st.write("**混淆矩阵**:")
            st.dataframe(result['confusion_matrix'])

    # ROC曲线（仅分类任务）
    if result.get('task_type') == "分类" and 'roc_data' in result:
        st.subheader("📈 ROC曲线")
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        roc_data = result['roc_data']
        ax.plot(roc_data['fpr'], roc_data['tpr'], color='blue', lw=2, 
               label=f'ROC Curve (AUC = {roc_data["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

def render_realtime_prediction_interface():
    """渲染实时预测界面"""
    # 检查是否有训练好的模型
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ 没有找到训练好的模型，请先训练模型")
        return
    
    st.subheader("🧪 实时分子预测")
    st.info("💡 模型已训练完成，您可以直接输入SMILES进行预测")
    
    # 显示模型信息
    with st.expander("📋 当前模型信息"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**算法**: {st.session_state.get('model_algorithm', 'Unknown')}")
            st.write(f"**任务类型**: {st.session_state.get('model_task_type', 'Unknown')}")
            st.write(f"**嵌入方法**: {st.session_state.get('embedding_method', 'Unknown')}")
        with col2:
            st.write("**模型性能**:")
            metrics = st.session_state.get('model_metrics', {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    st.write(f"- {metric}: {value:.4f}")
                else:
                    st.write(f"- {metric}: {value}")
    
    # 预测输入选项
    prediction_mode = st.radio(
        "选择预测方式",
        ["单个分子预测", "批量分子预测"],
        horizontal=True,
        key="prediction_mode_radio"
    )
    
    if prediction_mode == "单个分子预测":
        render_single_molecule_prediction()
    else:
        render_batch_molecule_prediction()

def render_single_molecule_prediction():
    """渲染单个分子预测界面"""
    st.write("### 单个分子预测")
    
    # 检查模型是否存在
    if 'trained_model' not in st.session_state:
        st.error("❌ 没有找到训练好的模型，请先训练模型")
        return
    
    # 使用表单来处理输入和提交
    with st.form(key='single_prediction_form'):
        smiles_input = st.text_input(
            "输入SMILES字符串", 
            value=st.session_state.get('prediction_smiles', ''),
            placeholder="例如: CCO (乙醇)",
            key="prediction_input"
        )
        
        submitted = st.form_submit_button("🔍 开始预测", type="primary", use_container_width=True)
        
        if submitted:
            if smiles_input.strip():
                # 执行预测
                result = perform_prediction(smiles_input.strip())
                if result:
                    st.session_state.prediction_result = result
                else:
                    # 如果预测失败，清除旧结果
                    if 'prediction_result' in st.session_state:
                        del st.session_state.prediction_result
            else:
                st.warning("请输入有效的SMILES字符串")

    # 在表单外部显示结果，避免结果在下次提交时消失
    if 'prediction_result' in st.session_state:
        display_prediction_result(st.session_state.prediction_result)

def render_batch_molecule_prediction():
    """渲染批量分子预测界面"""
    st.write("### 批量分子预测")
    
    if 'trained_model' not in st.session_state:
        st.error("❌ 没有找到训练好的模型，请先训练模型")
        return

    # 使用表单来处理输入和提交
    with st.form(key='batch_prediction_form'):
        smiles_batch = st.text_area(
            "输入多个SMILES（每行一个）",
            value=st.session_state.get('batch_smiles', ''),
            placeholder="CCO\nC1=CC=CC=C1\n...",
            height=150,
            help="每行输入一个SMILES字符串，最多支持50个分子",
            key="batch_input"
        )
        
        submitted = st.form_submit_button("🔍 批量预测", type="primary", use_container_width=True)
        
        if submitted:
            if smiles_batch.strip():
                smiles_list = [s.strip() for s in smiles_batch.strip().split('\n') if s.strip()]
                
                if not smiles_list:
                    st.warning("请输入有效的SMILES字符串")
                elif len(smiles_list) > 50:
                    st.warning("⚠️ 批量预测最多支持50个分子，已截取前50个")
                    smiles_list = smiles_list[:50]
                
                if smiles_list:
                    # 执行批量预测
                    result = perform_batch_prediction(smiles_list)
                    if result:
                        st.session_state.batch_prediction_result = result
                    else:
                        # 如果预测失败，清除旧结果
                        if 'batch_prediction_result' in st.session_state:
                            del st.session_state.batch_prediction_result
            else:
                st.warning("请输入有效的SMILES字符串")

    # 在表单外部显示结果
    if 'batch_prediction_result' in st.session_state:
        display_batch_prediction_result(st.session_state.batch_prediction_result)

def perform_prediction(smiles):
    """执行分子预测
    
    完整流程：
    1. 从session_state获取训练好的模型
    2. 从session_state获取训练时使用的嵌入方法
    3. 使用相同的嵌入方法为新分子生成嵌入
    4. 使用训练好的模型对嵌入进行预测
    """
    try:
        # 步骤1: 获取训练好的模型
        trained_model = st.session_state.get('trained_model')
        if trained_model is None:
            st.error("❌ 没有找到训练好的模型，请先训练模型")
            return None
        
        # 步骤2: 获取训练时使用的嵌入方法
        embedding_method = st.session_state.get('embedding_method')
        if not embedding_method:
            st.error("❌ 没有找到嵌入方法信息，请重新训练模型")
            return None
        
        # 获取其他模型信息
        task_type = st.session_state.get('model_task_type', '')
        algorithm = st.session_state.get('model_algorithm', 'Unknown')
        
        with st.spinner("正在提取分子嵌入并预测..."):
            # 步骤3: 使用训练时相同的嵌入方法为新分子生成嵌入
            from utils.embedding_utils import extract_embeddings
            embeddings = extract_embeddings([smiles], embedding_method)
            
            if embeddings is None or len(embeddings) == 0:
                st.error("❌ 分子嵌入提取失败，请检查SMILES格式是否正确")
                return None
            
            # 验证嵌入维度是否与模型期望一致
            if hasattr(trained_model, 'n_features_in_'):
                expected_features = trained_model.n_features_in_
                actual_features = embeddings.shape[1]
                if expected_features != actual_features:
                    st.error(f"❌ 特征维度不匹配：模型期望{expected_features}维，实际{actual_features}维")
                    return None
            
            # 步骤4: 使用训练好的模型进行预测
            prediction = trained_model.predict(embeddings)[0]
            
            # 获取预测概率（如果是分类任务）
            probabilities = None
            confidence = None
            if task_type == "分类" and hasattr(trained_model, 'predict_proba'):
                probabilities = trained_model.predict_proba(embeddings)[0]
                confidence = np.max(probabilities)
        
        # 返回预测结果
        return {
            'smiles': smiles,
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': confidence,
            'task_type': task_type,
            'algorithm': algorithm,
            'embedding_method': embedding_method
        }
        
    except Exception as e:
        st.error(f"❌ 预测失败: {str(e)}")
        import traceback
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None

def display_prediction_result(result):
    """显示单个分子预测结果"""
    st.subheader("💡 预测结果")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**输入分子**:")
        st.code(result['smiles'], language='smiles')
        
        # 显示分子结构
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(result['smiles'])
        if mol:
            img = Draw.MolToImage(mol, size=(250, 200))
            st.image(img)

    with col2:
        st.metric("预测标签", str(result['prediction']))
        if result.get('confidence'):
            st.metric("置信度", f"{result['confidence']:.2%}")
        
        # 可视化预测概率（如果是分类任务）
        if result.get('probabilities') is not None:
            st.write("**预测概率分布**:")
            prob_df = pd.DataFrame({
                '类别': [f'类别 {i}' for i in range(len(result['probabilities']))],
                '概率': result['probabilities']
            })
            st.dataframe(prob_df.style.format({'概率': '{:.2%}'}))

def perform_batch_prediction(smiles_list):
    """执行批量分子预测
    
    完整流程：
    1. 从session_state获取训练好的模型
    2. 从session_state获取训练时使用的嵌入方法
    3. 使用相同的嵌入方法为新分子列表生成嵌入
    4. 使用训练好的模型对嵌入进行批量预测
    """
    try:
        # 步骤1: 获取训练好的模型
        trained_model = st.session_state.get('trained_model')
        if trained_model is None:
            st.error("❌ 没有找到训练好的模型，请先训练模型")
            return None
        
        # 步骤2: 获取训练时使用的嵌入方法
        embedding_method = st.session_state.get('embedding_method')
        if not embedding_method:
            st.error("❌ 没有找到嵌入方法信息，请重新训练模型")
            return None
        
        # 获取其他模型信息
        task_type = st.session_state.get('model_task_type', '')
        algorithm = st.session_state.get('model_algorithm', 'Unknown')
        
        # 显示进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"正在提取 {len(smiles_list)} 个分子的嵌入...")
        
        # 步骤3: 使用训练时相同的嵌入方法为新分子列表生成嵌入
        from utils.embedding_utils import extract_embeddings
        embeddings = extract_embeddings(smiles_list, embedding_method)
        
        if embeddings is None or len(embeddings) == 0:
            st.error("❌ 分子嵌入提取失败，请检查SMILES格式是否正确")
            return None
        
        # 验证嵌入维度是否与模型期望一致
        if hasattr(trained_model, 'n_features_in_'):
            expected_features = trained_model.n_features_in_
            actual_features = embeddings.shape[1]
            if expected_features != actual_features:
                st.error(f"❌ 特征维度不匹配：模型期望{expected_features}维，实际{actual_features}维")
                return None
        
        progress_bar.progress(50)
        status_text.text(f"正在预测 {len(smiles_list)} 个分子...")
        
        # 步骤4: 使用训练好的模型进行批量预测
        predictions = trained_model.predict(embeddings)
        
        # 获取预测概率（如果是分类任务）
        probabilities = None
        confidence_scores = None
        if task_type == "分类" and hasattr(trained_model, 'predict_proba'):
            probabilities = trained_model.predict_proba(embeddings)
            confidence_scores = np.max(probabilities, axis=1)
        
        progress_bar.progress(100)
        status_text.text("预测完成!")
        
        # 返回预测结果
        return {
            'smiles_list': smiles_list,
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_scores': confidence_scores,
            'task_type': task_type,
            'algorithm': algorithm,
            'embedding_method': embedding_method
        }
        
    except Exception as e:
        st.error(f"❌ 批量预测失败: {str(e)}")
        import traceback
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None

def display_batch_prediction_result(result):
    """显示批量分子预测结果"""
    st.subheader("💡 批量预测结果")
    
    smiles_list = result['smiles_list']
    predictions = result['predictions']
    
    result_data = {
        'SMILES': smiles_list,
        '预测标签': predictions
    }
    
    if result.get('confidence_scores') is not None:
        result_data['置信度'] = result['confidence_scores']
    
    result_df = pd.DataFrame(result_data)
    
    # 显示结果表格
    st.dataframe(result_df.style.format({'置信度': '{:.2%}'}))
    
    # 结果摘要
    st.subheader("📊 结果摘要")
    if result.get('task_type') == '回归':
        st.write("**预测值统计**:")
        st.dataframe(result_df['预测标签'].describe())
    else: # 分类
        st.write("**预测类别分布**:")
        st.dataframe(result_df['预测标签'].value_counts())

def render_project_viewer():
    """渲染项目查看页面"""
    st.title("📁 个人项目管理")
    
    # 获取用户数据管理器
    from utils.user_data_utils import UserDataManager
    user_data = UserDataManager(st.session_state.username)
    
    # 获取用户项目
    projects = user_data.get_user_projects()
    
    # 项目统计
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("嵌入项目", len(projects['embeddings']))
    with col2:
        st.metric("模型项目", len(projects['models']))
    with col3:
        st.metric("上传文件", len(projects['uploads']))
    with col4:
        total_size = sum(f['size'] for f in projects['uploads'])
        st.metric("存储空间", f"{total_size/1024/1024:.1f} MB")
    
    # 项目详情展示
    tab1, tab2, tab3 = st.tabs(["🔬 嵌入项目", "🤖 模型项目", "📁 上传文件"])
    
    with tab1:
        st.subheader("分子嵌入项目")
        if projects['embeddings']:
            for proj in projects['embeddings']:
                with st.expander(f"{proj['project_name']} - {proj['method']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**项目名称**: {proj['project_name']}")
                        st.write(f"**嵌入方法**: {proj['method']}")
                        st.write(f"**分子数量**: {proj['num_molecules']}")
                        st.write(f"**创建时间**: {proj['created_at']}")
                    
                    with col2:
                        if st.button(f"删除", key=f"del_emb_{proj['project_id']}"):
                            if user_data.delete_project('embeddings', proj['project_id']):
                                st.success("项目已删除")
                                st.rerun()
                            else:
                                st.error("删除失败")
        else:
            st.info("暂无嵌入项目")
    
    with tab2:
        st.subheader("机器学习模型")
        st.info("💡 目前平台采用实时预测模式，模型训练完成后可立即进行预测，无需保存模型文件")
        st.write("**实时预测的优势**:")
        st.write("- ✅ 无需担心模型保存和加载的兼容性问题")
        st.write("- ✅ 训练完成立即可用，操作更简单")
        st.write("- ✅ 避免存储空间占用")
        st.write("- ✅ 始终使用最新训练的模型")
        
        st.markdown("---")
        st.write("如需进行预测，请前往 **🧠 智能建模** 页面训练模型，训练完成后即可直接预测。")
    
    with tab3:
        st.subheader("上传文件")
        if projects['uploads']:
            for file_info in projects['uploads']:
                with st.expander(f"{file_info['filename']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**文件大小**: {file_info['size']/1024:.1f} KB")
                        st.write(f"**上传时间**: {file_info['created_at']}")
                    
                    with col2:
                        # 下载按钮
                        with open(file_info['filepath'], 'rb') as f:
                            st.download_button(
                                label="下载",
                                data=f.read(),
                                file_name=file_info['filename'],
                                key=f"download_{file_info['filename']}"
                            )
        else:
            st.info("暂无上传文件")
    
    # 导出功能
    st.subheader("📤 数据导出")
    if st.button("导出项目摘要"):
        summary = user_data.export_project_summary()
        
        import json
        json_str = json.dumps(summary, ensure_ascii=False, indent=2)
        
        st.download_button(
            label="下载项目摘要 (JSON)",
            data=json_str,
            file_name=f"{st.session_state.username}_projects_summary.json",
            mime="application/json"
        )

def render_literature_mining():
    """渲染文献挖掘页面"""
    st.markdown('<h1 class="main-title">📚 文献挖掘</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d; margin-bottom: 2rem;">基于文献数据库的智能检索与分析系统</p>', unsafe_allow_html=True)
    
    # 导入文献挖掘工具
    from utils.literature_utils import LiteratureMiner, export_to_bibtex
    import json
    import requests
    
    # 初始化文献挖掘器
    if 'literature_miner' not in st.session_state:
        st.session_state.literature_miner = LiteratureMiner()
    
    # 搜索策略选择
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_mode = st.selectbox(
            "🔍 选择搜索模式",
            ["综合搜索 (推荐)", "PubMed专项", "CrossRef专项", "Semantic Scholar专项"],
            help="综合搜索会同时使用多个数据库并去重，获得更全面的结果"
        )
    
    with col2:
        max_results = st.slider("每个数据库最大结果数", 5, 50, 15)
    
    with col3:
        enable_suggestions = st.checkbox("启用搜索建议", value=True)
    
    # 搜索输入
    query = st.text_input(
        "🔎 输入搜索关键词", 
        placeholder="例如: QSAR machine learning drug discovery",
        help="支持英文关键词搜索，可以使用布尔操作符 AND, OR, NOT"
    )
    
    # 搜索建议
    if enable_suggestions and query and len(query) > 3:
        suggestions = st.session_state.literature_miner.generate_search_suggestions(query)
        if suggestions:
            st.markdown("💡 **搜索建议**:")
            suggestion_cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                with suggestion_cols[i]:
                    if st.button(f"🔍 {suggestion[:30]}...", key=f"suggestion_{i}"):
                        query = suggestion
                        st.rerun()
    
    # 高级搜索选项
    with st.expander("🔧 高级搜索选项"):
        col1, col2 = st.columns(2)
        with col1:
            year_filter = st.text_input(
                "年份筛选", 
                placeholder="例如: 2020:2024[dp] (PubMed格式)",
                help="PubMed支持日期范围格式，如 2020:2024[dp]"
            )
        with col2:
            additional_terms = st.text_input(
                "附加搜索词", 
                placeholder="例如: review, meta-analysis",
                help="添加额外的搜索限定词"
            )
    
    # 构建最终查询
    final_query = query
    if year_filter:
        final_query += f" AND {year_filter}"
    if additional_terms:
        final_query += f" AND {additional_terms}"
    
    # 搜索按钮
    if st.button("🚀 开始搜索", type="primary"):
        if query:
            try:
                # 根据搜索模式执行搜索
                if search_mode == "综合搜索 (推荐)":
                    results = st.session_state.literature_miner.comprehensive_search(final_query, max_results)
                elif search_mode == "PubMed专项":
                    with st.spinner("正在搜索PubMed数据库..."):
                        results = st.session_state.literature_miner.search_pubmed(final_query, max_results * 3)
                elif search_mode == "CrossRef专项":
                    with st.spinner("正在搜索CrossRef数据库..."):
                        results = st.session_state.literature_miner.search_crossref(final_query, max_results * 3)
                else:  # Semantic Scholar专项
                    with st.spinner("正在搜索Semantic Scholar数据库..."):
                        results = st.session_state.literature_miner.search_semantic_scholar(final_query, max_results * 3)
                
                if results:
                    st.success(f"🎉 找到 {len(results)} 篇相关文献")
                    
                    # 数据源统计
                    source_counts = pd.Series([r['source'] for r in results]).value_counts()
                    st.markdown("### 📊 数据源分布")
                    source_cols = st.columns(len(source_counts))
                    for i, (source, count) in enumerate(source_counts.items()):
                        with source_cols[i]:
                            st.metric(source, count)
                    
                    # 搜索结果统计
                    st.markdown("### 📈 搜索结果分析")
                    
                    # 创建结果DataFrame
                    df_results = pd.DataFrame(results)
                    
                    # 统计图表
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # 年份分布
                        year_counts = df_results['year'].value_counts().sort_index()
                        fig_year = plt.figure(figsize=(8, 4))
                        plt.bar(year_counts.index, year_counts.values, color='skyblue', alpha=0.7)
                        plt.title('Publication Year Distribution')
                        plt.xlabel('Year')
                        plt.ylabel('Number of Papers')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_year)
                        plt.close()
                    
                    with col2:
                        # 期刊分布
                        journal_counts = df_results['journal'].value_counts().head(10)
                        fig_journal = plt.figure(figsize=(8, 6))
                        plt.barh(range(len(journal_counts)), journal_counts.values, color='lightcoral', alpha=0.7)
                        plt.yticks(range(len(journal_counts)), journal_counts.index)
                        plt.title('Top Journals Distribution')
                        plt.xlabel('Number of Papers')
                        plt.tight_layout()
                        st.pyplot(fig_journal)
                        plt.close()
                    
                    with col3:
                        # 引用数分布
                        citation_ranges = ['0-10', '11-50', '51-100', '100+']
                        citation_counts = [
                            len([r for r in results if 0 <= r['citations'] <= 10]),
                            len([r for r in results if 11 <= r['citations'] <= 50]),
                            len([r for r in results if 51 <= r['citations'] <= 100]),
                            len([r for r in results if r['citations'] > 100])
                        ]
                        
                        fig_citations = plt.figure(figsize=(8, 4))
                        plt.bar(citation_ranges, citation_counts, color='lightgreen', alpha=0.7)
                        plt.title('Citation Count Distribution')
                        plt.xlabel('Citation Range')
                        plt.ylabel('Number of Papers')
                        plt.tight_layout()
                        st.pyplot(fig_citations)
                        plt.close()
                    
                    # 详细结果展示
                    st.markdown("### 📄 详细搜索结果")
                    
                    # 排序选项
                    sort_option = st.selectbox(
                        "排序方式", 
                        ["按引用数降序", "按年份降序", "按影响因子降序", "按相关性"],
                        key="sort_results"
                    )
                    
                    if sort_option == "按引用数降序":
                        results.sort(key=lambda x: x['citations'], reverse=True)
                    elif sort_option == "按年份降序":
                        results.sort(key=lambda x: x['year'], reverse=True)
                    elif sort_option == "按影响因子降序":
                        results.sort(key=lambda x: x['impact_factor'], reverse=True)
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"📑 {i}. {result['title'][:80]}... [{result['source']}]"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**标题**: {result['title']}")
                                st.markdown(f"**作者**: {', '.join(result['authors'][:5])}{'...' if len(result['authors']) > 5 else ''}")
                                st.markdown(f"**期刊**: {result['journal']}")
                                st.markdown(f"**年份**: {result['year']}")
                                if result['abstract']:
                                    st.markdown(f"**摘要**: {result['abstract'][:400]}...")
                                if result['doi']:
                                    st.markdown(f"**DOI**: [{result['doi']}](https://doi.org/{result['doi']})")
                                if result['pmid']:
                                    st.markdown(f"**PMID**: [{result['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{result['pmid']}/)")
                            
                            with col2:
                                st.metric("引用数", result['citations'])
                                st.metric("影响因子", f"{result['impact_factor']:.2f}")
                                st.markdown(f"**数据源**: {result['source']}")
                    
                    # 导出选项
                    st.markdown("### 💾 导出结果")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # CSV导出
                        csv_data = df_results.to_csv(index=False)
                        st.download_button(
                            label="📊 下载CSV格式",
                            data=csv_data,
                            file_name=f"literature_search_{query.replace(' ', '_')[:20]}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # BibTeX导出
                        bibtex_data = export_to_bibtex(results)
                        st.download_button(
                            label="📚 下载BibTeX格式",
                            data=bibtex_data,
                            file_name=f"literature_search_{query.replace(' ', '_')[:20]}.bib",
                            mime="text/plain"
                        )
                    
                    with col3:
                        # JSON导出
                        json_data = json.dumps(results, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📋 下载JSON格式",
                            data=json_data,
                            file_name=f"literature_search_{query.replace(' ', '_')[:20]}.json",
                            mime="application/json"
                        )
                    

                    # 保存到个人项目
                    if st.button("💾 保存到个人项目"):
                        from utils.user_data_utils import UserDataManager
                        user_data = UserDataManager(st.session_state.username)
                        
                        project_data = {
                            'query': final_query,
                            'search_mode': search_mode,
                            'results_count': len(results),
                            'results': results,

                            'timestamp': pd.Timestamp.now().isoformat()
                        }
                        
                        filename = f"literature_search_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
                        filepath = user_data.save_project_data(filename, project_data, 'literature')
                        st.success(f"✅ 搜索结果已保存到: {filepath}")
                
                else:
                    st.warning("⚠️ 未找到相关文献，请尝试以下方法：")
                    st.markdown("""
                    - 使用更通用的关键词
                    - 检查拼写是否正确
                    - 尝试英文关键词
                    - 使用同义词或相关术语
                    - 减少搜索词的数量
                    """)
                    
            except Exception as e:
                st.error(f"❌ 搜索过程中出现错误: {str(e)}")
                st.markdown("""
                **可能的原因：**
                - 网络连接问题
                - API服务暂时不可用
                - 搜索查询格式不正确
                
                **建议：**
                - 检查网络连接
                - 稍后重试
                - 简化搜索关键词
                """)
        else:
            st.error("❌ 请输入搜索内容")
    
    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 🔍 搜索技巧
        
        **关键词选择：**
        - 使用英文关键词效果更佳
        - 可以使用布尔操作符：AND, OR, NOT
        - 使用引号包围短语："machine learning"
        
        **搜索模式：**
        - **综合搜索**：同时搜索多个数据库，结果更全面
        - **PubMed专项**：专注生物医学文献
        - **CrossRef专项**：覆盖更广泛的学科领域
        - **Semantic Scholar专项**：AI增强的学术搜索
        
        **高级功能：**
        - 年份筛选：使用PubMed格式 "2020:2024[dp]"
        - 搜索建议：系统会根据输入提供相关建议
        - 多种导出格式：CSV、BibTeX、JSON
        
        ### 📊 数据来源
        - **PubMed**: 美国国立医学图书馆生物医学数据库
        - **CrossRef**: 学术出版物DOI注册机构
        - **Semantic Scholar**: AI驱动的学术搜索引擎
        
        ### 🤖 AI摘要功能
        - **综合摘要**: 全面分析文献内容，总结研究方向和发现
        - **技术方法总结**: 重点分析技术方法和算法
        - **研究趋势分析**: 识别研究热点和发展趋势
        - **关键发现提取**: 提取核心发现和重要结论
        - **批量摘要**: 对单篇文献进行快速摘要
        - **研究问题生成**: 基于文献内容生成有价值的研究问题
        
        **支持模型**: Qwen/QwQ-32B, Qwen2.5系列等硅基流动平台模型
        """)
    
    # API状态检查
    with st.expander("🔧 API状态检查"):
        if st.button("检查API连接状态"):
            status_col1, status_col2, status_col3 = st.columns(3)
            
            with status_col1:
                try:
                    response = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ PubMed API 正常")
                    else:
                        st.error("❌ PubMed API 异常")
                except:
                    st.error("❌ PubMed API 连接失败")
            
            with status_col2:
                try:
                    response = requests.get("https://api.crossref.org/works?rows=1", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ CrossRef API 正常")
                    else:
                        st.error("❌ CrossRef API 异常")
                except:
                    st.error("❌ CrossRef API 连接失败")
            
            with status_col3:
                try:
                    response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search?query=test&limit=1", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Semantic Scholar API 正常")
                    else:
                        st.error("❌ Semantic Scholar API 异常")
                except:
                    st.error("❌ Semantic Scholar API 连接失败")

def render_ai_summary():
    """渲染AI摘要总结页面"""
    st.markdown('<h1 class="main-title">🤖 AI文献摘要总结</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d; margin-bottom: 2rem;">智能提炼文献核心要点，中文输出关键信息</p>', unsafe_allow_html=True)
    
    # 导入AI摘要工具
    from utils.ai_summary_utils import AISummaryGenerator, test_api_connection
    import json
    
    # 初始化AI摘要生成器
    if 'ai_summary_generator' not in st.session_state:
        st.session_state.ai_summary_generator = AISummaryGenerator()
    
    # API配置区域
    st.markdown("### ⚙️ API配置")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        api_token = st.text_input(
            "🔑 硅基流动API Token", 
            type="password",
            placeholder="请输入您的API Token",
            help="获取Token: https://cloud.siliconflow.cn/"
        )
    
    with col2:
        model_input_method = st.radio(
            "模型选择方式",
            ["预设模型", "自定义模型"],
            horizontal=True,
            help="选择使用预设模型还是自定义输入模型名称"
        )
    
    with col3:
        if model_input_method == "预设模型":
            selected_model = st.selectbox(
                "🧠 选择AI模型",
                st.session_state.ai_summary_generator.available_models,
                index=0,
                help="不同模型效果和速度有差异"
            )
        else:
            selected_model = st.text_input(
                "🧠 自定义模型名称",
                placeholder="例如：Qwen/Qwen2.5-7B-Instruct",
                help="输入完整的模型名称，如 Qwen/Qwen2.5-7B-Instruct"
            )
    
    # API连接测试
    if api_token and selected_model:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔧 测试API连接"):
                with st.spinner("测试连接中..."):
                    if test_api_connection(api_token, selected_model):
                        st.success("✅ API连接正常")
                    else:
                        st.error("❌ API连接失败，请检查Token和模型名称")
        with col2:
            if selected_model:
                st.info(f"当前模型: {selected_model}")
    elif api_token and not selected_model:
        st.warning("⚠️ 请选择或输入模型名称")
    elif not api_token and selected_model:
        st.warning("⚠️ 请输入API Token")
    
    st.markdown("---")
    
    # 摘要输入区域
    st.markdown("### 📝 文献摘要输入")
    
    # 输入方式选择
    input_method = st.radio(
        "选择输入方式",
        ["📄 单篇摘要", "📚 批量摘要", "📋 从剪贴板粘贴"],
        horizontal=True
    )
    
    abstracts_to_process = []
    
    if input_method == "📄 单篇摘要":
        st.markdown("#### 输入单篇文献摘要")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            title = st.text_input("文献标题 (可选)", placeholder="例如：基于深度学习的QSAR模型研究")
        with col2:
            authors = st.text_input("作者 (可选)", placeholder="例如：张三, 李四")
        
        abstract_text = st.text_area(
            "文献摘要",
            height=200,
            placeholder="请粘贴文献摘要内容...",
            help="支持中英文摘要，建议字数在100-2000字之间"
        )
        
        if abstract_text.strip():
            abstracts_to_process.append({
                'title': title or "未提供标题",
                'authors': [authors] if authors else [],
                'abstract': abstract_text.strip(),
                'year': 2024,
                'journal': "用户输入",
                'citations': 0
            })
    
    elif input_method == "📚 批量摘要":
        st.markdown("#### 批量输入多篇摘要")
        st.info("💡 每行一篇摘要，或用空行分隔多篇摘要")
        
        batch_text = st.text_area(
            "批量摘要输入",
            height=300,
            placeholder="摘要1：这是第一篇文献的摘要内容...\n\n摘要2：这是第二篇文献的摘要内容...\n\n摘要3：这是第三篇文献的摘要内容...",
            help="支持多种分隔方式：空行、序号、或每行一篇"
        )
        
        if batch_text.strip():
            # 解析批量输入
            abstracts = []
            
            # 按空行分割
            parts = [part.strip() for part in batch_text.split('\n\n') if part.strip()]
            
            if len(parts) == 1:
                # 如果没有空行分割，尝试按行分割
                lines = [line.strip() for line in batch_text.split('\n') if line.strip()]
                if len(lines) > 1:
                    parts = lines
            
            for i, part in enumerate(parts, 1):
                # 移除可能的序号
                clean_text = part
                if part.startswith(f"{i}.") or part.startswith(f"{i}、") or part.startswith(f"摘要{i}"):
                    clean_text = part.split('：', 1)[-1].split(':', 1)[-1].strip()
                
                if len(clean_text) > 20:  # 过滤太短的文本
                    abstracts_to_process.append({
                        'title': f"批量输入文献 {i}",
                        'authors': [],
                        'abstract': clean_text,
                        'year': 2024,
                        'journal': "用户输入",
                        'citations': 0
                    })
    
    else:  # 从剪贴板粘贴
        st.markdown("#### 从剪贴板粘贴")
        st.info("💡 直接粘贴从其他地方复制的摘要内容")
        
        clipboard_text = st.text_area(
            "粘贴摘要内容",
            height=250,
            placeholder="Ctrl+V 粘贴摘要内容...",
            help="支持从PDF、网页、文档等复制的内容"
        )
        
        if clipboard_text.strip():
            abstracts_to_process.append({
                'title': "剪贴板输入",
                'authors': [],
                'abstract': clipboard_text.strip(),
                'year': 2024,
                'journal': "用户输入",
                'citations': 0
            })
    
    # 显示待处理摘要数量
    if abstracts_to_process:
        st.success(f"✅ 已准备 {len(abstracts_to_process)} 篇摘要待处理")
        
        # 预览摘要
        with st.expander("👀 预览待处理摘要"):
            for i, abstract in enumerate(abstracts_to_process, 1):
                st.markdown(f"**{i}. {abstract['title']}**")
                st.markdown(f"摘要: {abstract['abstract'][:200]}...")
                st.markdown("---")
    
    st.markdown("---")
    
    # AI处理区域
    if abstracts_to_process and api_token and selected_model:
        st.markdown("### 🚀 AI智能分析")
        
        # 分析选项
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 提炼核心要点", type="primary"):
                try:
                    with st.spinner("AI正在分析摘要，提炼核心要点..."):
                        # 为单个摘要分析构建特殊提示词
                        if len(abstracts_to_process) == 1:
                            abstract = abstracts_to_process[0]
                            prompt = f"""
请对以下文献摘要进行智能分析，提炼核心要点，用中文输出：

标题：{abstract['title']}
摘要：{abstract['abstract']}

请按以下格式输出：

**🎯 核心研究内容**
- [提炼研究的核心内容和目标]

**🔬 主要方法**
- [总结使用的主要研究方法]

**📊 关键发现**
- [列出重要的研究发现和结果]

**💡 创新点**
- [指出研究的创新之处]

**🔍 研究意义**
- [说明研究的理论和实际意义]

要求：
1. 用简洁明了的中文表达
2. 突出最重要的信息
3. 每个要点控制在1-2句话
4. 避免重复原文表述
"""
                        else:
                            # 多个摘要的综合分析
                            abstracts_text = "\n\n".join([
                                f"文献{i+1}：{abs['title']}\n摘要：{abs['abstract']}"
                                for i, abs in enumerate(abstracts_to_process)
                            ])
                            
                            prompt = f"""
请对以下多篇文献摘要进行综合分析，提炼共同的核心要点，用中文输出：

{abstracts_text}

请按以下格式输出：

**🎯 共同研究主题**
- [总结这些文献的共同研究领域和主题]

**🔬 主要研究方法**
- [归纳使用的主要研究方法和技术]

**📊 重要发现汇总**
- [汇总各文献的重要发现]

**💡 技术创新点**
- [总结技术和方法上的创新]

**🔍 研究趋势**
- [分析体现的研究趋势和方向]

**🚀 未来展望**
- [基于这些研究提出未来发展方向]

要求：
1. 用简洁明了的中文表达
2. 突出共性和重要信息
3. 每个要点控制在1-2句话
4. 体现综合分析的价值
"""
                        
                        # 调用AI API
                        headers = {
                            'Authorization': f'Bearer {api_token}',
                            'Content-Type': 'application/json'
                        }
                        
                        data = {
                            "model": selected_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "max_tokens": 1500,
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                        
                        response = requests.post(
                            "https://api.siliconflow.cn/v1/chat/completions",
                            headers=headers,
                            json=data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if 'choices' in result and len(result['choices']) > 0:
                                summary = result['choices'][0]['message']['content']
                                
                                st.markdown("### 📋 AI分析结果")
                                st.markdown(f"**使用模型**: {selected_model}")
                                st.markdown(f"**分析文献数**: {len(abstracts_to_process)}")
                                st.markdown("---")
                                st.markdown(summary)
                                
                                # 保存结果
                                if 'ai_summaries' not in st.session_state:
                                    st.session_state.ai_summaries = []
                                
                                st.session_state.ai_summaries.append({
                                    'summary': summary,
                                    'model': selected_model,
                                    'type': '核心要点提炼',
                                    'input_count': len(abstracts_to_process),
                                    'timestamp': pd.Timestamp.now().isoformat()
                                })
                            else:
                                st.error("AI响应格式异常")
                        else:
                            st.error(f"API调用失败: {response.status_code}")
                            
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")
        
        with col2:
            if st.button("📈 研究方法分析"):
                try:
                    with st.spinner("AI正在分析研究方法..."):
                        abstracts_text = "\n\n".join([
                            f"文献{i+1}：{abs['abstract']}"
                            for i, abs in enumerate(abstracts_to_process)
                        ])
                        
                        prompt = f"""
请重点分析以下文献摘要中的研究方法，用中文输出：

{abstracts_text}

请按以下格式输出：

**🔬 主要研究方法**
- [列出使用的主要研究方法]

**📊 数据分析技术**
- [总结数据处理和分析技术]

**🛠️ 实验设计**
- [描述实验设计思路]

**📏 评估指标**
- [列出使用的评估指标和标准]

**⚡ 方法优势**
- [分析方法的优势和特点]

**🔄 改进空间**
- [指出可能的改进方向]

要求：简洁明了，突出方法特色，每点1-2句话。
"""
                        
                        headers = {
                            'Authorization': f'Bearer {api_token}',
                            'Content-Type': 'application/json'
                        }
                        
                        data = {
                            "model": selected_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "max_tokens": 1200,
                            "temperature": 0.7
                        }
                        
                        response = requests.post(
                            "https://api.siliconflow.cn/v1/chat/completions",
                            headers=headers,
                            json=data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if 'choices' in result and len(result['choices']) > 0:
                                analysis = result['choices'][0]['message']['content']
                                
                                st.markdown("### 🔬 研究方法分析")
                                st.markdown(analysis)
                                
                                # 保存结果
                                if 'ai_summaries' not in st.session_state:
                                    st.session_state.ai_summaries = []
                                
                                st.session_state.ai_summaries.append({
                                    'summary': analysis,
                                    'model': selected_model,
                                    'type': '研究方法分析',
                                    'input_count': len(abstracts_to_process),
                                    'timestamp': pd.Timestamp.now().isoformat()
                                })
                        
                except Exception as e:
                    st.error(f"方法分析失败: {str(e)}")
        
        with col3:
            if st.button("💡 创新点挖掘"):
                try:
                    with st.spinner("AI正在挖掘创新点..."):
                        abstracts_text = "\n\n".join([
                            f"文献{i+1}：{abs['abstract']}"
                            for i, abs in enumerate(abstracts_to_process)
                        ])
                        
                        prompt = f"""
请深度挖掘以下文献摘要中的创新点和亮点，用中文输出：

{abstracts_text}

请按以下格式输出：

**💡 技术创新**
- [识别技术方法上的创新]

**🎯 应用创新**
- [发现应用领域的创新]

**📊 数据创新**
- [分析数据处理的创新]

**🔍 理论贡献**
- [总结理论层面的贡献]

**🚀 实用价值**
- [评估实际应用价值]

**🌟 突破性发现**
- [识别突破性的发现]

要求：深度挖掘，突出创新性，避免泛泛而谈。
"""
                        
                        headers = {
                            'Authorization': f'Bearer {api_token}',
                            'Content-Type': 'application/json'
                        }
                        
                        data = {
                            "model": selected_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "max_tokens": 1200,
                            "temperature": 0.8
                        }
                        
                        response = requests.post(
                            "https://api.siliconflow.cn/v1/chat/completions",
                            headers=headers,
                            json=data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if 'choices' in result and len(result['choices']) > 0:
                                innovation = result['choices'][0]['message']['content']
                                
                                st.markdown("### 💡 创新点分析")
                                st.markdown(innovation)
                                
                                # 保存结果
                                if 'ai_summaries' not in st.session_state:
                                    st.session_state.ai_summaries = []
                                
                                st.session_state.ai_summaries.append({
                                    'summary': innovation,
                                    'model': selected_model,
                                    'type': '创新点挖掘',
                                    'input_count': len(abstracts_to_process),
                                    'timestamp': pd.Timestamp.now().isoformat()
                                })
                        
                except Exception as e:
                    st.error(f"创新点挖掘失败: {str(e)}")
    
    elif not api_token:
        st.warning("⚠️ 请先配置API Token才能使用AI分析功能")
    elif not selected_model:
        st.warning("⚠️ 请选择或输入模型名称")
    elif not abstracts_to_process:
        st.info("💡 请先输入文献摘要内容")
    
    # 历史分析结果
    if 'ai_summaries' in st.session_state and st.session_state.ai_summaries:
        st.markdown("---")
        st.markdown("### 📚 历史分析结果")
        
        # 按时间倒序显示
        for i, summary_data in enumerate(reversed(st.session_state.ai_summaries), 1):
            with st.expander(f"📋 分析结果 {i} - {summary_data['type']} ({summary_data['model']})"):
                st.markdown(f"**分析时间**: {summary_data['timestamp']}")
                st.markdown(f"**处理文献数**: {summary_data['input_count']}")
                st.markdown("**分析结果**:")
                st.markdown(summary_data['summary'])
    
    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 🤖 AI摘要功能说明
        
        **输入方式**：
        - **单篇摘要**: 输入单篇文献的详细信息
        - **批量摘要**: 一次性输入多篇摘要，支持多种分隔方式
        - **剪贴板粘贴**: 直接粘贴从其他地方复制的内容
        
        **模型选择**：
        - **预设模型**: 从下拉列表中选择预配置的模型
        - **自定义模型**: 手动输入完整的模型名称
        - **支持模型**: Qwen/QwQ-32B, Qwen/Qwen2.5-72B-Instruct, Qwen/Qwen3-8B 等
        
        **分析类型**：
        - **核心要点提炼**: 全面分析摘要，提炼最重要的信息
        - **研究方法分析**: 重点分析研究方法和技术路线
        - **创新点挖掘**: 深度挖掘研究的创新性和突破点
        
        **使用技巧**：
        - 摘要内容建议在100-2000字之间
        - 支持中英文摘要，AI会自动用中文输出
        - 批量处理时建议不超过5篇摘要
        - 可以多次使用不同分析类型获得全面洞察
        - 自定义模型时请确保模型名称正确
        
        **注意事项**：
        - 需要有效的硅基流动API Token
        - AI分析结果仅供参考，建议结合专业判断
        - 不同模型的分析效果可能有差异
        - 自定义模型需要确保在API平台上可用
        """)



# 旧的模拟函数已被真实API替代



if __name__ == "__main__":
    main() 