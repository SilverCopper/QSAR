"""
用户认证工具模块
支持用户注册、登录、密码验证等功能
"""

import sqlite3
import hashlib
import os
import streamlit as st
from datetime import datetime
import re

class UserAuth:
    """用户认证类"""
    
    def __init__(self, db_path="database/users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_username(self, username):
        """验证用户名格式"""
        if not username:
            return False, "用户名不能为空"
        if len(username) < 3:
            return False, "用户名至少3个字符"
        if len(username) > 20:
            return False, "用户名最多20个字符"
        if not re.match("^[a-zA-Z0-9_]+$", username):
            return False, "用户名只能包含字母、数字和下划线"
        return True, ""
    
    def validate_password(self, password):
        """验证密码格式"""
        if not password:
            return False, "密码不能为空"
        if len(password) < 6:
            return False, "密码至少6个字符"
        if len(password) > 50:
            return False, "密码最多50个字符"
        return True, ""
    
    def validate_email(self, email):
        """验证邮箱格式"""
        if not email:
            return True, ""  # 邮箱可选
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return False, "邮箱格式不正确"
        return True, ""
    
    def register_user(self, username, password, email=""):
        """用户注册"""
        # 验证输入
        valid, msg = self.validate_username(username)
        if not valid:
            return False, msg
        
        valid, msg = self.validate_password(password)
        if not valid:
            return False, msg
        
        valid, msg = self.validate_email(email)
        if not valid:
            return False, msg
        
        # 检查用户名是否已存在
        if self.user_exists(username):
            return False, "用户名已存在"
        
        # 创建用户
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        
        try:
            cursor.execute('''
                INSERT INTO users (username, password_hash, email)
                VALUES (?, ?, ?)
            ''', (username, password_hash, email))
            
            conn.commit()
            
            # 创建用户个人文件夹
            self.create_user_folders(username)
            
            return True, "注册成功"
        
        except sqlite3.Error as e:
            return False, f"注册失败: {e}"
        
        finally:
            conn.close()
    
    def login_user(self, username, password):
        """用户登录"""
        if not username or not password:
            return False, "用户名和密码不能为空"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        
        cursor.execute('''
            SELECT id, username FROM users
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        
        if user:
            # 更新最后登录时间
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE username = ?
            ''', (username,))
            conn.commit()
            conn.close()
            return True, "登录成功"
        else:
            conn.close()
            return False, "用户名或密码错误"
    
    def user_exists(self, username):
        """检查用户是否存在"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def get_user_info(self, username):
        """获取用户信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, email, created_at, last_login
            FROM users WHERE username = ?
        ''', (username,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'username': result[0],
                'email': result[1] or '',
                'created_at': result[2],
                'last_login': result[3]
            }
        return None
    
    def create_user_folders(self, username):
        """创建用户个人文件夹"""
        base_path = f"users/{username}"
        folders = ['projects', 'embeddings', 'models', 'uploads']
        
        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            os.makedirs(folder_path, exist_ok=True)
        
        # 创建用户配置文件
        config_path = os.path.join(base_path, 'config.json')
        if not os.path.exists(config_path):
            import json
            config = {
                'created_at': datetime.now().isoformat(),
                'settings': {
                    'default_embedding_method': 'RDKit 指纹',
                    'default_algorithm': 'LightGBM'
                }
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

def render_login_page():
    """渲染登录页面"""
    st.markdown("""
    <style>
    .login-form {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">🧬 QSAR 预测平台</div>', unsafe_allow_html=True)
    
    # 初始化认证器
    auth = UserAuth()
    
    # 创建居中的登录表单
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # 登录/注册选择
        auth_mode = st.radio("选择操作", ["登录", "注册"], horizontal=True)
        
        if auth_mode == "登录":
            st.subheader("用户登录")
            
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            
            if st.button("登录", use_container_width=True):
                if username and password:
                    success, message = auth.login_user(username, password)
                    if success:
                        st.success(message)
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.current_page = 'home'
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("请填写所有字段")
        
        else:  # 注册
            st.subheader("用户注册")
            
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            password_confirm = st.text_input("确认密码", type="password")
            email = st.text_input("邮箱（可选）")
            
            if st.button("注册", use_container_width=True):
                if username and password and password_confirm:
                    if password != password_confirm:
                        st.error("两次输入的密码不一致")
                    else:
                        success, message = auth.register_user(username, password, email)
                        if success:
                            st.success(message)
                            st.info("请使用新账户登录")
                        else:
                            st.error(message)
                else:
                    st.error("请填写必填字段")

def logout_user():
    """用户登出"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.current_page = 'login'
    st.rerun() 