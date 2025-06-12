import requests
import json
import streamlit as st
from typing import List, Dict, Optional
import time
import random

class AISummaryGenerator:
    """AI文献摘要总结生成器"""
    
    def __init__(self):
        self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.default_model = "Qwen/QwQ-32B"
        self.available_models = [
            "Qwen/QwQ-32B",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen3-8B"
        ]
        self.last_request_time = 0
        self.min_request_interval = 2  # 最小请求间隔（秒）
    
    def _wait_for_rate_limit(self):
        """等待API限流间隔"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last_request
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _make_api_request(self, headers: dict, data: dict, max_retries: int = 3) -> dict:
        """
        带重试机制的API请求
        
        Args:
            headers: 请求头
            data: 请求数据
            max_retries: 最大重试次数
        
        Returns:
            API响应结果
        """
        for attempt in range(max_retries):
            try:
                # 等待限流间隔
                self._wait_for_rate_limit()
                
                # 发送请求
                response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
                
                # 处理不同的HTTP状态码
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        return result
                    else:
                        raise ValueError("API返回格式异常：缺少choices字段")
                
                elif response.status_code == 429:  # 限流
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # 指数退避
                    st.warning(f"⏳ API限流，等待 {wait_time:.1f} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 401:
                    raise Exception("API Token无效或已过期，请检查Token")
                
                elif response.status_code == 400:
                    error_msg = response.json().get('error', {}).get('message', '请求参数错误')
                    raise Exception(f"请求参数错误: {error_msg}")
                
                elif response.status_code == 500:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        st.warning(f"🔄 服务器错误，等待 {wait_time:.1f} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception("服务器内部错误，请稍后重试")
                
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    st.warning(f"⏰ 请求超时，正在重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
                    raise Exception("请求超时，请检查网络连接")
            
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    st.warning(f"🌐 网络连接错误，正在重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(3)
                    continue
                else:
                    raise Exception("网络连接失败，请检查网络设置")
            
            except json.JSONDecodeError as e:
                raise Exception(f"API响应解析失败: {str(e)}")
            
            except Exception as e:
                if attempt < max_retries - 1 and "API返回格式异常" not in str(e):
                    st.warning(f"🔄 请求失败，正在重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
                    raise e
        
        raise Exception(f"API请求失败，已重试 {max_retries} 次")
    
    def generate_summary(self, articles: List[Dict], api_token: str, 
                        model_name: str = None, summary_type: str = "综合摘要") -> str:
        """
        生成文献摘要
        
        Args:
            articles: 文献列表
            api_token: API令牌
            model_name: 模型名称
            summary_type: 摘要类型
        
        Returns:
            生成的摘要文本
        """
        if not api_token:
            raise ValueError("请提供有效的API令牌")
        
        if not articles:
            raise ValueError("没有文献可以总结")
        
        model = model_name or self.default_model
        
        # 构建提示词
        prompt = self._build_prompt(articles, summary_type)
        
        # 调用API
        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "max_tokens": 2048,
            "enable_thinking": False,
            "thinking_budget": 4096,
            "min_p": 0.05,
            "stop": None,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {
                "type": "text"
            }
        }
        
        try:
            result = self._make_api_request(headers, data)
            return result['choices'][0]['message']['content']
        except Exception as e:
            raise Exception(f"摘要生成失败: {str(e)}")
    
    def _build_prompt(self, articles: List[Dict], summary_type: str) -> str:
        """构建AI提示词"""
        
        # 准备文献信息
        literature_info = []
        for i, article in enumerate(articles[:10], 1):  # 限制最多10篇文献
            info = f"""
文献 {i}:
标题: {article.get('title', 'Unknown')}
作者: {', '.join(article.get('authors', [])[:3])}
期刊: {article.get('journal', 'Unknown')}
年份: {article.get('year', 'Unknown')}
摘要: {article.get('abstract', '无摘要')[:500]}...
引用数: {article.get('citations', 0)}
"""
            literature_info.append(info)
        
        literature_text = '\n'.join(literature_info)
        
        # 根据摘要类型构建不同的提示词
        if summary_type == "综合摘要":
            prompt = f"""
请对以下文献进行综合摘要分析，要求：

1. 总结主要研究领域和方向
2. 分析研究方法和技术趋势
3. 归纳主要发现和结论
4. 指出研究的创新点和局限性
5. 展望未来发展方向

文献信息：
{literature_text}

请用中文撰写一份结构清晰、逻辑严谨的综合摘要（800-1200字）。
"""
        
        elif summary_type == "技术方法总结":
            prompt = f"""
请重点分析以下文献中的技术方法，要求：

1. 总结主要使用的技术方法和算法
2. 分析方法的优缺点和适用场景
3. 比较不同方法的性能表现
4. 归纳技术发展趋势
5. 提出方法改进建议

文献信息：
{literature_text}

请用中文撰写一份专注于技术方法的分析报告（600-1000字）。
"""
        
        elif summary_type == "研究趋势分析":
            prompt = f"""
请分析以下文献反映的研究趋势，要求：

1. 识别研究热点和新兴方向
2. 分析研究方法的演进趋势
3. 总结技术发展的时间脉络
4. 预测未来可能的发展方向
5. 指出值得关注的研究机会

文献信息：
{literature_text}

请用中文撰写一份研究趋势分析报告（600-1000字）。
"""
        
        elif summary_type == "关键发现提取":
            prompt = f"""
请提取以下文献的关键发现和重要结论，要求：

1. 列出每篇文献的核心发现
2. 总结共同的重要结论
3. 识别有争议或矛盾的观点
4. 分析发现的科学意义
5. 评估结果的可靠性和影响

文献信息：
{literature_text}

请用中文撰写一份关键发现提取报告（600-1000字）。
"""
        
        else:  # 默认综合摘要
            prompt = f"""
请对以下文献进行智能摘要分析：

{literature_text}

请用中文撰写一份简洁明了的摘要（500-800字），包括：
1. 研究背景和目标
2. 主要方法和发现
3. 重要结论和意义
"""
        
        return prompt
    
    def batch_summarize_abstracts(self, articles: List[Dict], api_token: str, 
                                 model_name: str = None) -> List[str]:
        """
        批量总结单篇文献摘要
        """
        if not api_token:
            raise ValueError("请提供有效的API令牌")
        
        model = model_name or self.default_model
        summaries = []
        
        for article in articles:
            if not article.get('abstract'):
                summaries.append("该文献无摘要信息")
                continue
            
            prompt = f"""
请对以下文献摘要进行简洁总结，用中文回答（100-200字）：

标题: {article.get('title', 'Unknown')}
摘要: {article.get('abstract', '')}

总结要求：
1. 提取核心研究内容
2. 概括主要方法和发现
3. 突出创新点和意义
"""
            
            try:
                headers = {
                    'Authorization': f'Bearer {api_token}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.7
                }
                
                result = self._make_api_request(headers, data)
                summaries.append(result['choices'][0]['message']['content'])
                
            except Exception as e:
                summaries.append(f"摘要生成出错: {str(e)}")
                # 错误后等待更长时间
                time.sleep(3)
        
        return summaries
    
    def generate_research_questions(self, articles: List[Dict], api_token: str, 
                                  model_name: str = None) -> List[str]:
        """
        基于文献生成研究问题
        """
        if not api_token or not articles:
            return []
        
        model = model_name or self.default_model
        
        # 构建文献概要
        literature_summary = []
        for article in articles[:5]:  # 限制5篇文献
            summary = f"《{article.get('title', 'Unknown')}》- {article.get('abstract', '')[:200]}..."
            literature_summary.append(summary)
        
        prompt = f"""
基于以下文献，请生成5-8个有价值的研究问题，要求：

1. 问题应该具有科学意义和实用价值
2. 问题应该是可研究和可验证的
3. 问题应该体现当前研究的不足或空白
4. 问题应该具有一定的创新性

文献概要：
{chr(10).join(literature_summary)}

请用中文列出研究问题，每个问题一行，格式为：
1. 问题内容
2. 问题内容
...
"""
        
        try:
            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_tokens": 1024,
                "temperature": 0.8,
                "top_p": 0.9
            }
            
            result = self._make_api_request(headers, data)
            content = result['choices'][0]['message']['content']
            
            # 解析研究问题
            questions = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # 移除序号
                    question = line.split('.', 1)[-1].strip() if '.' in line else line
                    question = question.lstrip('-•').strip()
                    if question:
                        questions.append(question)
            return questions
            
        except Exception as e:
            st.error(f"研究问题生成失败: {str(e)}")
        
        return []

def test_api_connection(api_token: str, model_name: str = "Qwen/QwQ-32B") -> bool:
    """测试API连接"""
    if not api_token:
        return False
    
    try:
        # 创建临时的生成器实例进行测试
        temp_generator = AISummaryGenerator()
        
        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": "测试连接"}],
            "stream": False,
            "max_tokens": 10,
            "temperature": 0.7
        }
        
        # 使用带重试机制的请求方法
        result = temp_generator._make_api_request(headers, data, max_retries=1)
        return True
        
    except Exception as e:
        st.error(f"连接测试失败: {str(e)}")
        return False 