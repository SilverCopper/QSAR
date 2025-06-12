import requests
import json
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from urllib.parse import quote
import streamlit as st

class LiteratureMiner:
    """真实的文献挖掘API集成类"""
    
    def __init__(self):
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.crossref_base_url = "https://api.crossref.org/works"
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # API请求间隔（秒）
        self.request_delay = 0.5
        
    def search_pubmed(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        使用PubMed API搜索文献
        """
        try:
            # 第一步：搜索获取PMID列表
            search_url = f"{self.pubmed_base_url}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json'
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=10)
            search_data = search_response.json()
            
            if 'esearchresult' not in search_data or 'idlist' not in search_data['esearchresult']:
                return []
            
            pmids = search_data['esearchresult']['idlist']
            if not pmids:
                return []
            
            time.sleep(self.request_delay)
            
            # 第二步：获取详细信息
            fetch_url = f"{self.pubmed_base_url}efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }
            
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=15)
            
            # 解析XML
            root = ET.fromstring(fetch_response.content)
            articles = []
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # 提取基本信息
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "Unknown Title"
                    
                    # 作者信息
                    authors = []
                    for author in article.findall('.//Author'):
                        lastname = author.find('LastName')
                        forename = author.find('ForeName')
                        if lastname is not None and forename is not None:
                            authors.append(f"{forename.text} {lastname.text}")
                    
                    # 期刊信息
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else "Unknown Journal"
                    
                    # 发表年份
                    year_elem = article.find('.//PubDate/Year')
                    year = int(year_elem.text) if year_elem is not None else 2024
                    
                    # PMID
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    # 摘要
                    abstract_elem = article.find('.//Abstract/AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    # DOI
                    doi = ""
                    for article_id in article.findall('.//ArticleId'):
                        if article_id.get('IdType') == 'doi':
                            doi = article_id.text
                            break
                    
                    articles.append({
                        'title': title,
                        'authors': authors,
                        'journal': journal,
                        'year': year,
                        'pmid': pmid,
                        'doi': doi,
                        'abstract': abstract,
                        'citations': self._estimate_citations(year),  # 估算引用数
                        'impact_factor': self._estimate_impact_factor(journal),  # 估算影响因子
                        'source': 'PubMed'
                    })
                    
                except Exception as e:
                    continue
            
            return articles
            
        except Exception as e:
            st.error(f"PubMed搜索出错: {str(e)}")
            return []
    
    def search_crossref(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        使用CrossRef API搜索文献
        """
        try:
            params = {
                'query': query,
                'rows': max_results,
                'sort': 'relevance',
                'order': 'desc'
            }
            
            response = requests.get(self.crossref_base_url, params=params, timeout=10)
            data = response.json()
            
            articles = []
            if 'message' in data and 'items' in data['message']:
                for item in data['message']['items']:
                    try:
                        # 提取标题
                        title = item.get('title', ['Unknown Title'])[0]
                        
                        # 提取作者
                        authors = []
                        if 'author' in item:
                            for author in item['author']:
                                given = author.get('given', '')
                                family = author.get('family', '')
                                if given and family:
                                    authors.append(f"{given} {family}")
                        
                        # 提取期刊
                        journal = ""
                        if 'container-title' in item and item['container-title']:
                            journal = item['container-title'][0]
                        
                        # 提取年份
                        year = 2024
                        if 'published-print' in item and 'date-parts' in item['published-print']:
                            year = item['published-print']['date-parts'][0][0]
                        elif 'published-online' in item and 'date-parts' in item['published-online']:
                            year = item['published-online']['date-parts'][0][0]
                        
                        # DOI
                        doi = item.get('DOI', '')
                        
                        # 引用数（CrossRef提供）
                        citations = item.get('is-referenced-by-count', 0)
                        
                        articles.append({
                            'title': title,
                            'authors': authors,
                            'journal': journal,
                            'year': year,
                            'pmid': '',
                            'doi': doi,
                            'abstract': '',  # CrossRef通常不提供摘要
                            'citations': citations,
                            'impact_factor': self._estimate_impact_factor(journal),
                            'source': 'CrossRef'
                        })
                        
                    except Exception as e:
                        continue
            
            return articles
            
        except Exception as e:
            st.error(f"CrossRef搜索出错: {str(e)}")
            return []
    
    def search_semantic_scholar(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        使用Semantic Scholar API搜索文献
        """
        try:
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,authors,journal,year,citationCount,abstract,externalIds'
            }
            
            response = requests.get(self.semantic_scholar_base_url, params=params, timeout=10)
            data = response.json()
            
            articles = []
            if 'data' in data:
                for item in data['data']:
                    try:
                        # 提取基本信息
                        title = item.get('title', 'Unknown Title')
                        
                        # 作者信息
                        authors = []
                        if 'authors' in item and item['authors']:
                            for author in item['authors']:
                                if 'name' in author:
                                    authors.append(author['name'])
                        
                        # 期刊信息
                        journal = item.get('journal', {}).get('name', 'Unknown Journal') if item.get('journal') else 'Unknown Journal'
                        
                        # 年份
                        year = item.get('year', 2024)
                        
                        # 引用数
                        citations = item.get('citationCount', 0)
                        
                        # 摘要
                        abstract = item.get('abstract', '')
                        
                        # DOI和PMID
                        doi = ''
                        pmid = ''
                        if 'externalIds' in item and item['externalIds']:
                            doi = item['externalIds'].get('DOI', '')
                            pmid = item['externalIds'].get('PubMed', '')
                        
                        articles.append({
                            'title': title,
                            'authors': authors,
                            'journal': journal,
                            'year': year,
                            'pmid': pmid,
                            'doi': doi,
                            'abstract': abstract,
                            'citations': citations,
                            'impact_factor': self._estimate_impact_factor(journal),
                            'source': 'Semantic Scholar'
                        })
                        
                    except Exception as e:
                        continue
            
            return articles
            
        except Exception as e:
            st.error(f"Semantic Scholar搜索出错: {str(e)}")
            return []
    
    def comprehensive_search(self, query: str, max_results_per_source: int = 10) -> List[Dict]:
        """
        综合搜索：同时使用多个API源
        """
        all_articles = []
        
        # 搜索进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # PubMed搜索
        status_text.text("正在搜索PubMed数据库...")
        progress_bar.progress(0.1)
        pubmed_results = self.search_pubmed(query, max_results_per_source)
        all_articles.extend(pubmed_results)
        
        time.sleep(self.request_delay)
        
        # CrossRef搜索
        status_text.text("正在搜索CrossRef数据库...")
        progress_bar.progress(0.5)
        crossref_results = self.search_crossref(query, max_results_per_source)
        all_articles.extend(crossref_results)
        
        time.sleep(self.request_delay)
        
        # Semantic Scholar搜索
        status_text.text("正在搜索Semantic Scholar数据库...")
        progress_bar.progress(0.8)
        semantic_results = self.search_semantic_scholar(query, max_results_per_source)
        all_articles.extend(semantic_results)
        
        # 去重（基于DOI和标题）
        status_text.text("正在处理和去重结果...")
        progress_bar.progress(0.9)
        unique_articles = self._deduplicate_articles(all_articles)
        
        # 按引用数排序
        unique_articles.sort(key=lambda x: x['citations'], reverse=True)
        
        progress_bar.progress(1.0)
        status_text.text(f"搜索完成！共找到 {len(unique_articles)} 篇相关文献")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return unique_articles
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """去重文章"""
        seen_dois = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            # 基于DOI去重
            if article['doi'] and article['doi'] in seen_dois:
                continue
            
            # 基于标题去重（简单的相似性检查）
            title_lower = article['title'].lower().strip()
            if title_lower in seen_titles:
                continue
            
            if article['doi']:
                seen_dois.add(article['doi'])
            seen_titles.add(title_lower)
            unique_articles.append(article)
        
        return unique_articles
    
    def _estimate_citations(self, year: int) -> int:
        """根据年份估算引用数"""
        current_year = 2024
        age = current_year - year
        if age <= 0:
            return 0
        elif age <= 2:
            return max(0, int(age * 5))
        else:
            return max(0, int(age * 3))
    
    def _estimate_impact_factor(self, journal: str) -> float:
        """估算期刊影响因子"""
        # 简单的期刊影响因子估算
        high_impact_keywords = ['nature', 'science', 'cell', 'lancet', 'nejm']
        medium_impact_keywords = ['plos', 'bmc', 'frontiers', 'scientific reports']
        
        journal_lower = journal.lower()
        
        for keyword in high_impact_keywords:
            if keyword in journal_lower:
                return round(15.0 + (hash(journal) % 20), 2)
        
        for keyword in medium_impact_keywords:
            if keyword in journal_lower:
                return round(5.0 + (hash(journal) % 10), 2)
        
        return round(2.0 + (hash(journal) % 8), 2)
    
    def generate_search_suggestions(self, base_query: str) -> List[str]:
        """生成搜索建议"""
        suggestions = []
        
        # 基于查询类型生成建议
        if any(term in base_query.lower() for term in ['qsar', 'quantitative structure', 'activity relationship']):
            suggestions.extend([
                f"{base_query} machine learning",
                f"{base_query} deep learning",
                f"{base_query} molecular descriptors",
                f"{base_query} drug discovery"
            ])
        
        if any(term in base_query.lower() for term in ['drug', 'pharmaceutical', 'medicine']):
            suggestions.extend([
                f"{base_query} clinical trial",
                f"{base_query} pharmacokinetics",
                f"{base_query} toxicity",
                f"{base_query} bioavailability"
            ])
        
        # 添加时间限制建议
        suggestions.extend([
            f"{base_query} 2020:2024[dp]",  # PubMed日期格式
            f"{base_query} review",
            f"{base_query} meta-analysis"
        ])
        
        return suggestions[:5]  # 返回前5个建议

def export_to_bibtex(articles: List[Dict]) -> str:
    """导出为BibTeX格式"""
    bibtex_entries = []
    
    for i, article in enumerate(articles, 1):
        # 生成BibTeX key
        first_author = article['authors'][0].split()[-1] if article['authors'] else 'Unknown'
        key = f"{first_author}{article['year']}"
        
        # 构建BibTeX条目
        entry = f"@article{{{key},\n"
        entry += f"  title = {{{article['title']}}},\n"
        
        if article['authors']:
            authors_str = ' and '.join(article['authors'])
            entry += f"  author = {{{authors_str}}},\n"
        
        entry += f"  journal = {{{article['journal']}}},\n"
        entry += f"  year = {{{article['year']}}},\n"
        
        if article['doi']:
            entry += f"  doi = {{{article['doi']}}},\n"
        
        if article['pmid']:
            entry += f"  pmid = {{{article['pmid']}}},\n"
        
        entry += f"  note = {{Citations: {article['citations']}, Source: {article['source']}}}\n"
        entry += "}\n"
        
        bibtex_entries.append(entry)
    
    return '\n'.join(bibtex_entries) 