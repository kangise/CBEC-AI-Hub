#!/usr/bin/env python3
"""
多语言情感分析工具
用于分析跨境电商产品评论的情感倾向
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class MultilingualSentimentAnalyzer:
    """多语言情感分析器"""
    
    def __init__(self, model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 初始化情感分析管道
        self.analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True
        )
        
        # 支持的语言
        self.supported_languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'ar', 'zh'
        ]
    
    def analyze_text(self, text: str) -> Dict:
        """分析单个文本的情感"""
        if not text or pd.isna(text):
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
            }
        
        # 执行情感分析
        results = self.analyzer(text)[0]
        
        # 处理结果
        scores = {result['label'].lower(): result['score'] for result in results}
        
        # 确定主要情感
        main_sentiment = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[main_sentiment]
        
        return {
            'sentiment': main_sentiment,
            'confidence': confidence,
            'scores': scores
        }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """批量分析文本情感"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # 过滤空文本
            valid_texts = [text for text in batch if text and not pd.isna(text)]
            
            if not valid_texts:
                results.extend([self.analyze_text("") for _ in batch])
                continue
            
            # 批量分析
            batch_results = self.analyzer(valid_texts)
            
            # 处理结果
            for j, text in enumerate(batch):
                if text and not pd.isna(text):
                    result_idx = sum(1 for t in batch[:j+1] if t and not pd.isna(t)) - 1
                    raw_result = batch_results[result_idx]
                    
                    scores = {r['label'].lower(): r['score'] for r in raw_result}
                    main_sentiment = max(scores.keys(), key=lambda k: scores[k])
                    
                    results.append({
                        'sentiment': main_sentiment,
                        'confidence': scores[main_sentiment],
                        'scores': scores
                    })
                else:
                    results.append(self.analyze_text(""))
        
        return results
    
    def analyze_reviews(self, reviews_df: pd.DataFrame, 
                       text_column: str = 'review_text',
                       rating_column: str = 'rating') -> pd.DataFrame:
        """分析评论数据框"""
        
        # 复制数据框
        result_df = reviews_df.copy()
        
        # 批量分析情感
        print(f"分析 {len(reviews_df)} 条评论...")
        sentiment_results = self.analyze_batch(reviews_df[text_column].tolist())
        
        # 添加结果到数据框
        result_df['sentiment'] = [r['sentiment'] for r in sentiment_results]
        result_df['sentiment_confidence'] = [r['confidence'] for r in sentiment_results]
        result_df['positive_score'] = [r['scores'].get('positive', 0) for r in sentiment_results]
        result_df['negative_score'] = [r['scores'].get('negative', 0) for r in sentiment_results]
        result_df['neutral_score'] = [r['scores'].get('neutral', 0) for r in sentiment_results]
        
        # 如果有评分列，计算一致性
        if rating_column in reviews_df.columns:
            result_df['sentiment_rating_consistency'] = self._calculate_consistency(
                result_df[rating_column], result_df['sentiment']
            )
        
        return result_df
    
    def get_sentiment_summary(self, sentiment_results: List[Dict]) -> Dict:
        """获取情感分析摘要"""
        if not sentiment_results:
            return {}
        
        sentiments = [r['sentiment'] for r in sentiment_results]
        confidences = [r['confidence'] for r in sentiment_results]
        
        summary = {
            'total_reviews': len(sentiment_results),
            'sentiment_distribution': {
                'positive': sentiments.count('positive'),
                'negative': sentiments.count('negative'),
                'neutral': sentiments.count('neutral')
            },
            'average_confidence': np.mean(confidences),
            'sentiment_percentages': {
                'positive': sentiments.count('positive') / len(sentiments) * 100,
                'negative': sentiments.count('negative') / len(sentiments) * 100,
                'neutral': sentiments.count('neutral') / len(sentiments) * 100
            }
        }
        
        # 计算情感得分
        avg_positive = np.mean([r['scores'].get('positive', 0) for r in sentiment_results])
        avg_negative = np.mean([r['scores'].get('negative', 0) for r in sentiment_results])
        
        summary['overall_sentiment_score'] = avg_positive - avg_negative
        summary['sentiment_trend'] = self._determine_trend(summary['overall_sentiment_score'])
        
        return summary
    
    def _calculate_consistency(self, ratings: pd.Series, sentiments: pd.Series) -> pd.Series:
        """计算评分与情感的一致性"""
        consistency = []
        
        for rating, sentiment in zip(ratings, sentiments):
            if pd.isna(rating):
                consistency.append(None)
                continue
            
            # 定义一致性规则
            if rating >= 4 and sentiment == 'positive':
                consistency.append('consistent')
            elif rating <= 2 and sentiment == 'negative':
                consistency.append('consistent')
            elif rating == 3 and sentiment == 'neutral':
                consistency.append('consistent')
            else:
                consistency.append('inconsistent')
        
        return pd.Series(consistency)
    
    def _determine_trend(self, sentiment_score: float) -> str:
        """确定情感趋势"""
        if sentiment_score > 0.1:
            return 'positive'
        elif sentiment_score < -0.1:
            return 'negative'
        else:
            return 'neutral'

def generate_sample_reviews():
    """生成示例评论数据"""
    reviews = [
        {"review_text": "Great product! Fast shipping and excellent quality.", "rating": 5, "language": "en"},
        {"review_text": "Not what I expected. Poor quality materials.", "rating": 2, "language": "en"},
        {"review_text": "Average product, nothing special but works fine.", "rating": 3, "language": "en"},
        {"review_text": "Excelente producto, muy recomendado!", "rating": 5, "language": "es"},
        {"review_text": "Produit décevant, qualité médiocre.", "rating": 2, "language": "fr"},
        {"review_text": "Sehr gutes Produkt, schnelle Lieferung!", "rating": 4, "language": "de"},
        {"review_text": "产品质量很好，物流很快，推荐购买！", "rating": 5, "language": "zh"},
        {"review_text": "质量一般，价格偏高，不太值得。", "rating": 2, "language": "zh"}
    ]
    
    return pd.DataFrame(reviews)

def main():
    """示例使用"""
    print("初始化多语言情感分析器...")
    analyzer = MultilingualSentimentAnalyzer()
    
    # 生成示例数据
    reviews_df = generate_sample_reviews()
    print("\n示例评论数据:")
    print(reviews_df[['review_text', 'rating', 'language']])
    
    # 分析情感
    print("\n执行情感分析...")
    analyzed_df = analyzer.analyze_reviews(reviews_df)
    
    # 显示结果
    print("\n情感分析结果:")
    result_columns = ['review_text', 'rating', 'sentiment', 'sentiment_confidence', 'sentiment_rating_consistency']
    print(analyzed_df[result_columns])
    
    # 获取摘要
    sentiment_results = [
        {
            'sentiment': row['sentiment'],
            'confidence': row['sentiment_confidence'],
            'scores': {
                'positive': row['positive_score'],
                'negative': row['negative_score'],
                'neutral': row['neutral_score']
            }
        }
        for _, row in analyzed_df.iterrows()
    ]
    
    summary = analyzer.get_sentiment_summary(sentiment_results)
    
    print("\n情感分析摘要:")
    print(f"总评论数: {summary['total_reviews']}")
    print(f"情感分布: {summary['sentiment_distribution']}")
    print(f"平均置信度: {summary['average_confidence']:.3f}")
    print(f"整体情感得分: {summary['overall_sentiment_score']:.3f}")
    print(f"情感趋势: {summary['sentiment_trend']}")
    
    # 保存结果
    analyzed_df.to_csv('sentiment_analysis_results.csv', index=False, encoding='utf-8')
    print("\n结果已保存到 sentiment_analysis_results.csv")

if __name__ == "__main__":
    main()
