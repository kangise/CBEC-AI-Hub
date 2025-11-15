#!/usr/bin/env python3
"""
多语言Listing生成工具
基于开源LLM生成跨境电商产品标题、描述和关键词
"""

import re
import json
from typing import Dict, List, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class ListingGenerator:
    """多语言商品Listing生成器"""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 初始化文本生成管道
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if self.device == "cuda" else -1,
            max_length=512,
            do_sample=True,
            temperature=0.7
        )
        
        # 语言映射
        self.languages = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
    
    def generate_title(self, product_info: Dict, language: str = 'en', max_length: int = 80) -> str:
        """生成产品标题"""
        
        # 构建提示词
        prompt = self._build_title_prompt(product_info, language)
        
        # 生成标题
        result = self.generator(
            prompt,
            max_length=len(prompt.split()) + 20,
            num_return_sequences=1,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        
        # 提取生成的标题
        generated_text = result[0]['generated_text']
        title = generated_text.replace(prompt, '').strip()
        
        # 清理和截断
        title = self._clean_title(title, max_length)
        
        return title
    
    def generate_bullet_points(self, product_info: Dict, language: str = 'en', num_points: int = 5) -> List[str]:
        """生成产品五点描述"""
        
        bullet_points = []
        features = product_info.get('features', [])
        
        for i, feature in enumerate(features[:num_points]):
            prompt = self._build_bullet_prompt(feature, product_info, language)
            
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + 25,
                num_return_sequences=1,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            bullet = result[0]['generated_text'].replace(prompt, '').strip()
            bullet = self._clean_bullet_point(bullet)
            bullet_points.append(bullet)
        
        return bullet_points
    
    def generate_keywords(self, product_info: Dict, language: str = 'en', num_keywords: int = 10) -> List[str]:
        """生成SEO关键词"""
        
        # 基础关键词提取
        base_keywords = []
        
        # 从产品名称提取
        if 'name' in product_info:
            base_keywords.extend(product_info['name'].lower().split())
        
        # 从类别提取
        if 'category' in product_info:
            base_keywords.extend(product_info['category'].lower().split())
        
        # 从特性提取
        if 'features' in product_info:
            for feature in product_info['features']:
                base_keywords.extend(feature.lower().split())
        
        # 清理和去重
        keywords = list(set([kw for kw in base_keywords if len(kw) > 2]))
        
        # 生成相关关键词
        if len(keywords) < num_keywords:
            additional_keywords = self._generate_related_keywords(
                keywords, product_info, language, num_keywords - len(keywords)
            )
            keywords.extend(additional_keywords)
        
        return keywords[:num_keywords]
    
    def generate_complete_listing(self, product_info: Dict, language: str = 'en') -> Dict:
        """生成完整的商品Listing"""
        
        listing = {
            'language': language,
            'title': self.generate_title(product_info, language),
            'bullet_points': self.generate_bullet_points(product_info, language),
            'keywords': self.generate_keywords(product_info, language),
            'description': self._generate_description(product_info, language)
        }
        
        return listing
    
    def _build_title_prompt(self, product_info: Dict, language: str) -> str:
        """构建标题生成提示词"""
        lang_name = self.languages.get(language, 'English')
        
        prompt = f"Create a compelling product title in {lang_name} for: "
        prompt += f"{product_info.get('name', 'product')} "
        prompt += f"Category: {product_info.get('category', 'general')} "
        prompt += "Title:"
        
        return prompt
    
    def _build_bullet_prompt(self, feature: str, product_info: Dict, language: str) -> str:
        """构建五点描述提示词"""
        lang_name = self.languages.get(language, 'English')
        
        prompt = f"Write a product bullet point in {lang_name} about {feature} "
        prompt += f"for {product_info.get('name', 'this product')}: "
        
        return prompt
    
    def _clean_title(self, title: str, max_length: int) -> str:
        """清理标题"""
        # 移除多余空格和特殊字符
        title = re.sub(r'\s+', ' ', title).strip()
        title = re.sub(r'[^\w\s\-&]', '', title)
        
        # 截断到最大长度
        if len(title) > max_length:
            title = title[:max_length].rsplit(' ', 1)[0]
        
        return title
    
    def _clean_bullet_point(self, bullet: str) -> str:
        """清理五点描述"""
        bullet = re.sub(r'\s+', ' ', bullet).strip()
        
        # 确保以大写字母开头
        if bullet and bullet[0].islower():
            bullet = bullet[0].upper() + bullet[1:]
        
        # 移除末尾标点（如果不是句号）
        if bullet.endswith(('.', '!', '?')):
            pass
        else:
            bullet = bullet.rstrip('.,!?')
        
        return bullet
    
    def _generate_related_keywords(self, base_keywords: List[str], product_info: Dict, 
                                 language: str, num_needed: int) -> List[str]:
        """生成相关关键词"""
        # 简化版本：基于产品信息生成相关词
        related = []
        
        # 添加常见修饰词
        modifiers = {
            'en': ['best', 'premium', 'quality', 'durable', 'portable', 'professional'],
            'es': ['mejor', 'premium', 'calidad', 'duradero', 'portátil', 'profesional'],
            'fr': ['meilleur', 'premium', 'qualité', 'durable', 'portable', 'professionnel'],
            'de': ['beste', 'premium', 'qualität', 'langlebig', 'tragbar', 'professionell']
        }
        
        lang_modifiers = modifiers.get(language, modifiers['en'])
        
        for modifier in lang_modifiers[:num_needed]:
            if modifier not in base_keywords:
                related.append(modifier)
        
        return related[:num_needed]
    
    def _generate_description(self, product_info: Dict, language: str) -> str:
        """生成产品描述"""
        # 简化版本：组合现有信息
        description_parts = []
        
        if 'name' in product_info:
            description_parts.append(f"Introducing the {product_info['name']}")
        
        if 'features' in product_info:
            features_text = ", ".join(product_info['features'][:3])
            description_parts.append(f"featuring {features_text}")
        
        if 'benefits' in product_info:
            benefits_text = ". ".join(product_info['benefits'][:2])
            description_parts.append(f"Benefits include: {benefits_text}")
        
        return ". ".join(description_parts) + "."

def main():
    """示例使用"""
    # 示例产品信息
    product_info = {
        'name': 'Wireless Bluetooth Headphones',
        'category': 'Electronics Audio',
        'features': [
            'Noise Cancellation',
            '30-hour Battery Life', 
            'Quick Charge Technology',
            'Premium Sound Quality',
            'Comfortable Design'
        ],
        'benefits': [
            'Perfect for travel and commuting',
            'Crystal clear audio experience'
        ]
    }
    
    # 创建生成器（注意：实际使用时需要合适的模型）
    print("初始化Listing生成器...")
    generator = ListingGenerator()
    
    # 生成英文Listing
    print("\n生成英文Listing:")
    en_listing = generator.generate_complete_listing(product_info, 'en')
    
    print(f"标题: {en_listing['title']}")
    print("五点描述:")
    for i, bullet in enumerate(en_listing['bullet_points'], 1):
        print(f"  {i}. {bullet}")
    print(f"关键词: {', '.join(en_listing['keywords'])}")
    print(f"描述: {en_listing['description']}")
    
    # 保存结果
    with open('generated_listing.json', 'w', encoding='utf-8') as f:
        json.dump(en_listing, f, ensure_ascii=False, indent=2)
    
    print("\nListing已保存到 generated_listing.json")

if __name__ == "__main__":
    main()
