#!/usr/bin/env python3
"""
市场趋势预测工具
基于Facebook Prophet进行跨境电商产品需求预测
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class MarketTrendPredictor:
    """市场趋势预测器"""
    
    def __init__(self):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.is_fitted = False
    
    def prepare_data(self, df, date_col='date', value_col='sales'):
        """准备Prophet格式数据"""
        data = df[[date_col, value_col]].copy()
        data.columns = ['ds', 'y']
        data['ds'] = pd.to_datetime(data['ds'])
        return data.sort_values('ds')
    
    def fit(self, df, date_col='date', value_col='sales'):
        """训练预测模型"""
        data = self.prepare_data(df, date_col, value_col)
        self.model.fit(data)
        self.is_fitted = True
        print(f"模型训练完成，数据点数: {len(data)}")
    
    def predict(self, periods=30, freq='D'):
        """预测未来趋势"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def plot_forecast(self, forecast=None, periods=30):
        """绘制预测结果"""
        if forecast is None:
            forecast = self.predict(periods)
        
        fig = self.model.plot(forecast, figsize=(12, 6))
        plt.title('市场趋势预测')
        plt.xlabel('日期')
        plt.ylabel('销量')
        plt.show()
        
        return fig

def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-10-31', freq='D')
    
    # 模拟季节性和趋势
    trend = np.linspace(100, 200, len(dates))
    seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    weekly = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 15, len(dates))
    
    sales = trend + seasonal + weekly + noise
    sales = np.maximum(sales, 0)  # 确保非负
    
    return pd.DataFrame({
        'date': dates,
        'sales': sales
    })

def main():
    """示例使用"""
    # 生成示例数据
    data = generate_sample_data()
    print("生成示例数据:")
    print(data.head())
    
    # 创建预测器
    predictor = MarketTrendPredictor()
    
    # 训练模型
    predictor.fit(data)
    
    # 预测未来30天
    forecast = predictor.predict(periods=30)
    print("\n未来30天预测:")
    print(forecast.tail())
    
    # 绘制预测图
    predictor.plot_forecast(forecast)
    
    # 分析预测结果
    future_avg = forecast.tail(30)['yhat'].mean()
    current_avg = data.tail(30)['sales'].mean()
    growth_rate = (future_avg - current_avg) / current_avg * 100
    
    print(f"\n趋势分析:")
    print(f"当前30天平均销量: {current_avg:.2f}")
    print(f"预测30天平均销量: {future_avg:.2f}")
    print(f"预期增长率: {growth_rate:.2f}%")

if __name__ == "__main__":
    main()
