# ======================================
# 第一步：导入所需库
# ======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示
sns.set_style('whitegrid')

# ======================================
# 第二步：数据层面 - 数据获取与预处理
# ======================================
# 1. 数据生成（若有真实电商数据，可替换为 pd.read_csv/pd.read_excel）
# 模拟电商用户消费行为数据，包含8个核心特征
np.random.seed(42)  # 固定随机种子，保证结果可复现
n_users = 2000  # 模拟2000个用户

# 构建特征集
user_data = {
    'user_id': range(1, n_users+1),
    'total_consume': np.random.lognormal(8, 1.2, n_users).round(2),  # 总消费金额（对数正态分布，更符合真实消费）
    'order_count': np.random.poisson(5, n_users) + 1,  # 下单次数
    'avg_consume_per_order': (np.random.lognormal(6, 1, n_users)).round(2),  # 平均每单消费
    'last_consume_days': np.random.randint(1, 180, n_users),  # 最后一次消费距今天数
    'consume_frequency': np.random.randint(1, 30, n_users),  # 消费间隔（天/次）
    'coupon_use_rate': np.random.uniform(0, 1, n_users).round(2),  # 优惠券使用率
    'product_category_count': np.random.randint(1, 10, n_users),  # 购买商品品类数
    'is_member': np.random.randint(0, 2, n_users)  # 是否为会员（0=否，1=是）
}

# 转换为DataFrame
df = pd.DataFrame(user_data)

# 2. 数据探索性分析（EDA）
print("="*50)
print("数据基本信息")
print("="*50)
print(df.info())
print("\n" + "="*50)
print("数据描述性统计")
print("="*50)
print(df.describe().round(2))

# 缺失值检查与处理（模拟数据无缺失，真实数据可补充插值/删除逻辑）
print("\n" + "="*50)
print("缺失值统计")
print("="*50)
print(df.isnull().sum())

# 异常值处理（基于四分位数法处理总消费金额异常值）
def handle_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 替换异常值为边界值
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

df = handle_outliers(df, 'total_consume')
df = handle_outliers(df, 'avg_consume_per_order')

# 3. 特征工程
# 选择用于建模的特征（排除user_id）
features = ['total_consume', 'order_count', 'avg_consume_per_order',
            'last_consume_days', 'consume_frequency', 'coupon_use_rate',
            'product_category_count', 'is_member']
X = df[features].copy()

# 特征标准化（K-Means对数据尺度敏感，必须标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# ======================================
# 第三步：方法层面 - 模型构建与训练
# ======================================
# 第一部分：无监督聚类（K-Means++）
# 1. 确定最优K值（肘部法则 + 轮廓系数）
inertia_list = []  # 簇内平方和
silhouette_list = []  # 轮廓系数
calinski_list = []  # 卡林斯基-哈拉巴斯指数
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    inertia_list.append(kmeans.inertia_)
    silhouette_list.append(silhouette_score(X_scaled, cluster_labels))
    calinski_list.append(calinski_harabasz_score(X_scaled, cluster_labels))

# 绘制肘部法则图
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(k_range, inertia_list, 'bo-')
plt.xlabel('聚类数量K')
plt.ylabel('簇内平方和（Inertia）')
plt.title('肘部法则确定最优K')
plt.grid(True, alpha=0.3)

# 绘制轮廓系数图
plt.subplot(1, 3, 2)
plt.plot(k_range, silhouette_list, 'ro-')
plt.xlabel('聚类数量K')
plt.ylabel('轮廓系数（Silhouette Score）')
plt.title('轮廓系数确定最优K')
plt.grid(True, alpha=0.3)

# 绘制卡林斯基-哈拉巴斯指数图
plt.subplot(1, 3, 3)
plt.plot(k_range, calinski_list, 'go-')
plt.xlabel('聚类数量K')
plt.ylabel('卡林斯基-哈拉巴斯指数')
plt.title('CH指数确定最优K')
plt.tight_layout()
plt.savefig('k_selection.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 确定最优K=4（基于上述可视化结果，经验最优值）
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
df['cluster_label'] = kmeans.fit_predict(X_scaled)

# 第二部分：有监督分类（XGBoost）- 预测用户聚类标签
# 1. 构建分类数据集（聚类标签作为目标变量）
y = df['cluster_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. 初始化并训练XGBoost模型
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mlogloss', verbose=False)

# ======================================
# 第四步：分析层面 - 模型验证与结果解读
# ======================================
# 第一部分：聚类模型验证与结果分析
print("\n" + "="*50)
print("聚类模型评估指标")
print("="*50)
cluster_labels = df['cluster_label']
print(f"轮廓系数（越接近1越好）：{silhouette_score(X_scaled, cluster_labels):.4f}")
print(f"卡林斯基-哈拉巴斯指数（越高越好）：{calinski_harabasz_score(X_scaled, cluster_labels):.4f}")

# 聚类结果统计与画像分析
print("\n" + "="*50)
print("各聚类群体数量统计")
print("="*50)
cluster_count = df['cluster_label'].value_counts().sort_index()
print(cluster_count)

# 各聚类群体的特征均值分析（用户画像）
print("\n" + "="*50)
print("各聚类群体用户画像（特征均值）")
print("="*50)
cluster_profile = df.groupby('cluster_label')[features].mean().round(2)
print(cluster_profile)

# 聚类结果可视化（PCA降维到2D，方便展示）
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'cluster': cluster_labels})

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='cluster', palette='viridis', s=50, alpha=0.8)
plt.xlabel('PCA维度1')
plt.ylabel('PCA维度2')
plt.title(f'K-Means++聚类结果可视化（K={optimal_k}）')
plt.legend(title='用户群体标签')
plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 第二部分：分类模型验证与评估
# 1. 模型预测
y_pred = xgb_model.predict(X_test)

# 2. 模型评估指标
print("\n" + "="*50)
print("XGBoost分类模型评估结果")
print("="*50)
print(f"模型准确率（Accuracy）：{accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 3. 混淆矩阵可视化
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(optimal_k), yticklabels=range(optimal_k))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('XGBoost模型混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 特征重要性可视化
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.xlabel('特征重要性得分')
plt.ylabel('特征名称')
plt.title('XGBoost模型特征重要性排名')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================
# 第五步：结果保存
# ======================================
# 保存处理后的数据与聚类结果
df.to_excel('user_consume_cluster_result.xlsx', index=False)
# 保存特征重要性结果
feature_importance.to_excel('feature_importance_result.xlsx', index=False)

print("\n" + "="*50)
print("项目运行完成，所有结果已保存")
print("="*50)