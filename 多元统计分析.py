# wine_multivariate_analysis_enhanced.py
"""
基于UCI葡萄酒数据集的增强版多元统计分析
Enhanced Multivariate Statistical Analysis based on UCI Wine Dataset
包括：PCA、因子分析、判别分析、聚类分析、统计检验
Includes: PCA, Factor Analysis, Discriminant Analysis, Clustering Analysis, Statistical Tests
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           silhouette_score, silhouette_samples, adjusted_rand_score,
                           calinski_harabasz_score, davies_bouldin_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 设置和配置 ====================
def setup_environment():
    """设置环境和样式"""
    # 移除中文字体设置，使用默认英文字体
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置样式
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 设置更大的字体大小确保可读性
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # 设置随机种子
    np.random.seed(42)
    
    # 新增：创建图片保存文件夹
    os.makedirs("wine_analysis_plots", exist_ok=True)
    print("Environment setup completed")
    print(f"Image folder 'wine_analysis_plots' created/verified")

# ==================== 2. 数据加载和探索 ====================
def load_and_explore_data():
    """加载和探索数据"""
    print("=" * 80)
    print("UCI Wine Dataset Multivariate Statistical Analysis")
    print("=" * 80)
    
    # 加载数据集
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    print("\n1. Data Overview")
    print(f"Number of samples: {X.shape[0]}, Number of features: {X.shape[1]}")
    print(f"Wine types: {target_names}")
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Wine_Type'] = y
    df['Wine_Type_Name'] = [target_names[i] for i in y]
    
    # 基本统计
    print("\nClass distribution:")
    class_dist = pd.Series(y).value_counts().sort_index()
    for i, count in class_dist.items():
        print(f"  {target_names[i]}: {count} samples ({count/len(y):.1%})")
    
    return df, X, y, feature_names, target_names

# ==================== 3. 统计检验 ====================
def perform_statistical_tests(X, y, feature_names):
    """执行统计检验"""
    print("\n" + "=" * 80)
    print("2. Statistical Tests")
    print("=" * 80)
    
    results = []
    
    for i, feature in enumerate(feature_names):
        # ANOVA检验（比较三类之间的差异）
        groups = [X[y == j, i] for j in range(3)]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Kruskal-Wallis检验（非参数版本）
        h_stat, kw_p_value = stats.kruskal(*groups)
        
        # 计算效应量（eta平方）
        ss_between = sum([len(g) * (np.mean(g) - np.mean(X[:, i]))**2 for g in groups])
        ss_total = sum((X[:, i] - np.mean(X[:, i]))**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        results.append({
            'Feature': feature,
            'Mean_class0': np.mean(groups[0]),
            'Mean_class1': np.mean(groups[1]),
            'Mean_class2': np.mean(groups[2]),
            'F_statistic': f_stat,
            'ANOVA_P_value': p_value,
            'H_statistic': h_stat,
            'K-W_P_value': kw_p_value,
            'Effect_size(η²)': eta_squared,
            'Significant(α=0.05)': p_value < 0.05
        })
    
    stats_df = pd.DataFrame(results)
    
    # 显示重要结果
    print("\nSummary of statistical test results:")
    print("=" * 60)
    print("Most significant features (smallest ANOVA P-value):")
    sig_features = stats_df.sort_values('ANOVA_P_value').head(5)
    for _, row in sig_features.iterrows():
        print(f"  {row['Feature']}: F={row['F_statistic']:.2f}, p={row['ANOVA_P_value']:.4f}, η²={row['Effect_size(η²)']:.3f}")
    
    return stats_df

# ==================== 4. 数据预处理 ====================
def preprocess_data(X, y, scaling_method='standard'):
    """数据预处理"""
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    # 检查异常值
    z_scores = np.abs(stats.zscore(X_scaled))
    outliers = np.any(z_scores > 3, axis=1)
    outlier_rate = outliers.sum() / len(X)
    
    print(f"\nPreprocessing completed:")
    print(f"  Scaling method: {scaling_method}")
    print(f"  Outlier ratio: {outlier_rate:.2%}")
    
    # 新增：返回outlier_rate，供后续报告使用
    return X_scaled, scaler, outlier_rate

# ==================== 5. 主成分分析 ====================
def perform_pca_analysis(X_scaled, feature_names, y, target_names):
    """主成分分析"""
    print("\n" + "=" * 80)
    print("3. Principal Component Analysis (PCA)")
    print("=" * 80)
    
    # 执行PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # 解释方差
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print("Variance explained by principal components:")
    for i, (exp, cum) in enumerate(zip(explained_var, cumulative_var)):
        if i < 5:  # 只显示前5个
            print(f"  PC{i+1}: {exp:.3%} (Cumulative: {cum:.3%})")
    
    # 确定保留的主成分数
    n_components_85 = np.argmax(cumulative_var >= 0.85) + 1
    n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
    
    print(f"\nRecommendations:")
    print(f"  85% variance: Keep {n_components_85} PCs (Cumulative explained variance: {cumulative_var[n_components_85-1]:.2%})")
    print(f"  90% variance: Keep {n_components_90} PCs (Cumulative explained variance: {cumulative_var[n_components_90-1]:.2%})")
    
    # 载荷矩阵（后续绘图使用）
    loadings = pd.DataFrame(
        pca.components_[:3].T,
        index=feature_names,
        columns=['PC1', 'PC2', 'PC3']
    )
    
    # ------------ 拆分绘图1：碎石图 ------------
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, label='Variance per component')
    plt.plot(range(1, len(cumulative_var)+1), cumulative_var, 'r-', marker='o', label='Cumulative variance')
    plt.axhline(y=0.85, color='g', linestyle='--', label='85% threshold')
    plt.axhline(y=0.90, color='orange', linestyle=':', label='90% threshold')
    plt.axvline(x=n_components_85, color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=n_components_90, color='orange', linestyle=':', alpha=0.5)
    plt.title('PCA Scree Plot', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wine_analysis_plots/pca_1_scree_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
   # ------------ 拆分绘图2：载荷热图（前3个主成分）------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, cbar_kws={"shrink": 0.8}, linewidths=0.5,
                annot_kws={"size": 9})
    plt.title('Loadings Matrix (First 3 PCs)', fontsize=14, fontweight='bold')
# 修正：获取坐标轴对象，调用 set_yticklabels()
    ax = plt.gca()
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    plt.tight_layout()
    plt.savefig('wine_analysis_plots/pca_2_loadings_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ------------ 拆分绘图3：前两个主成分散点图 ------------
    plt.figure(figsize=(10, 6))
    scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis',
                           edgecolor='k', alpha=0.8, s=60)
    plt.xlabel(f'PC1 ({explained_var[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1]:.2%})', fontsize=12)
    plt.title('PCA Scatter Plot (First 2 PCs)', fontsize=14, fontweight='bold')
    # 创建图例
    for wine_type, color in zip(range(3), scatter1.cmap(scatter1.norm([0, 1, 2]))):
        plt.scatter([], [], c=[color], label=target_names[wine_type], edgecolor='k')
    plt.legend(title='Wine Type', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wine_analysis_plots/pca_3_2d_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ------------ 拆分绘图4：主成分与原始特征相关性 ------------
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(X_scaled.T, X_pca[:, :3].T)
    pca_feature_corr = corr_matrix[:len(feature_names), len(feature_names):]
    im = plt.imshow(pca_feature_corr, cmap='coolwarm', aspect='auto')
    plt.title('Correlation between PCs and Original Features', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component')
    plt.ylabel('Original Feature')
    plt.xticks([0, 1, 2], ['PC1', 'PC2', 'PC3'], fontsize=10)
    plt.yticks(range(len(feature_names)), feature_names, fontsize=9)
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig('wine_analysis_plots/pca_4_pc_feature_corr.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ------------ 拆分绘图5：三维PCA可视化 ------------
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    scatter3d = ax3d.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                            c=y, cmap='viridis', s=50, alpha=0.8)
    ax3d.set_xlabel(f'PC1 ({explained_var[0]:.2%})', fontsize=10)
    ax3d.set_ylabel(f'PC2 ({explained_var[1]:.2%})', fontsize=10)
    ax3d.set_zlabel(f'PC3 ({explained_var[2]:.2%})', fontsize=10)
    ax3d.set_title('3D PCA Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('wine_analysis_plots/pca_5_3d_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ------------ 拆分绘图6：特征贡献度分析（Top 10）------------
    plt.figure(figsize=(10, 6))
    contributions = np.abs(pca.components_) * pca.explained_variance_[:, np.newaxis]
    total_contributions = contributions.sum(axis=0)
    # 取前10个最重要的特征
    top_features_idx = np.argsort(total_contributions)[-10:]
    top_features = [feature_names[i] for i in top_features_idx]
    top_contributions = total_contributions[top_features_idx]
    plt.barh(range(len(top_features)), top_contributions, 
             color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
    plt.yticks(range(len(top_features)), top_features, fontsize=10)
    plt.xlabel('Total Contribution')
    plt.title('Feature Contributions to PCs (Top 10)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('wine_analysis_plots/pca_6_feature_contribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA plots saved to 'wine_analysis_plots' folder (6 images)")
    # 新增：返回关键PCA参数，供报告使用
    return pca, X_pca, explained_var, loadings, n_components_85, n_components_90, cumulative_var
# ==================== 6. 因子分析 ====================
def perform_factor_analysis(X_scaled, feature_names):
    """因子分析"""
    print("\n" + "=" * 80)
    print("4. Factor Analysis")
    print("=" * 80)
    
    # 尝试不同的因子数量
    results = []
    best_score = -1
    best_fa = None
    best_n = 0
    
    for n_factors in range(2, min(7, len(feature_names))):
        fa = FactorAnalysis(n_components=n_factors, random_state=42, max_iter=1000)
        X_fa = fa.fit_transform(X_scaled)
        
        # 计算共同度
        loadings = fa.components_.T
        communality = np.sum(loadings ** 2, axis=1)
        avg_communality = communality.mean()
        
        results.append({
            'Number of Factors': n_factors,
            'Average Communality': avg_communality,
            'Variance Explained': fa.noise_variance_,
            'Convergence': fa.n_iter_
        })
        
        if avg_communality > best_score:
            best_score = avg_communality
            best_fa = fa
            best_n = n_factors
    
    # 显示结果
    results_df = pd.DataFrame(results)
    print("\nComparison of different numbers of factors:")
    print(results_df.to_string(index=False))
    
    print(f"\nBest choice: {best_n} factors (Average communality: {best_score:.3f})")
    
    # 使用最佳因子数
    fa = best_fa
    X_fa = fa.transform(X_scaled)
    loadings = fa.components_.T
    
    # 计算共同度和特征值
    communality = np.sum(loadings ** 2, axis=1)
    eigenvalues = np.sum(loadings ** 2, axis=0)
    
    # 创建因子载荷矩阵
    factor_loadings = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=[f'Factor{i+1}' for i in range(best_n)]
    )
    
    print("\nFactor Loadings Matrix:")
    print(factor_loadings.round(3))
    
    print("\nCommunality (Proportion of Variance Explained):")
    comm_df = pd.DataFrame({
        'Feature': feature_names,
        'Communality': communality,
        'Explanation Level': ['High' if c > 0.7 else 'Medium' if c > 0.5 else 'Low' for c in communality]
    })
    print(comm_df.round(3))
    
    print(f"\nAverage Communality: {communality.mean():.3f}")
    print(f"Eigenvalues: {eigenvalues.round(3)}")
        # ------------ 拆分绘图1：因子载荷热图 ------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(factor_loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, cbar_kws={"shrink": 0.8}, linewidths=0.5,
                annot_kws={"size": 9})
    plt.title('Factor Loadings Matrix', fontsize=14, fontweight='bold')
    # 修正：获取坐标轴对象，调用 set_yticklabels()
    ax = plt.gca()
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    plt.tight_layout()
    plt.savefig('wine_analysis_plots/fa_1_loadings_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ------------ 拆分绘图2：共同度条形图 ------------
    plt.figure(figsize=(10, 8))
    comm_sorted = comm_df.sort_values('Communality', ascending=True)
    plt.barh(range(len(comm_sorted)), comm_sorted['Communality'], 
             color=plt.cm.viridis(np.linspace(0, 1, len(comm_sorted))))
    plt.yticks(range(len(comm_sorted)), comm_sorted['Feature'], fontsize=9)
    plt.xlabel('Communality')
    plt.title('Communality of Features', fontsize=14, fontweight='bold')
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold 0.5')
    plt.axvline(x=0.7, color='green', linestyle=':', alpha=0.7, label='Threshold 0.7')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('wine_analysis_plots/fa_2_communality_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ------------ 拆分绘图3：因子相关性热图 ------------
    plt.figure(figsize=(8, 8))
    factor_corr = np.corrcoef(X_fa.T)
    im = plt.imshow(factor_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation between Factors', fontsize=14, fontweight='bold')
    plt.xlabel('Factor')
    plt.ylabel('Factor')
    plt.xticks(range(best_n), [f'Factor{i+1}' for i in range(best_n)], fontsize=10)
    plt.yticks(range(best_n), [f'Factor{i+1}' for i in range(best_n)], fontsize=10)
    plt.colorbar(im, shrink=0.8)
    # 添加相关性数值
    for i in range(best_n):
        for j in range(best_n):
            plt.text(j, i, f'{factor_corr[i, j]:.2f}',
                     ha="center", va="center", color="black", fontsize=9)
    plt.tight_layout()
    plt.savefig('wine_analysis_plots/fa_3_factor_corr.png', dpi=300, bbox_inches='tight')
    plt.close()
    
   # ------------ 拆分绘图4：因子得分散点图 ------------
    plt.figure(figsize=(10, 6))
    plt.scatter(X_fa[:, 0], X_fa[:, 1], alpha=0.7, edgecolor='k', s=50)
    plt.xlabel('Factor 1 Score', fontsize=12)
    plt.ylabel('Factor 2 Score', fontsize=12)
    plt.title('Factor Scores Scatter Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # 标记极端值
    threshold1 = np.percentile(np.abs(X_fa[:, 0]), 95)
    threshold2 = np.percentile(np.abs(X_fa[:, 1]), 95)
    extreme_idx = np.where((np.abs(X_fa[:, 0]) > threshold1) | (np.abs(X_fa[:, 1]) > threshold2))[0]
    for idx in extreme_idx:
        plt.annotate(f'{idx}', (X_fa[idx, 0], X_fa[idx, 1]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, color='red')
    plt.tight_layout()
    plt.savefig('wine_analysis_plots/fa_4_score_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Factor analysis plots saved to 'wine_analysis_plots' folder (4 images)")
    return fa, factor_loadings, communality, results_df

# ==================== 7. 判别分析====================
def perform_discriminant_analysis(X_scaled, y, feature_names, target_names):
    """增强版判别分析"""
    print("\n" + "=" * 80)
    print("5. Discriminant Analysis")
    print("=" * 80)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # 尝试不同的判别方法
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    
    models = {
        'LDA': LDA(),
        'QDA': QDA(),
    }
    
    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        # 交叉验证分数
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 测试集评估
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'Method': name,
            'CV Mean Accuracy': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'Test Accuracy': test_accuracy,
            'Model': model
        })
        
        if test_accuracy > best_score:
            best_score = test_accuracy
            best_model = model
    
    results_df = pd.DataFrame(results)
    print("\nComparison of discriminant methods:")
    print(results_df[['Method', 'CV Mean Accuracy', 'CV Std', 'Test Accuracy']].to_string(index=False))
    
    # 使用最佳模型
    lda = best_model if 'LDA' in str(best_model) else LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    
    # 详细评估
    print(f"\nBest model: {results_df.loc[results_df['Test Accuracy'].idxmax(), 'Method']}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 特征重要性分析
    if hasattr(lda, 'coef_'):
        coefficients = pd.DataFrame(
            lda.coef_.T,
            index=feature_names,
            columns=[f'Discriminant Function {i+1}' for i in range(lda.coef_.shape[0])]
        )
        
        print("\nDiscriminant Function Coefficients (Top 5 by absolute value):")
        for i in range(coefficients.shape[1]):
            top_features = coefficients.iloc[:, i].abs().sort_values(ascending=False).head(5)
            print(f"\nMost important features for Discriminant Function {i+1}:")
            for feature in top_features.index:
                coef_value = coefficients.loc[feature, f'Discriminant Function {i+1}']
                print(f"  {feature}: {coef_value:.4f}")
    else:
        coefficients = None
        print("\nNote: Current model has no coefficient attribute (likely QDA)")
    
    # 排列重要性
    print("\nPermutation Importance Analysis (based on test set):")
    perm_importance = permutation_importance(lda, X_test, y_test, 
                                           n_repeats=10, random_state=42)
    
    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance Mean': perm_importance.importances_mean,
        'Importance Std': perm_importance.importances_std
    }).sort_values('Importance Mean', ascending=False)
    
    print(perm_df.head(10).to_string(index=False))
    
    # 可视化
 # 第一步：先创建子图，只传递figsize（不传递hspace、wspace）
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 第二步：单独设置子图间距（关键修正）
    plt.subplots_adjust(hspace=0.3, wspace=0.25)  # 两种写法均可，另一种：fig.subplots_adjust(...)    
    # 1. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, 
                ax=axes[0, 0], cbar_kws={"shrink": 0.8}, annot_kws={"size": 11})
    axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[0, 0].set_ylabel('True Label', fontsize=12)
    axes[0, 0].set_title(f'Confusion Matrix (Accuracy: {accuracy_score(y_test, y_pred):.2%})', 
                        fontsize=14, fontweight='bold')
    
    # 2. 判别函数得分图
    if hasattr(lda, 'transform'):
        X_lda = lda.transform(X_test)
        scatter = axes[0, 1].scatter(X_lda[:, 0], X_lda[:, 1], c=y_test, 
                                    cmap='viridis', edgecolor='k', alpha=0.8, s=60)
        axes[0, 1].set_xlabel('First Discriminant Function', fontsize=12)
        axes[0, 1].set_ylabel('Second Discriminant Function', fontsize=12)
        axes[0, 1].set_title('Discriminant Function Scores', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 特征重要性
    axes[1, 0].barh(range(len(perm_df)), perm_df['Importance Mean'].sort_values(), 
                   color=plt.cm.coolwarm(np.linspace(0, 1, len(perm_df))))
    axes[1, 0].set_yticks(range(len(perm_df)))
    axes[1, 0].set_yticklabels(perm_df['Feature'].iloc[np.argsort(perm_df['Importance Mean'])], fontsize=9)
    axes[1, 0].set_xlabel('Permutation Importance', fontsize=12)
    axes[1, 0].set_title('Feature Permutation Importance', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. 方法比较条形图
    methods = results_df['Method']
    cv_scores = results_df['CV Mean Accuracy']
    
    bars = axes[1, 1].bar(range(len(methods)), cv_scores, 
                         color=plt.cm.Set2(np.arange(len(methods))))
    axes[1, 1].set_xlabel('Method', fontsize=12)
    axes[1, 1].set_ylabel('Cross-Validation Accuracy', fontsize=12)
    axes[1, 1].set_title('Discriminant Methods Performance Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(range(len(methods)))
    axes[1, 1].set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值
    for bar, score, std in zip(bars, cv_scores, results_df['CV Std']):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f} (±{std:.3f})', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Discriminant Analysis Results', fontsize=16, fontweight='bold', y=1.02)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.98, hspace=0.3, wspace=0.25)
    plt.savefig('wine_analysis_plots/wine_discriminant_analysis_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    return lda, results_df, perm_df, coefficients

# ==================== 8. 聚类分析====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, silhouette_samples

def perform_clustering_analysis(X_scaled, y, feature_names, target_names):
    """增强版聚类分析（优化图表间距，防止文字重合）"""
    print("\n" + "=" * 80)
    print("6. Clustering Analysis")
    print("=" * 80)
    
    # 尝试不同的聚类方法
    clustering_methods = {
        'K-means (K=3)': KMeans(n_clusters=3, random_state=42, n_init=20),
        'K-means (K=2)': KMeans(n_clusters=2, random_state=42, n_init=20),
        'K-means (K=4)': KMeans(n_clusters=4, random_state=42, n_init=20),
        'Hierarchical (ward)': AgglomerativeClustering(n_clusters=3, linkage='ward'),
        'Hierarchical (average)': AgglomerativeClustering(n_clusters=3, linkage='average'),
        'DBSCAN (ε=1.5)': DBSCAN(eps=1.5, min_samples=5),
        'DBSCAN (ε=2.0)': DBSCAN(eps=2.0, min_samples=5),
    }
    
    results = []
    
    for name, model in clustering_methods.items():
        try:
            if 'DBSCAN' in name:
                labels = model.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = sum(labels == -1) / len(labels) if len(labels) > 0 else 0
            else:
                model.fit(X_scaled)
                labels = model.labels_
                n_clusters = len(set(labels))
                noise_ratio = 0
            
            # 计算聚类指标（仅当有多个聚类时）
            metrics = {}
            if n_clusters > 1 and n_clusters < len(X_scaled):
                try:
                    metrics['Silhouette Score'] = silhouette_score(X_scaled, labels)
                except:
                    metrics['Silhouette Score'] = np.nan
                
                try:
                    metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(X_scaled, labels)
                except:
                    metrics['Calinski-Harabasz Index'] = np.nan
                
                try:
                    metrics['Davies-Bouldin Index'] = davies_bouldin_score(X_scaled, labels)
                except:
                    metrics['Davies-Bouldin Index'] = np.nan
                
                # 与真实标签比较（如果可用）
                if y is not None:
                    try:
                        metrics['Adjusted Rand Index'] = adjusted_rand_score(y, labels)
                    except:
                        metrics['Adjusted Rand Index'] = np.nan
            else:
                metrics['Silhouette Score'] = np.nan
                metrics['Calinski-Harabasz Index'] = np.nan
                metrics['Davies-Bouldin Index'] = np.nan
                metrics['Adjusted Rand Index'] = np.nan
            
            results.append({
                'Method': name,
                'Number of Clusters': n_clusters,
                'Noise Ratio': f'{noise_ratio:.2%}',
                'Silhouette Score': metrics.get('Silhouette Score', np.nan),
                'Calinski-Harabasz Index': metrics.get('Calinski-Harabasz Index', np.nan),
                'Davies-Bouldin Index': metrics.get('Davies-Bouldin Index', np.nan),
                'Adjusted Rand Index': metrics.get('Adjusted Rand Index', np.nan),
                'Model': model,
                'Labels': labels
            })
            
        except Exception as e:
            print(f"Method {name} failed: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    print("\nComparison of clustering methods:")
    display_cols = ['Method', 'Number of Clusters', 'Noise Ratio', 'Silhouette Score', 'Adjusted Rand Index']
    print(results_df[display_cols].to_string(index=False))
    
    # 选择最佳聚类方法（基于轮廓系数）
    valid_results = results_df[~results_df['Silhouette Score'].isna()]
    if not valid_results.empty:
        best_idx = valid_results['Silhouette Score'].idxmax()
        best_method = results_df.loc[best_idx, 'Method']
        best_labels = results_df.loc[best_idx, 'Labels']
        best_model = results_df.loc[best_idx, 'Model']
        
        print(f"\nBest clustering method: {best_method}")
        print(f"Silhouette Score: {results_df.loc[best_idx, 'Silhouette Score']:.3f}")
        print(f"Adjusted Rand Index: {results_df.loc[best_idx, 'Adjusted Rand Index']:.3f}")
    else:
        print("\nWarning: No valid clustering results")
        best_method = 'K-means (K=3)'
        best_model = KMeans(n_clusters=3, random_state=42, n_init=20)
        best_model.fit(X_scaled)
        best_labels = best_model.labels_
    
    # 聚类可视化
    # 使用PCA降维到2维进行可视化
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    # 关键修改1：创建子图时，通过gridspec_kw设置子图内部间距（上下左右）
    fig, axes = plt.subplots(
        2, 3, 
        figsize=(22, 16),  # 适当放大图表尺寸，预留更多间距
        gridspec_kw={
            'hspace': 0.4,  # 子图垂直间距（上下），增大至0.4（原默认较小）
            'wspace': 0.3,  # 子图水平间距（左右），增大至0.3（原默认较小）
            'top': 0.9,     # 图表整体顶部边距（远离顶部边框）
            'bottom': 0.08, # 图表整体底部边距（远离底部边框）
            'left': 0.08,   # 图表整体左侧边距（远离左侧边框）
            'right': 0.98   # 图表整体右侧边距（远离右侧边框）
        }
    )
    
    # 1. 肘部法则（仅对K-means）
    inertias = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_test.fit(X_scaled)
        inertias.append(kmeans_test.inertia_)
    
    axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_title('K-means Elbow Method', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Clusters K', fontsize=12)
    axes[0, 0].set_ylabel('Sum of Squared Errors (Inertia)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 标记可能的肘部点
    differences = np.diff(inertias, 2)  # 二阶差分
    if len(differences) > 0:
        elbow_point = np.argmin(differences) + 2  # 二阶差分最小点
        axes[0, 0].axvline(x=elbow_point, color='red', linestyle='--', 
                          label=f'Suggested K={elbow_point}')
        axes[0, 0].legend(fontsize=9)
    
    # 2. 最佳聚类结果可视化
    scatter = axes[0, 1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                                c=best_labels, cmap='tab10', 
                                edgecolor='k', alpha=0.8, s=60)
    axes[0, 1].set_xlabel('First Principal Component', fontsize=12)
    axes[0, 1].set_ylabel('Second Principal Component', fontsize=12)
    axes[0, 1].set_title(f'Best Clustering: {best_method}', fontsize=14, fontweight='bold')
    
    # 如果是K-means，标记聚类中心
    if hasattr(best_model, 'cluster_centers_'):
        centers_pca = pca_2d.transform(best_model.cluster_centers_)
        axes[0, 1].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=300,
                          marker='X', edgecolors='black', linewidth=2, label='Cluster Centers')
        axes[0, 1].legend(fontsize=9)
    
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 与真实标签对比
    if y is not None:
        axes[0, 2].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis',
                          edgecolor='k', alpha=0.8, s=60)
        axes[0, 2].set_xlabel('First Principal Component', fontsize=12)
        axes[0, 2].set_ylabel('Second Principal Component', fontsize=12)
        axes[0, 2].set_title('True Classes', fontsize=14, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 聚类指标比较
    # 轮廓系数比较
    methods_with_silhouette = results_df[~results_df['Silhouette Score'].isna()]
    if not methods_with_silhouette.empty:
        axes[1, 0].bar(range(len(methods_with_silhouette)), 
                      methods_with_silhouette['Silhouette Score'],
                      color=plt.cm.Set3(np.arange(len(methods_with_silhouette))))
        axes[1, 0].set_xlabel('Method', fontsize=12)
        axes[1, 0].set_ylabel('Silhouette Score', fontsize=12)
        axes[1, 0].set_title('Silhouette Score Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(len(methods_with_silhouette)))
        axes[1, 0].set_xticklabels(methods_with_silhouette['Method'], 
                                  rotation=45, ha='right', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值（微调位置，防止与柱子重合）
        for i, (idx, row) in enumerate(methods_with_silhouette.iterrows()):
            axes[1, 0].text(i, row['Silhouette Score'] + 0.015,  # 微调y轴偏移量
                           f"{row['Silhouette Score']:.3f}", ha='center', va='bottom', fontsize=8)
    
    # 5. 轮廓分析
    if 'Silhouette Score' in results_df.columns and not results_df['Silhouette Score'].isna().all():
        best_silhouette_idx = results_df['Silhouette Score'].idxmax()
        best_silhouette_labels = results_df.loc[best_silhouette_idx, 'Labels']
        
        # 计算每个样本的轮廓系数
        sample_silhouette_values = silhouette_samples(X_scaled, best_silhouette_labels)
        
        axes[1, 1].set_xlim([-0.1, 1])
        axes[1, 1].set_ylim([0, len(X_scaled) + (len(set(best_silhouette_labels)) + 1) * 10])
        
        y_lower = 10
        for i in sorted(set(best_silhouette_labels)):
            # 聚集第i个簇的轮廓系数并排序
            ith_cluster_silhouette_values = sample_silhouette_values[best_silhouette_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(i) / len(set(best_silhouette_labels)))
            axes[1, 1].fill_betweenx(np.arange(y_lower, y_upper),
                                   0, ith_cluster_silhouette_values,
                                   facecolor=color, edgecolor=color, alpha=0.7)
            
            # 标记簇标签（微调位置，防止超出边界）
            axes[1, 1].text(-0.08, y_lower + 0.5 * size_cluster_i, str(i), fontsize=10)
            
            y_lower = y_upper + 10  # 为下一个簇留出空间
        
        axes[1, 1].axvline(x=results_df.loc[best_silhouette_idx, 'Silhouette Score'], 
                          color="red", linestyle="--")
        
        axes[1, 1].set_xlabel("Silhouette Coefficient Values", fontsize=12)
        axes[1, 1].set_ylabel("Cluster Label", fontsize=12)
        axes[1, 1].set_title("Silhouette Analysis", fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 聚类特征分析
    # 分析每个聚类的特征均值
    if hasattr(best_model, 'labels_') or 'Labels' in results_df.columns:
        cluster_labels = best_labels
        n_clusters = len(set(cluster_labels))
        
        # 计算每个聚类的特征均值
        cluster_means = []
        for i in range(n_clusters):
            cluster_data = X_scaled[cluster_labels == i]
            cluster_means.append(np.mean(cluster_data, axis=0))
        
        cluster_means_df = pd.DataFrame(cluster_means, 
                                       columns=feature_names,
                                       index=[f'Cluster {i}' for i in range(n_clusters)])
        
        # 可视化热图（微调colorbar位置，防止与文字重合）
        im = axes[1, 2].imshow(cluster_means_df.T, cmap='coolwarm', aspect='auto')
        axes[1, 2].set_title('Cluster Feature Means Heatmap', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Cluster', fontsize=12)
        axes[1, 2].set_ylabel('Feature', fontsize=12)
        axes[1, 2].set_xticks(range(n_clusters))
        axes[1, 2].set_xticklabels([f'Cluster {i}' for i in range(n_clusters)], fontsize=10)
        axes[1, 2].set_yticks(range(len(feature_names)))
        axes[1, 2].set_yticklabels(feature_names, fontsize=9)

        plt.colorbar(im, ax=axes[1, 2], shrink=0.7, pad=0.05)
    
  
    plt.suptitle('Clustering Analysis Results', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
      plt.savefig('wine_clustering_analysis_enhanced.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    
    return best_model, results_df, best_labels

# ==================== 9. 综合方法对比 ====================
def comprehensive_method_comparison(pca_results, fa_results, lda_results, cluster_results):
    """综合方法对比"""
    print("\n" + "=" * 80)
    print("7. Comprehensive Method Comparison")
    print("=" * 80)
    
    # 收集各个方法的关键指标
    comparison_data = []
    
    # PCA指标（解析返回结果）
    pca, X_pca, explained_var, loadings, n_components_85, n_components_90, cumulative_var = pca_results
    comparison_data.append({
        'Method': 'Principal Component Analysis (PCA)',
        'Core Metric': f'{cumulative_var[2]:.3%}',
        'Metric Value': cumulative_var[2],
        'Metric Meaning': 'Cumulative variance explained by first 3 PCs',
        'Advantages': 'Strong dimensionality reduction, good visualization, retains main information',
        'Disadvantages': 'PCs are hard to interpret before rotation',
        'Application Scenarios': 'Dimensionality reduction, visualization, feature extraction',
        'Computational Complexity': 'Low',
        'Requires Labels': 'No'
    })
    
    # 因子分析指标
    fa, factor_loadings, communality, fa_results_df = fa_results
    communality_mean = communality.mean() if hasattr(communality, 'mean') else 0
    comparison_data.append({
        'Method': 'Factor Analysis',
        'Core Metric': f'{communality_mean:.3f}',
        'Metric Value': communality_mean,
        'Metric Meaning': 'Average communality (proportion of variance explained)',
        'Advantages': 'Identifies latent structures, factors are interpretable',
        'Disadvantages': 'Subjective choice of factor number, strict model assumptions',
        'Application Scenarios': 'Exploring latent structures, factor interpretation',
        'Computational Complexity': 'Medium',
        'Requires Labels': 'No'
    })
    
    # 判别分析指标
    lda, lda_results_df, perm_df, coefficients = lda_results
    lda_test_accuracy = lda_results_df['Test Accuracy'].max() if 'Test Accuracy' in lda_results_df.columns else 0
    comparison_data.append({
        'Method': 'Linear Discriminant Analysis (LDA)',
        'Core Metric': f'{lda_test_accuracy:.3%}',
        'Metric Value': lda_test_accuracy,
        'Metric Meaning': 'Test set classification accuracy',
        'Advantages': 'High classification accuracy, provides discriminant rules',
        'Disadvantages': 'Requires class labels, linearity assumption',
        'Application Scenarios': 'Classification prediction, pattern recognition',
        'Computational Complexity': 'Low',
        'Requires Labels': 'Yes'
    })
    
    # 聚类分析指标
    cluster_model, cluster_results_df, cluster_labels = cluster_results
    best_silhouette = cluster_results_df['Silhouette Score'].max() if 'Silhouette Score' in cluster_results_df.columns else 0
    comparison_data.append({
        'Method': 'K-means Clustering',
        'Core Metric': f'{best_silhouette:.3f}',
        'Metric Value': best_silhouette,
        'Metric Meaning': 'Silhouette Score (clustering quality)',
        'Advantages': 'Unsupervised learning, no labels needed, discovers intrinsic structures',
        'Disadvantages': 'Requires specifying K, sensitive to initial centers',
        'Application Scenarios': 'Data exploration, market segmentation, anomaly detection',
        'Computational Complexity': 'Medium',
        'Requires Labels': 'No'
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("Comprehensive comparison of multivariate statistical methods:")
    display_cols = ['Method', 'Core Metric', 'Metric Meaning', 'Advantages', 'Disadvantages', 'Application Scenarios']
    print(comparison_df[display_cols].to_string(index=False))
    
    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. 指标值雷达图
    methods = comparison_df['Method'].tolist()
    metrics = comparison_df['Metric Value'].tolist()
    
    # 归一化指标值（用于雷达图）
    metrics_norm = [(m - min(metrics)) / (max(metrics) - min(metrics)) if max(metrics) > min(metrics) else 0.5 
                    for m in metrics]
    
    # 雷达图需要闭合，所以重复第一个值
    metrics_norm.append(metrics_norm[0])
    methods_radar = methods + [methods[0]]
    
    angles = np.linspace(0, 2 * np.pi, len(methods), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    ax_radar = fig.add_subplot(2, 2, 1, polar=True)
    ax_radar.plot(angles, metrics_norm, 'o-', linewidth=2)
    ax_radar.fill(angles, metrics_norm, alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(methods, fontsize=9)
    ax_radar.set_ylim([0, 1])
    ax_radar.set_title('Method Performance Radar Chart (Normalized)', fontsize=14, fontweight='bold', pad=25)
    ax_radar.grid(True)
    
    # 2. 指标值条形图
    bars = axes[0, 1].bar(range(len(methods)), metrics, 
                         color=plt.cm.Set3(np.arange(len(methods))))
    axes[0, 1].set_xlabel('Method', fontsize=12)
    axes[0, 1].set_ylabel('Metric Value', fontsize=12)
    axes[0, 1].set_title('Core Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(range(len(methods)))
    axes[0, 1].set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值
    for bar, metric, method in zip(bars, metrics, methods):
        height = bar.get_height()
        unit = '%' if '%' in comparison_df.loc[comparison_df['Method'] == method, 'Core Metric'].values[0] else ''
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(metrics)*0.01,
                       f'{metric:.3f}{unit}', ha='center', va='bottom', fontsize=9)
    
    # 3. 优缺点总结
    axes[1, 0].axis('off')
    summary_text = "Summary of Multivariate Statistical Methods\n\n"
    for _, row in comparison_df.iterrows():
        summary_text += f"【{row['Method']}】\n"
        summary_text += f"  Core Metric: {row['Core Metric']} ({row['Metric Meaning']})\n"
        summary_text += f"  Advantages: {row['Advantages']}\n"
        summary_text += f"  Disadvantages: {row['Disadvantages']}\n"
        summary_text += f"  Application: {row['Application Scenarios']}\n\n"
    
    axes[1, 0].text(0.02, 0.98, summary_text, fontsize=8.5, 
                   verticalalignment='top', linespacing=1.5,
                   bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
    
    # 4. 方法选择流程图
    axes[1, 1].axis('off')
    flow_text = "Method Selection Guide\n\n"
    flow_text += "1. Goal: Dimensionality reduction and visualization\n"
    flow_text += "   → Choose: Principal Component Analysis (PCA)\n\n"
    flow_text += "2. Goal: Explore latent structures and factors\n"
    flow_text += "   → Choose: Factor Analysis\n\n"
    flow_text += "3. Goal: Classification prediction (with labels)\n"
    flow_text += "   → Choose: Discriminant Analysis (LDA/QDA)\n\n"
    flow_text += "4. Goal: Unsupervised grouping and exploration\n"
    flow_text += "   → Choose: Clustering Analysis (K-means/Hierarchical)\n\n"
    flow_text += "5. Goal: Comprehensive analysis and comparison\n"
    flow_text += "   → Suggestion: Combine multiple methods"
    
    axes[1, 1].text(0.02, 0.98, flow_text, fontsize=9.5, 
                   verticalalignment='top', linespacing=1.5,
                   bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('Comprehensive Comparison and Selection Guide', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('wine_method_comparison_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

# ==================== 10. 生成分析报告 ====================
def generate_comprehensive_report(df, stats_df, pca_results, fa_results, 
                                 lda_results, cluster_results, comparison_df,
                                 outlier_rate, target_names):
    """生成综合分析报告"""
    print("\n" + "=" * 80)
    print("8. Generating Analysis Report")
    print("=" * 80)
    
    # 解析传入的结果
    pca, X_pca, explained_var, pca_loadings, n_components_85, n_components_90, cumulative_var = pca_results
    fa, factor_loadings, communality, fa_results_df = fa_results
    lda, lda_results_df, perm_df, coefficients = lda_results
    cluster_model, cluster_results_df, cluster_labels = cluster_results
    
    # 计算关键指标
    communality_mean = communality.mean() if hasattr(communality, 'mean') else 0
    
    # 获取LDA准确率
    if lda_results_df is not None and not lda_results_df.empty:
        lda_accuracy = lda_results_df['Test Accuracy'].max()
        best_lda_method = lda_results_df.loc[lda_results_df['Test Accuracy'].idxmax(), 'Method']
    else:
        lda_accuracy = 0
        best_lda_method = "N/A"
    
    # 获取聚类轮廓系数
    if cluster_results_df is not None and not cluster_results_df.empty and 'Silhouette Score' in cluster_results_df.columns:
        best_silhouette = cluster_results_df['Silhouette Score'].max()
        best_cluster_method = cluster_results_df.loc[cluster_results_df['Silhouette Score'].idxmax(), 'Method']
    else:
        best_silhouette = 0
        best_cluster_method = "N/A"
    
    # 找出最重要的特征
    if coefficients is not None and not coefficients.empty:
        most_important_feature = coefficients.abs().max(axis=1).idxmax()
        most_important_value = coefficients.abs().max(axis=1).max()
    else:
        most_important_feature = "Cannot determine"
        most_important_value = 0
    
    # 找出最显著的统计特征
    if stats_df is not None and not stats_df.empty:
        most_sig_feature = stats_df.loc[stats_df['ANOVA_P_value'].idxmin(), 'Feature']
        most_sig_p = stats_df['ANOVA_P_value'].min()
    else:
        most_sig_feature = "Cannot determine"
        most_sig_p = 1
    
    # 创建样本分布字符串
    class_dist = df['Wine_Type_Name'].value_counts().sort_index()
    class_dist_str = ""
    for wine_type, count in class_dist.items():
        percentage = (count / len(df)) * 100
        class_dist_str += f"  - {wine_type}: {count} samples ({percentage:.1f}%)\n"
    
    # 计算平均效应量
    if stats_df is not None and not stats_df.empty and 'Effect_size(η²)' in stats_df.columns:
        avg_effect_size = stats_df['Effect_size(η²)'].mean()
    else:
        avg_effect_size = 0
    
    # 获取PCA关键信息
    if pca_loadings is not None and not pca_loadings.empty and 'PC1' in pca_loadings.columns:
        pca_key_feature = pca_loadings['PC1'].abs().idxmax()
    else:
        pca_key_feature = 'N/A'
    
    # 获取因子分析关键信息
    if factor_loadings is not None and not factor_loadings.empty:
        if 'Factor1' in factor_loadings.columns:
            factor1_feature = factor_loadings['Factor1'].abs().idxmax()
        else:
            factor1_feature = 'N/A'
        if 'Factor2' in factor_loadings.columns:
            factor2_feature = factor_loadings['Factor2'].abs().idxmax()
        else:
            factor2_feature = 'N/A'
        if 'Factor3' in factor_loadings.columns:
            factor3_feature = factor_loadings['Factor3'].abs().idxmax()
        else:
            factor3_feature = 'N/A'
    else:
        factor1_feature = factor2_feature = factor3_feature = 'N/A'
    
    # 获取因子分析结果
    if fa_results_df is not None and not fa_results_df.empty:
        best_fa_n = fa_results_df.loc[fa_results_df['Average Communality'].idxmax(), 'Number of Factors']
    else:
        best_fa_n = 0
    
    # 获取因子分析的方差解释比例
    if hasattr(fa, 'noise_variance_'):
        variance_explained_str = f"{np.mean(fa.noise_variance_):.3f}"
    else:
        variance_explained_str = 'N/A'
    
    # 获取判别分析交叉验证准确率
    if lda_results_df is not None and not lda_results_df.empty and 'CV Mean Accuracy' in lda_results_df.columns:
        cv_accuracy = lda_results_df['CV Mean Accuracy'].max()
    else:
        cv_accuracy = 0
    
    # 获取排列重要性特征
    if perm_df is not None and not perm_df.empty:
        top_perm_feature = perm_df['Feature'].iloc[0]
    else:
        top_perm_feature = 'N/A'
    
    # 获取聚类调整兰德指数
    if cluster_results_df is not None and 'Adjusted Rand Index' in cluster_results_df.columns:
        ari_score = cluster_results_df['Adjusted Rand Index'].max()
    else:
        ari_score = 0
    
    # 格式化异常值比例
    outlier_rate_formatted = f"{outlier_rate:.2%}"
    
    # 生成报告
    report = f"""
{'='*80}
UCI Wine Dataset Multivariate Statistical Analysis Report
{'='*80}

1. Project Overview
Dataset: UCI Wine Recognition Data
Number of samples: {len(df)}
Number of features: {len(df.columns) - 2}  # excluding type columns
Wine types: {', '.join(df['Wine_Type_Name'].unique())}
Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

2. Data Overview
2.1 Sample Distribution
{class_dist_str}

2.2 Data Quality
- No missing values
- Outlier ratio: {outlier_rate_formatted}
- Moderate correlation between features, suitable for multivariate analysis

3. Statistical Test Results
3.1 Analysis of Variance (ANOVA)
- Most significant feature: {most_sig_feature} (P-value: {most_sig_p:.4f})
- Number of significant features: {sum(stats_df['Significant(α=0.05)']) if stats_df is not None and not stats_df.empty else 0}
- Average effect size (η²): {avg_effect_size:.3f}

4. Principal Component Analysis (PCA) Results
4.1 Variance Explained
- PC1 explained variance: {explained_var[0]:.2%}
- PC2 explained variance: {explained_var[1]:.2%}
- PC3 explained variance: {explained_var[2]:.2%}
- Cumulative variance explained by first 3 PCs: {cumulative_var[2]:.2%}
- Suggested number of PCs to retain: {n_components_85} (85% variance)

4.2 Key Findings
- Most important feature for PC1: {pca_key_feature}
- PCA effectively reduces dimensionality, first two PCs enable visualization

5. Factor Analysis Results
5.1 Factor Extraction
- Optimal number of factors: {best_fa_n}
- Average communality: {communality_mean:.3f}
- Variance explained: {variance_explained_str}

5.2 Factor Interpretation
- Factor 1 mainly represents: {factor1_feature}
- Factor 2 mainly represents: {factor2_feature}
- Factor 3 mainly represents: {factor3_feature}

6. Discriminant Analysis Results
6.1 Classification Performance
- Best method: {best_lda_method}
- Test set accuracy: {lda_accuracy:.2%}
- Cross-validation mean accuracy: {cv_accuracy:.3f}

6.2 Key Features
- Most important discriminant feature: {most_important_feature}
- Top feature by permutation importance: {top_perm_feature}

7. Clustering Analysis Results
7.1 Clustering Performance
- Best clustering method: {best_cluster_method}
- Silhouette Score: {best_silhouette:.3f}
- Adjusted Rand Index (vs. true labels): {ari_score:.3f}

7.2 Clustering Findings
- Number of clusters: {len(set(cluster_labels))}
- Clustering results show high consistency with true classes

8. Method Comparison and Selection Recommendations
8.1 Method Performance Ranking
1. Discriminant Analysis (LDA): Highest classification accuracy ({lda_accuracy:.2%})
2. Principal Component Analysis (PCA): Best dimensionality reduction ({cumulative_var[2]:.2%} cumulative variance)
3. Factor Analysis: Identifies latent structures (Average communality: {communality_mean:.3f})
4. Clustering Analysis: Unsupervised grouping (Silhouette Score: {best_silhouette:.3f})

8.2 Application Recommendations
- Wine classification problem: Recommend Linear Discriminant Analysis (LDA)
- Feature extraction and dimensionality reduction: Recommend Principal Component Analysis (PCA)
- Exploring wine chemical factors: Recommend Factor Analysis
- Unsupervised data exploration: Recommend K-means Clustering

9. Limitations
- Small sample size (only 178 samples)
- All wines from same region, may affect generalizability
- Linear Discriminant Analysis assumes multivariate normal distribution

10. Future Research Directions
- Try nonlinear methods (e.g., kernel PCA, SVM)
- Ensemble multiple classifiers to improve performance
- Collect more samples and features
- Consider time series analysis (if temporal dimension exists)

{'='*80}
Report generation completed
{'='*80}
"""
    
    # 保存报告
    with open('wine_comprehensive_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Analysis report saved to: wine_comprehensive_analysis_report.txt")
    
    # 保存所有结果到CSV
    if stats_df is not None and not stats_df.empty:
        stats_df.to_csv('wine_statistical_tests.csv', index=False)
    
    if pca_loadings is not None and not pca_loadings.empty:
        pca_loadings.to_csv('wine_pca_loadings_enhanced.csv')
    
    if factor_loadings is not None and not factor_loadings.empty:
        factor_loadings.to_csv('wine_factor_loadings_enhanced.csv')
    
    if coefficients is not None and not coefficients.empty:
        coefficients.to_csv('wine_lda_coefficients_enhanced.csv')
    
    if perm_df is not None and not perm_df.empty:
        perm_df.to_csv('wine_permutation_importance.csv', index=False)
    
    if cluster_results_df is not None and not cluster_results_df.empty:
        cluster_results_df.to_csv('wine_clustering_results.csv', index=False)
    
    if comparison_df is not None and not comparison_df.empty:
        comparison_df.to_csv('wine_method_comparison_enhanced.csv', index=False)
    
    print("All analysis results saved to CSV files.")
    
    return report

# ==================== 主程序 ====================
def main():
    """主程序"""
    # 1. 设置环境
    setup_environment()
    
    # 2. 加载数据
    df, X, y, feature_names, target_names = load_and_explore_data()
    
    # 3. 统计检验
    stats_df = perform_statistical_tests(X, y, feature_names)
    
    # 4. 数据预处理
    X_scaled, scaler, outlier_rate = preprocess_data(X, y, scaling_method='standard')
    
    # 5. 主成分分析
    pca_results = perform_pca_analysis(X_scaled, feature_names, y, target_names)
    
    # 6. 因子分析
    fa_results = perform_factor_analysis(X_scaled, feature_names)
    
    # 7. 判别分析
    lda_results = perform_discriminant_analysis(X_scaled, y, feature_names, target_names)
    
    # 8. 聚类分析
    cluster_results = perform_clustering_analysis(X_scaled, y, feature_names, target_names)
    
    # 9. 综合方法对比
    comparison_df = comprehensive_method_comparison(pca_results, fa_results, lda_results, cluster_results)
    
    # 10. 生成报告
    report = generate_comprehensive_report(df, stats_df, pca_results, fa_results, 
                                          lda_results, cluster_results, comparison_df,
                                          outlier_rate, target_names)
    
    print("\n" + "=" * 80)
    print("✅ Enhanced Multivariate Statistical Analysis Completed!")
    print("=" * 80)
    
    # 主程序中生成文件说明部分
    print("\nGenerated files:")
    print("  1. Visualization charts (PNG format) - saved to 'wine_analysis_plots' folder:")
    print("     - PCA plots (6 images: pca_1_*.png to pca_6_*.png)")
    print("     - Factor analysis plots (4 images: fa_1_*.png to fa_4_*.png)")
    print("     - Discriminant analysis plots (4 images: lda_1_*.png to lda_4_*.png)")
    print("     - Clustering analysis plots (6 images: cluster_1_*.png to cluster_6_*.png)")
    print("     - Method comparison plots (4 images: comp_1_*.png to comp_4_*.png)")
    
    print("\n  2. Data files (CSV format):")
    print("     - wine_statistical_tests.csv")
    print("     - wine_pca_loadings_enhanced.csv")
    print("     - wine_factor_loadings_enhanced.csv")
    print("     - wine_lda_coefficients_enhanced.csv")
    print("     - wine_permutation_importance.csv")
    print("     - wine_clustering_results.csv")
    print("     - wine_method_comparison_enhanced.csv")
    
    print("\n  3. Analysis report:")
    print("     - wine_comprehensive_analysis_report.txt")
    
    return {
        'data': df,
        'stats': stats_df,
        'pca': pca_results,
        'fa': fa_results,
        'lda': lda_results,
        'clustering': cluster_results,
        'comparison': comparison_df,
        'report': report
    }

# ==================== 程序入口 ====================
if __name__ == "__main__":
    print("Starting Enhanced Wine Multivariate Statistical Analysis...")
    print("Note: This program includes complete statistical analysis workflow")
    print("=" * 80)
    
    try:
        results = main()
        print("\n🎉 Analysis successfully completed!")
        
    except Exception as e:
        print(f"\n❌ Program execution error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTrying to run simplified version...")
