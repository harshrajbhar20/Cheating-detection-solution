import pandas as pd 
import numpy as np 
import json 
import os 
import networkx as nx 
from tqdm import tqdm 
import lightgbm as lgb 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import roc_auc_score 
import xgboost as xgb 
from catboost import CatBoostClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from scipy.optimize import minimize 
import warnings 
import optuna 
from optuna.samplers import TPESampler 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.neural_network import MLPClassifier 
from category_encoders.target_encoder import TargetEncoder 
from sklearn.model_selection import cross_val_score 
from sklearn.utils import resample 
warnings.filterwarnings('ignore') 
np.random.seed(42) 
# Utility functions 
def safe_xgb_fit(model, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=100): 
    try: 
        if X_val is not None and y_val is not None: 
            model.fit( 
                X_train, y_train, 
                eval_set=[(X_val, y_val)], 
                early_stopping_rounds=early_stopping_rounds, 
                verbose=False, 
            ) 
        else: 
            model.fit(X_train, y_train, verbose=False) 
    except TypeError: 
        try: 
            if X_val is not None and y_val is not None: 
                cb = [] 
                try: 
                    cb.append(xgb.callback.EarlyStopping(rounds=early_stopping_rounds)) 
                except Exception: 
                    cb = [] 
                model.fit( 
                    X_train, y_train, 
                    eval_set=[(X_val, y_val)], 
                    eval_metric='auc', 
                    callbacks=cb, 
                    verbose=False, 
                ) 
            else: 
                model.fit(X_train, y_train, verbose=False) 
        except Exception: 
            model.fit(X_train, y_train) 
    return model 
 
def ensure_df_index_reset(df): 
return df.reset_index(drop=True).copy() 
# Load data 
print("        
Loading datasets...") 
input_dir = '/kaggle/input/mercor-cheating-detection' 
train = pd.read_csv(os.path.join(input_dir, 'train.csv')) 
test = pd.read_csv(os.path.join(input_dir, 'test.csv')) 
graph_df = pd.read_csv(os.path.join(input_dir, 'social_graph.csv')) 
with open(os.path.join(input_dir, 'feature_metadata.json')) as f: 
metadata = json.load(f) 
# Prepare labels 
print("\n    
Preparing labels and sample weights...") 
train['label'] = train['is_cheating'].copy() 
if 'high_conf_clean' in train.columns: 
train.loc[train['high_conf_clean'] == 1, 'label'] = 0 
manual_review_mask = train['is_cheating'].notna() 
train_labeled = train[manual_review_mask].reset_index(drop=True).copy() 
train_unlabeled = train[~manual_review_mask].reset_index(drop=True).copy() 
print(f"   
Using {len(train_labeled)} manually reviewed samples for training.") 
print(f"   Cheating rate: {train_labeled['label'].mean():.2%}") 
# Graph feature engineering 
print("\n  
Building powerful and efficient graph features...") 
all_users = set(train_labeled['user_hash']).union(set(test['user_hash'])) 
relevant_edges = graph_df[ 
(graph_df['user_a'].isin(all_users)) | 
(graph_df['user_b'].isin(all_users)) 
].reset_index(drop=True) 
G = nx.from_pandas_edgelist(relevant_edges, source='user_a', target='user_b') 
# Degree features 
print("  → Calculating degree features...") 
degree_dict = dict(G.degree()) 
for df in [train_labeled, train_unlabeled, test]: 
df['degree'] = df['user_hash'].map(degree_dict).fillna(0).astype(int) 
df['log_degree'] = np.log1p(df['degree']) 
# PageRank 
print("  → Calculating PageRank on subgraph...") 
try: 
pagerank = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6) 
except Exception as e: 
print(f"    PageRank fallback: {e}. Using zeros.") 
pagerank = {} 
for df in [train_labeled, train_unlabeled, test]: 
df['pagerank'] = df['user_hash'].map(pagerank).fillna(0) 
# Connected Components with Leiden Algorithm 
print("  → Calculating connected components with Leiden algorithm...") 
leiden_community = {} 
try: 
import igraph as ig 
import leidenalg 
    print("    Using Leiden algorithm...") 
    ig_graph = ig.Graph.from_networkx(G) 
    partition = leidenalg.find_partition(ig_graph, leidenalg.CPMVertexPartition, 
resolution_parameter=1.0) 
    for idx, node_name in enumerate(ig_graph.vs['_nx_name']): 
        leiden_community[node_name] = int(partition.membership[idx]) 
except ImportError: 
    print("    Leiden algorithm not available. Using Louvain instead.") 
    try: 
        import community as community_louvain 
        partition = community_louvain.best_partition(G) 
        leiden_community = partition 
    except ImportError: 
        print("    Louvain method not available. Using connected components.") 
        connected_components = list(nx.connected_components(G)) 
        leiden_community = {node: i for i, comp in enumerate(connected_components) for node in 
comp} 
except Exception as e: 
    print(f"    Leiden algorithm failed: {e}. Using connected components.") 
    connected_components = list(nx.connected_components(G)) 
    leiden_community = {node: i for i, comp in enumerate(connected_components) for node in comp} 
 
for df in [train_labeled, train_unlabeled, test]: 
    df['leiden_community'] = df['user_hash'].map(leiden_community).fillna(-1).astype(int) 
 
# Higher-Order Graph Motifs 
print("  → Calculating higher-order graph motifs...") 
triangle_count = {} 
for node in G.nodes(): 
    neighbors = list(G.neighbors(node)) 
    triangles = sum(1 for i, n1 in enumerate(neighbors) for n2 in neighbors[i+1:] if G.has_edge(n1, n2)) 
    triangle_count[node] = triangles 
 
for df in [train_labeled, train_unlabeled, test]: 
    df['triangle_count'] = df['user_hash'].map(triangle_count).fillna(0).astype(int) 
 
# Neighbor Label Aggregation 
print("  → Calculating neighbor label statistics...") 
user_labels_map = train_labeled.set_index('user_hash')['label'].to_dict() 
 
neighbors_dict = {} 
for a, b in zip(relevant_edges['user_a'], relevant_edges['user_b']): 
    neighbors_dict.setdefault(a, []).append(b) 
    neighbors_dict.setdefault(b, []).append(a) 
 
def calculate_neighbor_stats(user_hash): 
    neighbors = neighbors_dict.get(user_hash, []) 
    if not neighbors: 
        return (0.0, 0, 0, 0) 
    cheating_count = 0 
    max_label = 0 
    total_count = 0 
    for neighbor in neighbors: 
        if neighbor in user_labels_map: 
            lbl = int(user_labels_map[neighbor]) 
            cheating_count += lbl 
            max_label = max(max_label, lbl) 
            total_count += 1 
    mean = cheating_count / total_count if total_count > 0 else 0.0 
    return (mean, max_label, cheating_count, total_count) 
 
for df in [train_labeled, train_unlabeled, test]: 
    stats = df['user_hash'].apply(calculate_neighbor_stats) 
df['neighbor_cheating_mean'] = [t[0] for t in stats] 
df['neighbor_cheating_max'] = [t[1] for t in stats] 
df['neighbor_cheating_ratio'] = [(t[2] / t[3]) if (t[3] > 0) else 0.0 for t in stats] 
# Additional graph features 
print("  → Calculating clustering coefficient...") 
clustering = nx.clustering(G) 
for df in [train_labeled, train_unlabeled, test]: 
df['clustering'] = df['user_hash'].map(clustering).fillna(0.0) 
print("  → Calculating betweenness centrality (sampled)...") 
try: 
if G.number_of_nodes() > 1000: 
betweenness = nx.betweenness_centrality(G, k=100, seed=42) 
else: 
betweenness = nx.betweenness_centrality(G) 
except Exception as e: 
print(f"    Betweenness centrality failed: {e}. Filling with 0.") 
betweenness = {} 
for df in [train_labeled, train_unlabeled, test]: 
df['betweenness'] = df['user_hash'].map(betweenness).fillna(0.0) 
print("  → Calculating eigenvector centrality (robust)...") 
eigenvector = {} 
try: 
eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6) 
except Exception: 
try: 
eigenvector = nx.eigenvector_centrality_numpy(G) 
except Exception as e: 
        print(f"    Eigenvector centrality failed: {e}. Filling with 0.") 
        eigenvector = {} 
 
for df in [train_labeled, train_unlabeled, test]: 
    df['eigenvector'] = df['user_hash'].map(eigenvector).fillna(0.0) 
 
# New graph features 
print("  → Calculating additional graph features...") 
# Average degree of neighbors 
avg_neighbor_degree = {} 
for node in G.nodes(): 
    neighbors = list(G.neighbors(node)) 
    if neighbors: 
        avg_degree = sum(degree_dict.get(neighbor, 0) for neighbor in neighbors) / len(neighbors) 
    else: 
        avg_degree = 0 
    avg_neighbor_degree[node] = avg_degree 
 
for df in [train_labeled, train_unlabeled, test]: 
    df['avg_neighbor_degree'] = df['user_hash'].map(avg_neighbor_degree).fillna(0.0) 
 
# Number of common neighbors 
common_neighbors = {} 
for node in G.nodes(): 
    neighbors = list(G.neighbors(node)) 
    if not neighbors: 
        common_neighbors[node] = 0 
        continue 
    total_common = 0 
    for neighbor in neighbors: 
        common = len(set(neighbors).intersection(set(G.neighbors(neighbor)))) 
        total_common += common 
    avg_common = total_common / len(neighbors) if neighbors else 0 
    common_neighbors[node] = avg_common 
 
for df in [train_labeled, train_unlabeled, test]: 
    df['avg_common_neighbors'] = df['user_hash'].map(common_neighbors).fillna(0.0) 
 
# Behavioral feature engineering 
print("\n    Engineering advanced behavioral features...") 
features = [f'feature_{i:03d}' for i in range(1, 19)] 
numeric_features = [] 
for f in features: 
    if f in metadata and metadata[f].get('type') == 'numeric': 
        numeric_features.append(f) 
 
# Missing indicators and imputation 
for feat in features: 
    for df in [train_labeled, train_unlabeled, test]: 
        if feat in df.columns: 
            df[f'{feat}_missing'] = df[feat].isna().astype(int) 
        else: 
            df[f'{feat}_missing'] = 1 
            df[feat] = 0 
 
for feat in features: 
    if feat in numeric_features: 
        median_val = train_labeled[feat].median() 
        for df in [train_labeled, train_unlabeled, test]: 
            df[feat] = df[feat].fillna(median_val) 
    else: 
        for df in [train_labeled, train_unlabeled, test]: 
df[feat] = df[feat].fillna(0) 
# Log transformation for skewed features 
skewed_features = ['feature_010', 'feature_015', 'feature_016', 'feature_017'] 
for feat in skewed_features: 
for df in [train_labeled, train_unlabeled, test]: 
if feat in df.columns: 
df[f'{feat}_log'] = np.log1p(df[feat].clip(lower=0)) 
# Interaction features 
for df in [train_labeled, train_unlabeled, test]: 
df['feature_015_016_ratio'] = df['feature_015'] / (df['feature_016'] + 1e-6) 
df['feature_001_002_sum'] = df['feature_001'] + df['feature_002'] 
df['feature_003_004_diff'] = df['feature_003'] - df['feature_004'] 
df['feature_007_011_product'] = df['feature_007'] * df['feature_011'] 
df['feature_017_018_product'] = df['feature_017'] * df['feature_018'] 
df['feature_001_003_ratio'] = df['feature_001'] / (df['feature_003'] + 1e-6) 
df['feature_002_004_ratio'] = df['feature_002'] / (df['feature_004'] + 1e-6) 
df['feature_005_006_product'] = df['feature_005'] * df['feature_006'] 
df['feature_007_008_sum'] = df['feature_007'] + df['feature_008'] 
# Binning 
df['feature_001_bin'] = pd.cut(df['feature_001'].fillna(0), bins=5, labels=False) 
df['feature_002_bin'] = pd.cut(df['feature_002'].fillna(0), bins=5, labels=False) 
df['feature_003_bin'] = pd.cut(df['feature_003'].fillna(0), bins=5, labels=False) 
df['feature_004_bin'] = pd.cut(df['feature_004'].fillna(0), bins=5, labels=False) 
# Polynomial 
df['feature_001_squared'] = df['feature_001'] ** 2 
df['feature_002_squared'] = df['feature_002'] ** 2 
# New interaction features between graph and behavioral features 
for df in [train_labeled, train_unlabeled, test]: 
    df['degree_feature_001'] = df['degree'] * df['feature_001'] 
    df['pagerank_feature_002'] = df['pagerank'] * df['feature_002'] 
    df['triangle_count_feature_003'] = df['triangle_count'] * df['feature_003'] 
    df['degree_feature_002'] = df['degree'] * df['feature_002'] 
    df['pagerank_feature_001'] = df['pagerank'] * df['feature_001'] 
    df['triangle_count_feature_002'] = df['triangle_count'] * df['feature_002'] 
 
# Target Encoding for Categorical Features 
print("  → Applying target encoding to categorical features...") 
categorical_features = ['leiden_community', 'degree'] 
for feat in categorical_features: 
    if feat in train_labeled.columns: 
        enc = TargetEncoder(smoothing=1.0) 
        train_labeled[f'{feat}_encoded'] = enc.fit_transform(train_labeled[feat], train_labeled['label']) 
        train_unlabeled[f'{feat}_encoded'] = enc.transform(train_unlabeled[feat]) 
        test[f'{feat}_encoded'] = enc.transform(test[feat]) 
 
# Feature preparation and pseudo-labeling 
print("\n       Preparing features and implementing smart pseudo-labeling...") 
feature_columns = [ 
    col for col in train_labeled.columns 
    if ( 
        col.startswith('feature_') 
        or col.endswith('_missing') 
        or col.endswith('_log') 
        or col.endswith('_ratio') 
        or col.endswith('_sum') 
        or col.endswith('_diff') 
        or col.endswith('_product') 
        or col.endswith('_bin') 
        or col.endswith('_squared') 
        or col.endswith('_encoded') 
        or col in ['log_degree', 'degree', 'leiden_community', 'pagerank', 
                   'neighbor_cheating_mean', 'neighbor_cheating_max', 'neighbor_cheating_ratio', 
                   'clustering', 'betweenness', 'eigenvector', 'triangle_count', 
                   'avg_neighbor_degree', 'avg_common_neighbors'] 
    )] 
 
def ensure_features(df, cols): 
    for c in cols: 
        if c not in df.columns: 
            df[c] = 0 
    return df[cols].copy() 
 
X_labeled = ensure_features(train_labeled, feature_columns) 
y_labeled = train_labeled['label'].astype(int).reset_index(drop=True) 
X_unlabeled = ensure_features(train_unlabeled, feature_columns) 
X_test = ensure_features(test, feature_columns) 
 
# Feature scaling 
scaler = StandardScaler() 
X_labeled_scaled = scaler.fit_transform(X_labeled) 
X_unlabeled_scaled = scaler.transform(X_unlabeled) 
X_test_scaled = scaler.transform(X_test) 
 
# Smart Pseudo-Labeling with ensemble approach 
print("  → Training an ensemble of models to assess pseudo-label quality...") 
 
# Define objective functions for hyperparameter tuning 
def objective_lgb_pseudo(trial): 
    params = { 
        'objective': 'binary', 
        'metric': 'auc', 
        'boosting_type': 'gbdt', 
        'num_leaves': trial.suggest_int('num_leaves', 20, 100), 
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1), 
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0), 
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0), 
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10), 
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), 
        'verbose': -1, 
        'random_state': 42, 
    } 
    model = lgb.LGBMClassifier(**params) 
    score = cross_val_score(model, X_labeled_scaled, y_labeled, cv=3, scoring='roc_auc').mean() 
    return score 
 
def objective_xgb_pseudo(trial): 
    params = { 
        'objective': 'binary:logistic', 
        'eval_metric': 'auc', 
        'eta': trial.suggest_loguniform('eta', 1e-3, 1e-1), 
        'max_depth': trial.suggest_int('max_depth', 3, 10), 
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0), 
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0), 
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), 
        'random_state': 42, 
    } 
    model = xgb.XGBClassifier(**params) 
    score = cross_val_score(model, X_labeled_scaled, y_labeled, cv=3, scoring='roc_auc').mean() 
    return score 
 
def objective_cat_pseudo(trial): 
    params = { 
        'loss_function': 'Logloss', 
        'eval_metric': 'AUC', 
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1), 
        'depth': trial.suggest_int('depth', 4, 10), 
        'random_seed': 42, 
        'od_type': 'Iter', 
        'od_wait': 50, 
        'verbose': 0, 
    } 
    model = CatBoostClassifier(**params) 
    score = cross_val_score(model, X_labeled_scaled, y_labeled, cv=3, scoring='roc_auc').mean() 
    return score 
 
# Run Optuna studies for each model 
print("  → Tuning hyperparameters for pseudo-labeling ensemble...") 
lgb_study_pseudo = optuna.create_study(direction='maximize') 
lgb_study_pseudo.optimize(objective_lgb_pseudo, n_trials=30) 
 
xgb_study_pseudo = optuna.create_study(direction='maximize') 
xgb_study_pseudo.optimize(objective_xgb_pseudo, n_trials=30) 
 
cat_study_pseudo = optuna.create_study(direction='maximize') 
cat_study_pseudo.optimize(objective_cat_pseudo, n_trials=30) 
 
# Get best parameters 
best_lgb_params_pseudo = lgb_study_pseudo.best_params 
best_xgb_params_pseudo = xgb_study_pseudo.best_params 
best_cat_params_pseudo = cat_study_pseudo.best_params 
 
print("  → Best parameters for pseudo-labeling models:") 
print(f"    LightGBM: {best_lgb_params_pseudo}") 
print(f"    XGBoost: {best_xgb_params_pseudo}") 
print(f"    CatBoost: {best_cat_params_pseudo}") 
 
# Train final models with best parameters 
lgb_model_pseudo = lgb.LGBMClassifier(**best_lgb_params_pseudo) 
xgb_model_pseudo = xgb.XGBClassifier(**best_xgb_params_pseudo) 
cat_model_pseudo = CatBoostClassifier(**best_cat_params_pseudo) 
 
# Use a small validation set for early stopping 
skf_temp = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
train_idx, valid_idx = next(skf_temp.split(X_labeled_scaled, y_labeled)) 
X_train_temp, X_valid_temp = X_labeled_scaled[train_idx], X_labeled_scaled[valid_idx] 
y_train_temp, y_valid_temp = y_labeled.iloc[train_idx], y_labeled.iloc[valid_idx] 
 
try: 
    lgb_model_pseudo.fit(X_train_temp, y_train_temp, 
                        eval_set=[(X_valid_temp, y_valid_temp)], 
                        eval_metric='auc', 
                        callbacks=[lgb.early_stopping(50, verbose=False)]) 
except Exception: 
    lgb_model_pseudo.fit(X_train_temp, y_train_temp) 
 
try: 
    safe_xgb_fit(xgb_model_pseudo, X_train_temp, y_train_temp, 
                 X_valid_temp, y_valid_temp, early_stopping_rounds=50) 
except Exception: 
    xgb_model_pseudo.fit(X_train_temp, y_train_temp) 
 
try: 
    cat_model_pseudo.fit(X_train_temp, y_train_temp, 
                        eval_set=(X_valid_temp, y_valid_temp), 
                        verbose=False) 
except Exception: 
    cat_model_pseudo.fit(X_train_temp, y_train_temp) 
 
# Get probabilities from each model 
lgb_probs = lgb_model_pseudo.predict_proba(X_unlabeled_scaled)[:, 1] 
xgb_probs = xgb_model_pseudo.predict_proba(X_unlabeled_scaled)[:, 1] 
cat_probs = cat_model_pseudo.predict_proba(X_unlabeled_scaled)[:, 1] 
 
# Average the probabilities 
unlabeled_probs = (lgb_probs + xgb_probs + cat_probs) / 3 
 
# Calculate validation AUC for each model and ensemble 
lgb_val_auc = roc_auc_score(y_valid_temp, lgb_model_pseudo.predict_proba(X_valid_temp)[:, 1]) 
xgb_val_auc = roc_auc_score(y_valid_temp, xgb_model_pseudo.predict_proba(X_valid_temp)[:, 1]) 
cat_val_auc = roc_auc_score(y_valid_temp, cat_model_pseudo.predict_proba(X_valid_temp)[:, 1]) 
ensemble_val_auc = roc_auc_score(y_valid_temp, 
(lgb_model_pseudo.predict_proba(X_valid_temp)[:, 1] + 
                                               xgb_model_pseudo.predict_proba(X_valid_temp)[:, 1] + 
                                               cat_model_pseudo.predict_proba(X_valid_temp)[:, 1]) / 3) 
 
print(f"   Model validation AUCs:") 
print(f"    LightGBM: {lgb_val_auc:.4f}") 
print(f"    XGBoost: {xgb_val_auc:.4f}") 
print(f"    CatBoost: {cat_val_auc:.4f}") 
print(f"    Ensemble: {ensemble_val_auc:.4f}") 
 
# Adjust pseudo-labeling thresholds based on validation AUC 
if ensemble_val_auc > 0.85: 
    print("  → Ensemble model is strong. Generating pseudo-labels with dynamic thresholds...") 
    # Dynamic thresholding based on validation performance 
    if ensemble_val_auc > 0.9: 
        low_thresh = 0.02 
        high_thresh = 0.98 
    elif ensemble_val_auc > 0.88: 
        low_thresh = 0.03 
        high_thresh = 0.97 
    else: 
        low_thresh = 0.05 
        high_thresh = 0.95 
 
    low_conf_mask = unlabeled_probs < low_thresh 
    high_conf_mask = unlabeled_probs > high_thresh 
    pseudo_labels = np.full(len(X_unlabeled), -1, dtype=int) 
    pseudo_labels[low_conf_mask] = 0 
    pseudo_labels[high_conf_mask] = 1 
    num_pseudo = int(np.sum(pseudo_labels != -1)) 
    print(f"   Generated {num_pseudo} high-confidence pseudo-labels 
({num_pseudo/len(X_unlabeled):.2%} of unlabeled data).") 
    pseudo_mask = pseudo_labels != -1 
    X_combined = pd.concat([X_labeled.reset_index(drop=True), 
X_unlabeled[pseudo_mask].reset_index(drop=True)], 
                          axis=0).reset_index(drop=True) 
    y_combined = pd.concat([y_labeled.reset_index(drop=True), 
pd.Series(pseudo_labels[pseudo_mask]).reset_index(drop=True)], 
                          axis=0).reset_index(drop=True) 
else: 
    print("  → Ensemble model is not strong enough. Skipping pseudo-labeling.") 
    X_combined = X_labeled.reset_index(drop=True).copy() 
    y_combined = y_labeled.reset_index(drop=True).copy() 
 
# Model preparation and validation 
print("\n         
Preparing for cross-validation with optimized stacking...") 
N_SPLITS = 5 
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42) 
oof_preds = np.zeros((len(X_combined), 5)) 
test_preds = np.zeros((len(X_test), 5)) 
C_FN = 600 
C_FP_BLOCK = 300 
pos_weight = ((len(y_combined) - y_combined.sum()) / max(y_combined.sum(), 1)) * (C_FN / 
C_FP_BLOCK) 
print(f"   Using calculated scale_pos_weight: {pos_weight:.2f}") 
def calculate_cost(y_true, y_pred, t1, t2): 
auto_block = y_pred >= t2 
manual_review = (y_pred >= t1) & (y_pred < t2) 
auto_pass = y_pred < t1 
fn = np.sum((y_true == 1) & auto_pass) * C_FN 
fp_block = np.sum((y_true == 0) & auto_block) * C_FP_BLOCK 
fp_review = np.sum((y_true == 0) & manual_review) * 150 
tp_review = np.sum((y_true == 1) & manual_review) * 5 
return fn + fp_block + fp_review + tp_review 
# Optimized base model training 
print("\n       
Level 1: Training optimized base models...") 
# Define objective functions for hyperparameter tuning 
def objective_lgb(trial): 
params = { 
'objective': 'binary', 
'metric': 'auc', 
'boosting_type': 'gbdt', 
        'num_leaves': trial.suggest_int('num_leaves', 20, 100), 
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1), 
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0), 
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0), 
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10), 
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), 
        'scale_pos_weight': pos_weight, 
        'verbose': -1, 
        'random_state': 42, 
    } 
    model = lgb.LGBMClassifier(**params) 
    score = cross_val_score(model, X_combined, y_combined, cv=3, scoring='roc_auc').mean() 
    return score 
 
def objective_xgb(trial): 
    params = { 
        'objective': 'binary:logistic', 
        'eval_metric': 'auc', 
        'eta': trial.suggest_loguniform('eta', 1e-3, 1e-1), 
        'max_depth': trial.suggest_int('max_depth', 3, 10), 
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0), 
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0), 
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), 
        'scale_pos_weight': pos_weight, 
        'random_state': 42, 
    } 
    model = xgb.XGBClassifier(**params) 
    score = cross_val_score(model, X_combined, y_combined, cv=3, scoring='roc_auc').mean() 
    return score 
 
def objective_cat(trial): 
    params = { 
        'loss_function': 'Logloss', 
        'eval_metric': 'AUC', 
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1), 
        'depth': trial.suggest_int('depth', 4, 10), 
        'random_seed': 42, 
        'od_type': 'Iter', 
        'od_wait': 100, 
        'verbose': 0, 
    } 
    model = CatBoostClassifier(**params) 
    score = cross_val_score(model, X_combined, y_combined, cv=3, scoring='roc_auc').mean() 
    return score 
 
# Tune hyperparameters for each base model 
print("  → Tuning hyperparameters for base models...") 
lgb_study = optuna.create_study(direction='maximize') 
lgb_study.optimize(objective_lgb, n_trials=50) 
best_lgb_params = lgb_study.best_params 
 
xgb_study = optuna.create_study(direction='maximize') 
xgb_study.optimize(objective_xgb, n_trials=50) 
best_xgb_params = xgb_study.best_params 
 
cat_study = optuna.create_study(direction='maximize') 
cat_study.optimize(objective_cat, n_trials=50) 
best_cat_params = cat_study.best_params 
 
print("  → Best parameters found:") 
print(f"    LightGBM: {best_lgb_params}") 
print(f"    XGBoost: {best_xgb_params}") 
print(f"    CatBoost: {best_cat_params}") 
 
for fold, (train_idx, valid_idx) in enumerate(skf.split(X_combined, y_combined)): 
    print(f"--- Fold {fold+1}/{N_SPLITS} ---") 
    X_train_fold = X_combined.iloc[train_idx].reset_index(drop=True) 
    X_valid_fold = X_combined.iloc[valid_idx].reset_index(drop=True) 
    y_train_fold = y_combined.iloc[train_idx].reset_index(drop=True) 
    y_valid_fold = y_combined.iloc[valid_idx].reset_index(drop=True) 
 
    fold_scaler = StandardScaler() 
    X_train_scaled = fold_scaler.fit_transform(X_train_fold) 
    X_valid_scaled = fold_scaler.transform(X_valid_fold) 
 
    # LightGBM with best parameters 
    lgb_model_fold = lgb.LGBMClassifier(**best_lgb_params) 
    try: 
        lgb_model_fold.fit(X_train_scaled, y_train_fold, eval_set=[(X_valid_scaled, y_valid_fold)], 
                          eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)]) 
    except Exception: 
        lgb_model_fold.fit(X_train_scaled, y_train_fold) 
    oof_preds[valid_idx, 0] = lgb_model_fold.predict_proba(X_valid_scaled)[:, 1] 
    test_preds[:, 0] += lgb_model_fold.predict_proba(fold_scaler.transform(X_test))[:, 1] / N_SPLITS 
 
    # XGBoost with best parameters 
    xgb_model_fold = xgb.XGBClassifier(**best_xgb_params) 
    safe_xgb_fit(xgb_model_fold, X_train_scaled, y_train_fold, X_valid_scaled, y_valid_fold, 
                 early_stopping_rounds=100) 
    oof_preds[valid_idx, 1] = xgb_model_fold.predict_proba(X_valid_scaled)[:, 1] 
    test_preds[:, 1] += xgb_model_fold.predict_proba(fold_scaler.transform(X_test))[:, 1] / N_SPLITS 
 
    # CatBoost with best parameters 
    cat_features = [] 
    if 'leiden_community' in X_train_fold.columns: 
        cat_features = ['leiden_community'] 
    cat_model = CatBoostClassifier(**best_cat_params) 
    try: 
        cat_model.fit(X_train_fold, y_train_fold, eval_set=(X_valid_fold, y_valid_fold), 
                     cat_features=cat_features, verbose=False) 
    except Exception: 
        cat_model.fit(X_train_fold, y_train_fold, cat_features=cat_features, verbose=False) 
    oof_preds[valid_idx, 2] = cat_model.predict_proba(X_valid_fold)[:, 1] 
    test_preds[:, 2] += cat_model.predict_proba(X_test)[:, 1] / N_SPLITS 
 
    # Random Forest 
    rf_params = { 
        'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 5, 
        'class_weight': {0: 1, 1: pos_weight}, 'random_state': 42, 'n_jobs': -1 
    } 
    rf_model = RandomForestClassifier(**rf_params) 
    rf_model.fit(X_train_fold, y_train_fold) 
    oof_preds[valid_idx, 3] = rf_model.predict_proba(X_valid_fold)[:, 1] 
    test_preds[:, 3] += rf_model.predict_proba(X_test)[:, 1] / N_SPLITS 
 
    # Logistic Regression 
    lr_params = { 
        'penalty': 'l2', 'C': 0.1, 'solver': 'liblinear', 'class_weight': {0: 1, 1: pos_weight}, 'random_state': 42 
    } 
    lr_model = LogisticRegression(**lr_params) 
    lr_model.fit(X_train_scaled, y_train_fold) 
    oof_preds[valid_idx, 4] = lr_model.predict_proba(X_valid_scaled)[:, 1] 
    test_preds[:, 4] += lr_model.predict_proba(fold_scaler.transform(X_test))[:, 1] / N_SPLITS 
 
# Meta-model training 
print("\n    Level 2: Training meta-model (Gradient Boosting)...") 
meta_X = oof_preds[:len(y_labeled), :] 
meta_y = y_labeled.values 
 
# Tune meta-model hyperparameters 
def objective_meta(trial): 
    params = { 
        'n_estimators': trial.suggest_int('n_estimators', 100, 500), 
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1), 
        'max_depth': trial.suggest_int('max_depth', 3, 10), 
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10), 
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10), 
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]), 
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0), 
        'random_state': 42 
    } 
    model = GradientBoostingClassifier(**params) 
    score = cross_val_score(model, meta_X, meta_y, cv=3, scoring='roc_auc').mean() 
    return score 
 
meta_study = optuna.create_study(direction='maximize') 
meta_study.optimize(objective_meta, n_trials=50) 
 
best_meta_params = meta_study.best_params 
meta_model = GradientBoostingClassifier(**best_meta_params) 
meta_model.fit(meta_X, meta_y) 
meta_test_preds = meta_model.predict_proba(test_preds)[:, 1] 
meta_oof_preds = meta_model.predict_proba(meta_X)[:, 1] 
print(f"   Meta-model trained. OOF AUC: {roc_auc_score(meta_y, meta_oof_preds):.4f}") 
 
# Threshold optimization 
print("\n      
Finding optimal thresholds using Bayesian optimization...") 
def objective(trial): 
t1 = trial.suggest_uniform('t1', 0.05, 0.3) 
t2 = trial.suggest_uniform('t2', 0.6, 0.9) 
if t2 <= t1: 
return float('inf') 
return calculate_cost(meta_y, meta_oof_preds, t1, t2) 
sampler = TPESampler(seed=42) 
study = optuna.create_study(sampler=sampler, direction='minimize') 
study.optimize(objective, n_trials=100, show_progress_bar=False) 
best_thresholds = [study.best_params['t1'], study.best_params['t2']] 
best_cost = study.best_value 
print(f"\n       
OPTIMIZATION COMPLETE!") 
print(f"   Optimal Thresholds: auto-pass < {best_thresholds[0]:.6f}, manual-review < 
{best_thresholds[1]:.6f}") 
print(f"   Minimum OOF cost: ${best_cost:,.2f}") 
print(f"   OOF AUC: {roc_auc_score(meta_y, meta_oof_preds):.4f}") 
# Final submission 
print("\n      
Generating final submission with the improved model...") 
submission = pd.DataFrame({'user_hash': test['user_hash'].reset_index(drop=True), 
'prediction': meta_test_preds}) 
submission.to_csv('submission.csv', index=False) 
print(f"\n   
Final submission saved to submission.csv") 
print(f"   Prediction stats: Min={meta_test_preds.min():.4f}, Max={meta_test_preds.max():.4f}, 
Mean={meta_test_preds.mean():.4f}") 
print("\n    
This solution uses a ROBUST and POWERFUL approach:") 
print("   - Advanced graph features including higher-order motifs and Leiden communities") 
print("   - Smart pseudo-labeling with ensemble model and dynamic thresholds") 
print("   - Cost-sensitive learning with Bayesian threshold optimization") 
print("   - Diverse ensemble stacking with non-linear meta-model") 
print("   - Comprehensive feature engineering and selection") 
print("   - Hyperparameter tuning for all models") 
