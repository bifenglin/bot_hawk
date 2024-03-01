import pandas as pd
import pickle

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    precision_recall_curve, roc_curve, auc
import xgboost as xgb

def load_data():
    # read CSV file
    df = pd.read_csv("data/bothawk_data.csv")
    # df['Number of Connection Account'].astype('int64')
    df.dropna()
    #
    # # 定义归一化函数
    # normalize = lambda x: (x - x.min()) / (x.max() - x.min())
    #
    # # 对数据框中的各列进行归一化处理
    # df[["Number of followers", "Number of following", "Number of Activity",
    #     "Number of Issue", "Number of Pull Request", "Number of Repository", "Number of Commit",
    #     "Number of Active day", "Periodicity of Activities", "Number of Connection Account",
    #     "Median Response Time"]] = df[['Number of followers', 'Number of following', "Number of Activity", "Number of Issue",
    #                                    "Number of Pull Request", "Number of Repository", "Number of Commit",
    #                                    "Number of Active day", "Periodicity of Activities",
    #                                    "Number of Connection Account", "Median Response Time"]].apply(normalize)

    # 选取需要的特征与标签作为模型输入
    df = df[["login", "name", "email", "bio", "Number of followers", "Number of following", "tfidf_similarity", "Number of Activity",
             "Number of Issue", "Number of Pull Request", "Number of Repository", "Number of Commit",
             "Number of Active day", "Periodicity of Activities", "Number of Connection Account",
             "Median Response Time", 'label']]

    # 将标签“Bot”和“Human”映射为数值0和1
    bot_mapping = {'Human': 0, 'Bot': 1}
    df['label'] = df['label'].replace(bot_mapping)

    # 筛选出正样本和负样本
    pos_samples = df[df['label'] == 1]
    neg_samples = df[df['label'] == 0]

    # 随机下采样
    neg_samples = neg_samples.sample(n=pos_samples.shape[0], replace=False, random_state=42)

    # 合并正样本和负样本
    df = pd.concat([pos_samples, neg_samples])

    # 定义特征和标签
    X = df.drop('label', axis=1)
    y = df['label']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def preprocess_data(X):
    # 数据预处理（无需进行任何操作）
    return X


def get_evaluation_metrics(y_true, y_pred):
    # 计算评估指标（准确率、精确率、召回率、F1分数和混淆矩阵）
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, cm


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    base_clf = [
        ("DecisionTree", DecisionTreeClassifier()),
        ("KNeighbors", KNeighborsClassifier()),
        ("RandomForest", RandomForestClassifier()),
        ("XGBoost", xgb.XGBClassifier()),
        ("LogisticRegression", LogisticRegression()),
        ("SVC", SVC()),
        ("GaussianNB", GaussianNB())
    ]

    params = [
        {'base_estimator__max_depth': [3, 5, 7],
         'base_estimator__min_samples_split': [2, 4, 8]},
        {'base_estimator__n_neighbors': [3, 5, 7],
         'base_estimator__weights': ['uniform', 'distance']},
        {'base_estimator__n_estimators': [10, 50, 100],
         'base_estimator__max_depth': [3, 5, 7]},
        {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001]
        },
        {'base_estimator__C': [0.1, 1.0, 10.0]},
        {},
        {}
    ]

    eval_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC'])

    for i, (name, model) in enumerate(base_clf):
        print(f"Optimizing {name} with Bagging...")

        if i == 3:
            # XGBoost
            xgb_model = xgb.XGBClassifier()
            grid_search = GridSearchCV(xgb_model, params[i], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            with open('model/baggingXGBoost.pickle', 'wb') as f:
                pickle.dump(grid_search.best_estimator_, f)

            y_pred = grid_search.predict(X_test)
            y_prob = grid_search.predict_proba(X_test)[:, 1]
        else:
            # Define base estimator
            base_est = model

            # Define Bagging model with base estimator
            bagging = BaggingClassifier(base_est, random_state=0)

            # Define GridSearchCV object
            grid_search = GridSearchCV(bagging, params[i], cv=5, scoring='accuracy')

            # Fit the model
            grid_search.fit(X_train, y_train)

            # Save the best estimator
            with open(f'model/bagging{name}.pickle', 'wb') as f:
                pickle.dump(grid_search.best_estimator_, f)

            # Predict labels
            y_pred = grid_search.predict(X_test)
            y_prob = grid_search.predict_proba(X_test)[:, 1]

        # Compute evaluation metrics
        accuracy, precision, recall, f1, cm = get_evaluation_metrics(y_test, y_pred)

        # 计算ROC曲线的点
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

        pr_auc = auc(recall_curve, precision_curve)

        # 统计特征重要性
        perm_importance = permutation_importance(grid_search.best_estimator_, X_test, y_test, n_repeats=5,
                                                 random_state=0, n_jobs=-1)
        sorted_idx = perm_importance.importances_mean.argsort()

        # 将数据存入DataFrame中
        df_roc = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Thresholds': thresholds})
        df_pr = pd.DataFrame({'Precision': precision_curve, 'Recall': recall_curve})
        # df_perm_imp = pd.DataFrame({'Sorted Index': sorted_idx, 'Feature Importance': perm_importance.importances_mean})
        # 获取特征名称
        feature_names = X_train.columns.tolist()
        df_perm_imp = pd.DataFrame({'Feature': [feature_names[i] for i in sorted_idx],
                                    'Feature Importance': perm_importance.importances_mean})


        eval_df.loc['bagging' + name] = [accuracy, precision, recall, f1, pr_auc]

        if i == len(base_clf) - 1:
            eval_df.to_csv(f'./result/eva/bagging{name}_metrics.csv')
            # 将DataFrame写入CSV文件中
            df_roc.to_csv(f'./result/eva/bagging{name}_roc_curve_data.csv', index=False)
            df_pr.to_csv(f'./result/eva/bagging{name}_pr_curve_data.csv', index=False)
            df_perm_imp.to_csv(f'./result/eva/bagging{name}_perm_imp.csv', index=False)

        # y_pred = grid_search.predict(X_test)
        # y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
        #
        # results_df = pd.DataFrame({
        #     'y_test': y_test,
        #     'y_pred': y_pred,
        #     'y_pred_proba': y_pred_proba
        # })
        # results_df.to_csv(f'./result/eva/bagging{name}_test_results.csv', index=False)

        # 输出最佳模型所用的参数组合
        print(f"Best parameters for {name}: {grid_search.best_params_}")

        # 输出评估指标
        print(f"{name} Classifier Evaluation Metrics:")
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("Precision: %.2f%%" % (precision * 100.0))
        print("Recall: %.2f%%" % (recall * 100.0))
        print("F1-score: %.2f%%" % (f1 * 100.0))
        print("Confusion Matrix:")
        print(cm)


def main():
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()

    # 数据预处理
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    # 训练和评估模型
    train_and_evaluate_model(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
