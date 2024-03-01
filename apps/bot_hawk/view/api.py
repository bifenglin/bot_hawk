import pickle
from datetime import datetime
import os

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, current_app, request, abort
from bot_hawk.apps.utils.constants import METHODTYPE
from bot_hawk.apps.utils.github import get_user_info_by_name, get_user_info_by_id
from sklearn.preprocessing import MinMaxScaler

from apps.utils.ClickHouseUtils import ClickHouseUtils
from apps.utils.TFIDFUtil import TFIDFUtil
from statsmodels.tsa.stattools import acf

api = Blueprint('api', __name__, url_prefix='/api')


@api.route('/', methods=[METHODTYPE.GET, METHODTYPE.POST])
def api_index():
    current_app.logger.info(f'{request.method} api.index')
    if request.method == METHODTYPE.GET:
        data = request.args
        return jsonify({"success": True, "name": 'api.index', 'data': data})
    else:
        data = request.json     # for request that POST with application/json
        return jsonify({"success": True, "name": 'api.index', 'data': data})

@api.route('/user', methods=[METHODTYPE.GET, METHODTYPE.POST])
def search():
    current_app.logger.info(f'{request.method} api.search')

    base_dir = os.path.dirname(__file__)  # 获取当前文件的目录
    model_path = os.path.join(base_dir, '..', '..', '..', 'training', 'model', 'baggingRandomForest.pickle')

    # 加载预训练的模型
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    if request.method == METHODTYPE.GET:
        data = request.args
    else:
        data = request.json     # for request that POST with application/json
    # name = data.get("name")
    account_type = data.get("accountType", "username")  # 默认为"username"
    account = data.get("account")

    # 根据account_type决定调用的函数
    if account_type == "username":
        user_info = analyze_user_by_name(account)
    else:
        user_info = analyze_user_by_id(account)


    print('user_info', user_info)
    # 限制user信息只包含指定的列，并更新列名
    columns_mapping = {
        "login": "login",
        "name": "name",
        "email": "email",
        "bio": "bio",
        "followers": "Number of followers",
        "following": "Number of following",
        "similarity": "tfidf_similarity",
        "event_count": "Number of Activity",
        "issue_count": "Number of Issue",
        "pull_request_count": "Number of Pull Request",
        "repo_count": "Number of Repository",
        "push_count": "Number of Commit",
        "active_days": "Number of Active day",
        "fft": "Periodicity of Activities",
        "connect_accounts": "Number of Connection Account",
        "response_time": "Median Response Time"
    }
    user = {new_key: user_info[old_key] for old_key, new_key in columns_mapping.items() if old_key in user_info}

    # # 更新需要归一化的字段列表
    # normalize_columns = [
    #     "Number of followers", "Number of following", "tfidf_similarity", "Number of Activity",
    #     "Number of Issue", "Number of Pull Request", "Number of Repository", "Number of Commit",
    #     "Number of Active day", "Periodicity of Activities", "Number of Connection Account", "Median Response Time"
    # ]

    # # 提取需要归一化的数据
    # features_to_normalize = np.array([[user[col] for col in normalize_columns if col in user]], dtype=float)
    #
    # # min-max归一化
    # scaler = MinMaxScaler()
    # features_normalized = scaler.fit_transform(features_to_normalize).flatten()

    # # 将归一化后的数据整合回user字典
    # for i, col in enumerate(normalize_columns):
    #     if col in user:
    #         user[col] = features_normalized[i]

    # 将user信息转换为DataFrame以便模型使用
    user_df = pd.DataFrame([user])
    print(user_df)
    # 预测
    # 注意: 根据您的模型和数据结构，您可能需要对user_df进行适当的预处理
    prediction = model.predict(user_df)
    # 修改这里，将numpy数组转换为列表
    prediction_list = prediction.tolist()
    return jsonify({"success": True, 'data': user, 'prediction_list':prediction_list})

def analyze_user_by_id(id):
    user_info = get_user_info_by_id(id, current_app.config['GITHUB_API_TOKEN'])

    for user in user_info:
        print(user)

    user = clickhouse_quey(user)
    bot_str = ['bot ', '-ci', '-io', '-cla', '-bot', '-test', 'bot@']
    user['login'] = 1 if any(s in user['login'].lower() for s in bot_str) else 0
    user['name'] = 1 if any(s in user['name'].lower() for s in bot_str) else 0
    user['bio'] = 1 if any(s in user['bio'].lower() for s in bot_str) else 0
    user['email'] = 1 if any(s in user['email'].lower() for s in bot_str) else 0

    return user

def analyze_user_by_name(name):
    user_info = get_user_info_by_name(name, current_app.config['GITHUB_API_TOKEN'])
    for user in user_info:
        print(user)
    user = clickhouse_quey(user)
    bot_str = ['bot ', '-ci', '-io', '-cla', '-bot', '-test', 'bot@']
    user['login'] = 1 if any(s in user['login'].lower() for s in bot_str) else 0
    user['name'] = 1 if any(s in user['name'].lower() for s in bot_str) else 0
    user['bio'] = 1 if any(s in user['bio'].lower() for s in bot_str) else 0
    user['email'] = 1 if any(s in user['email'].lower() for s in bot_str) else 0

    return user

def clickhouse_quey(user):
    clickhouse = ClickHouseUtils(
        host=current_app.config['CLICKHOUSE_HOST'],
        user=current_app.config['CLICKHOUSE_USER'],
        password=current_app.config['CLICKHOUSE_PASSWORD'],
        database=current_app.config['CLICKHOUSE_DATABASE']
    )

    query = '''
    SELECT body 
    FROM events 
    where 
        body != ''
    and
        actor_id = {id} 
    limit 100
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    # for row in result:
    #     print(row)

    tfidf_util = TFIDFUtil()
    documents = result
    similarity = tfidf_util.calculate_similarity(documents)
    user['similarity'] = similarity

    query = '''
    SELECT count(*)
    FROM events
    where
        actor_id = {id}
    '''.format(id=user['id'])

    result = clickhouse.execute_query(query)
    user['event_count'] = result[0][0]

    query = '''
    SELECT count(*)
    FROM events
    where
        actor_id = {id}
    and type = 'IssuesEvent' and type='IssueCommentEvent'
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    user['issue_count'] = result[0][0]

    query = '''
    SELECT count(*)
    FROM events
    where
        actor_id = {id}
    and type = 'PullRequestEvent' and type='PullRequestReviewEvent' and type='PullRequestReviewCommentEvent'
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    user['pull_request_count'] = result[0][0]

    #Number of Repository
    query = '''
    SELECT count(distinct(repo_id))
    FROM events
    where
        actor_id = {id}
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    user['repo_count'] = result[0][0]

    query = '''
    SELECT count(distinct(repo_id))
    FROM events
    where
        actor_id = {id}
    and type= 'PushEvent'
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    user['push_count'] = result[0][0]

    query = '''
    SELECT
        count(*) as active_days
    FROM
        (
        SELECT
            toDate(created_at) as date
        FROM
            events
        where
            actor_id = {id}
            AND created_at >= toDate(now()) - 365
        GROUP BY
            date
		)
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    print(result)
    user['active_days'] = result[0][0]
    # user['event_count']/user['active_days']


    query = '''
    SELECT
        created_at
    FROM
       events
    where
        actor_id = {id}
        AND created_at >= toDate(now()) - 365
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    # 将这些日期转换为DataFrame
    df = pd.DataFrame(result, columns=['created_at'])

    # 为简化，我们假设每个日期对应一个事件，生成每日计数
    # 实际情况下，你应该使用从数据库获取的真实事件数据
    df['count'] = 1

    # 将'created_at'设置为索引并对日期进行分组计数
    df.set_index('created_at', inplace=True)
    daily_counts = df.resample('D').count()

    # 使用之前的FFT分析代码
    # 计算FFT
    fft_result = np.fft.fft(daily_counts['count'].values)

    # 计算频率
    freq = np.fft.fftfreq(len(fft_result))

    # 找到幅度最大的成分（除去直流分量即freq=0的情况）
    idx = np.argmax(np.abs(fft_result[1:])) + 1  # 加1是因为我们跳过了第一个成分

    # 最显著的周期性频率
    dominant_freq = freq[idx]

    # 计算周期（时间单位取决于原始数据的时间单位）
    period = 1 / dominant_freq if dominant_freq != 0 else 0  # 避免除以零的错误

    # 幅度作为周期性的强度指标
    amplitude = np.abs(fft_result[idx])

    print(f"主要周期: {period} 时间单位")
    print(f"周期性强度（幅度）: {amplitude}")
    # Assuming `user` is a dictionary where you want to store the ACF values
    user['fft'] = amplitude  # Store the ACF values as a list

    query = '''
    SELECT 
    SUM(other_actor_count) AS total_other_actor_count
    FROM (
        SELECT 
            issue_id,
            COUNT(DISTINCT actor_id) - 1 AS other_actor_count
        FROM 
            events
        WHERE 
            issue_id IN (
                SELECT DISTINCT issue_id
                FROM events
                WHERE 
                    actor_id = 15813364
                    AND (type='IssuesEvent'
                        OR type='PullRequestEvent'
                        OR type='IssueCommentEvent'
                        OR type='PullRequestReviewCommentEvent')
            )
            AND actor_id != 15813364
        GROUP BY 
            issue_id
    ) AS subquery
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    # print(result)
    user['connect_accounts'] = result[0][0]

    query = '''
    SELECT 
    AVG(dateDiff('second', NULLIF(prev_created_at, '1970-01-01 00:00:00'), created_at)) AS avg_time_diff
    FROM (
        SELECT 
            actor_id,
            issue_id,
            created_at,
            lagInFrame(created_at) OVER (PARTITION BY actor_id, issue_id ORDER BY created_at) AS prev_created_at
        FROM 
            events
        WHERE
            actor_id = {id} 
            AND (type='IssuesEvent'
                 OR type='PullRequestEvent'
                 OR type='IssueCommentEvent'
                 OR type='PullRequestReviewCommentEvent')
    ) AS subquery
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    print(result)
    user['response_time'] = result[0][0]


    return user


@api.route('/upload', methods=[METHODTYPE.POST])
def api_upload():
    current_app.logger.info(f'{request.method} api.upload')
    if request.method == METHODTYPE.GET:
        abort(405)

    files = request.files       # for request that POST with multipart/form-data's files
    data = request.form         # for request that POST with multipart/form-data's data
    return jsonify({"success": True, "name": 'api.upload', 'data': data, 'files': files})
