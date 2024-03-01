import sys
import os
def init_env_path(_file_):
    package_dir = os.path.join(os.path.dirname(_file_), '../../')
    abs_path = os.path.abspath(package_dir)
    if abs_path not in sys.path:
        print(f'Add {abs_path} to python path')
        sys.path.insert(0, abs_path)

# 调用init_env_path来更新sys.path
init_env_path(__file__)

from bot_hawk.apps.utils.ClickHouseUtils import ClickHouseUtils
from bot_hawk.apps.utils.TFIDFUtil import TFIDFUtil
from github import Github
import numpy as np
import pandas as pd
def get_user_info_by_id(actor_ids, api_token):
    g = Github(api_token)
    users = []
    err_user = []

    if not isinstance(actor_ids, list):
        actor_ids = [actor_ids]

    for actor_id in actor_ids:
        try:
            user = g.get_user_by_id(actor_id)
            user_dict = {
                'id': user.id,
                'login': user.login,
                'name': user.name if user.name is not None else '',
                'email': user.email if user.email is not None else '',
                'type': user.type,
                # 'location': user.location if user.location is not None else '',
                'bio': user.bio if user.bio is not None else '',
                'followers': user.followers,
                'following': user.following,
                'blog': user.blog if user.blog is not None else '',
            }
            users.append(user_dict)
        except Exception as e:
            print(f"Failed to retrieve information for user '{actor_id}': {str(e)}")
            err_user.append(actor_id)

    return users

def analyze_user_by_id(id):
    # user_info = get_user_info_by_id(id, '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')
    # for user in user_info:
    #     print(user)
    user = {}  # Initialize an empty dictionary
    user['id'] = id
    user = clickhouse_quey(user)
    bot_str = ['bot ', '-ci', '-io', '-cla', '-bot', '-test', 'bot@']
    user['login'] = 1 if any(s in user['login'].lower() for s in bot_str) else 0
    user['name'] = 1 if any(s in user['name'].lower() for s in bot_str) else 0
    user['bio'] = 1 if any(s in user['bio'].lower() for s in bot_str) else 0
    user['email'] = 1 if any(s in user['email'].lower() for s in bot_str) else 0

    return user


def clickhouse_quey(user):
    clickhouse = ClickHouseUtils(
        host='cc-uf6764sn662413tc9.public.clickhouse.ads.aliyuncs.com',
        user='xlab',
        password='Xlab2021!',
        database='opensource'
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
    user['response_time'] = result[0][0]
    print(user)
    return user

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('./data/bothawk_data.csv')
df = df[:10000]
# # 初始化一个空列表来存储每个用户分析的结果
# results = []
#
# # 遍历DataFrame中的每一行
# for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Analyzing Users"):
#     user_id = row['actor_id']
#     user_label = row['label']  # 获取用户的label
#     user_result = analyze_user_by_id(user_id)
#     user_result['label'] = user_label  # 将label添加到分析结果中
#     results.append(user_result)
#
# # 将结果列表转换成DataFrame
# results_df = pd.DataFrame(results)
#
# # 可选：将结果保存到新的CSV文件
# results_df.to_csv('./data/analyzed_users.csv', index=False)

def process_row(row):
    user_id = row['actor_id']
    user_label = row['label']  # 获取用户的label
    user_result = analyze_user_by_id(user_id)
    user_result['label'] = user_label  # 将label添加到分析结果中
    return user_result

# 使用ThreadPoolExecutor并行处理
with ThreadPoolExecutor(max_workers=5) as executor:  # 可以根据你的系统调整max_workers的数量
    # 准备工作：为每行创建一个future
    futures = [executor.submit(process_row, row) for index, row in df.iterrows()]

    results = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing Users"):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"Error processing row: {e}")

# 将结果列表转换成DataFrame
results_df = pd.DataFrame(results)

directory = "./data/"

# 如果目录不存在，则创建它
if not os.path.exists(directory):
    os.makedirs(directory)

# 可选：将结果保存到新的CSV文件
results_df.to_csv('./data/user_feature.csv', index=False)
