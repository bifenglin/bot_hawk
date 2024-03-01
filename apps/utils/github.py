from github import Github

def get_user_info_by_id(actor_ids, api_token):
    print(api_token)
    g = Github(api_token)
    users = []
    err_user = []

    if not isinstance(actor_ids, list):
        actor_ids = [actor_ids]

    for actor_id in actor_ids:
        # try:
        user = g.get_user_by_id(int(actor_id))
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
            # 'blog': user.blog if user.blog is not None else '',
        }
        users.append(user_dict)
        # except Exception as e:
        #     print(f"Failed to retrieve information for user '{actor_id}': {str(e)}")
        #     err_user.append(actor_id)

    return users

def get_user_info_by_name(user_names, api_token):
    g = Github(api_token)
    users = []
    err_user = []

    if not isinstance(user_names, list):
        user_names = [user_names]

    for user_name in user_names:
        try:
            user = g.get_user(user_name)
            # print(dir(user))
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
            print(f"Failed to retrieve information for user '{user_name}': {str(e)}")
            err_user.append(user_name)

    return users

import requests
import json

def run_query(query, headers):
    request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception(f"Query failed to run by returning code of {request.status_code}. {query}")

def get_user_info_hql_by_names(user_names, api_token):
    # 构建GraphQL查询
    user_names_query = '", "'.join(user_names)
    query = f'''
    {{
      search(query: "type:user {user_names_query}", type: USER, first: {len(user_names)}) {{
        nodes {{
          ... on User {{
            id
            login
            name
            email
            bio
            followers {{
              totalCount
            }}
            following {{
              totalCount
            }}
          }}
        }}
      }}
    }}
    '''

    headers = {"Authorization": f"Bearer {api_token}"}
    result = run_query(query, headers)
    users = []
    err_user = []

    # 解析结果
    try:
        for user in result['data']['search']['nodes']:
            users.append({
                'id': user['id'],
                'login': user['login'],
                'name': user['name'] if user['name'] is not None else '',
                'email': user['email'] if user['email'] is not None else '',
                'bio': user['bio'] if user['bio'] is not None else '',
                'followers': user['followers']['totalCount'],
                'following': user['following']['totalCount'],
            })
    except KeyError as e:
        print(f"Failed to retrieve information: {str(e)}")
        err_user = user_names  # 假设所有请求都失败

    return users, err_user
