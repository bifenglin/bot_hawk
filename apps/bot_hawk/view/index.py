import os
from flask import Blueprint, current_app, render_template, request, jsonify
from bot_hawk.apps.utils import get_abs_dir
from bot_hawk.apps.utils.constants import METHODTYPE

index = Blueprint('index', __name__, url_prefix='/',
                  template_folder=os.path.join(get_abs_dir(__file__), '../templates'))


@index.route('/', methods=[METHODTYPE.GET, METHODTYPE.POST])
def index_home():
    current_app.logger.info(f'{request.method} for index.home')
    if request.method == METHODTYPE.GET:
        name = request.args.get('name', 'Python')
        return render_template('home.html', name=name)  # return html Content
    else:
        name = request.form.get('name', 'Python')   # for request that POST with application/x-www-form-urlencoded
        return f"Hello {name}", 200         # return plain text Content

@index.route('/check', methods=['POST'])
def check_account():
    account = request.form.get('account')
    # 这里是一个示例，你需要用实际的逻辑来判断账号是否为机器人
    # 例如，调用 GitHub API 获取用户信息
    # 注意：GitHub API 需要认证，你需要注册一个 OAuth token
    # 这里只是一个演示，没有实际调用 API
    return jsonify({"account": account, "is_bot": "unknown"})
