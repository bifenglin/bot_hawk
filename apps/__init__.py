import os
from flask import Flask
from .bot_hawk import init_blue_print
from .utils import config_log
from bot_hawk.config import config
from flask_sqlalchemy import SQLAlchemy
from apps.database import db

def create_app():
    config_log()
    app = Flask(__name__)
    env = os.environ.get('FLASK_ENV', 'default')
    app.config.from_object(config.get(env))
    db.init_app(app)
    init_blue_print(app)

    return app

