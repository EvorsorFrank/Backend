from flask import Flask, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
CORS(app)
#app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://capstone_samonte_db_user:j3vFR3PT7BepaBcEl789ocCL5UzzsYfi@dpg-cnqqb5v109ks73fd77pg-a.singapore-postgres.render.com/capstone_samonte_db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin123@localhost:5432/test-capstone'

db = SQLAlchemy(app)


class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    done = db.Column(db.Boolean, default=False)


with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return "hello"


@app.route('/tasks')
def get_tasks():
    tasks = Task.query.all()
    task_list = [
        {'id': task.id, 'title': task.title, 'done': task.done} for task in tasks
    ]
    return jsonify({"tasks": task_list})


if __name__ == '__main__':
    app.run()
