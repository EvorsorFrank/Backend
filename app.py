from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/hello')
def hello_world():
    return 'Hello betch'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
