from flask import Flask, request
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/hello')
def hello_cow():
    return 'Hello betch'

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['image']
    print(file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
