from fastapi import FastAPI


app = FastAPI()

@app.get('/')
def hello_world():
    return 'Hello World'


@app.get('/hello')
def hello_cow():
    return 'Hello betch'

