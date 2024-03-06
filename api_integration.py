#import AiforMobile as Ai
from flask import Flask,request,jsonify

app = Flask(__name__)

@app.route('/')
def function1():
    return 'hello world'



if __name__ == "__main__":
    app.run(debug=True)