from flask import Flask, render_template

app = Flask(__name__)
#app.static_folder ='static'

@app.route('/')
def home(): 
    return render_template('Start.html')


if __name__ == '__main__':
    app.run()