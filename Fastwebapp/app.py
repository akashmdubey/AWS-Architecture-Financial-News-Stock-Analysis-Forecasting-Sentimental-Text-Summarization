from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/dataeda')
def dataeda():
    return render_template('dataeda.html')


@app.route('/architecture')
def show_architecture():
    return render_template('architecture.html')


@app.route('/sentimentanalysis')
def sentimentanalysis():
    return render_template('sentimentanalysis.html')

@app.route('/timeseriesanalysis')
def timeseriesanalysis():
    return render_template('timeseriesanalysis.html')

@app.route('/electronics')
def electronics():
    return render_template('electronics.html')

@app.route('/metric')
def metric():
    return render_template('metric.html')

@app.route('/cloud')
def cloud():
    return render_template('cloud.html')

@app.route('/retail')
def retail():
    return render_template('retail.html')

@app.route('/health')
def health():
    return render_template('health.html')

@app.route('/automobile')
def automobile():
    return render_template('automobile.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, use_reloader=True)
