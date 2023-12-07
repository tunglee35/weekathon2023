from flask import Flask, request, jsonify
from anomaly import process_anomalies

app = Flask(__name__)

@app.route("/")
def hello_world():
  return "<p>Hello, World!</p>"

@app.route('/summary', methods = ['POST'])
def summary():
  data = request.get_json()
  anomalies = process_anomalies(data)

  return anomalies

if __name__ == '__main__':
  app.run(port=3001, debug=True)