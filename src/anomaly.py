import numpy as np
from sklearn.ensemble import IsolationForest

def find_anomalies(list):
  v = np.reshape(list, (-1, 1))
  clf = IsolationForest(random_state=0, contamination=0.1).fit(v)
  return clf.predict(v)

def process_anomalies(data):
  dates = data['dates']['data']
  date_label = data['dates']['label']
  values = data['values']['data']
  value_label = data['values']['label']
  prediction = find_anomalies(values)

  anomalies = {
    'dates': {
      'label': date_label,
      'data': []
    },
    'values': {
      'label': date_label,
      'data': []
    },
    'count': 0,
  }

  for idx, predict in enumerate(prediction.tolist()):
    if predict == -1:
      anomalies['dates']['data'].append(dates[idx])
      anomalies['values']['data'].append(values[idx])
      anomalies['count'] += 1

  summary_body = ''
  if anomalies['count'] > 0:
    summary_body = f"The most recent anomaly was on **{anomalies['dates']['data'][-1]}** when **{value_label}** had value of **{anomalies['values']['data'][-1]}**"

  return {
    'anomalies': anomalies,
    'count': anomalies['count'],
    'summary': {
      'title': '' if anomalies['count'] == 0 else f'Recent anomalies in **{value_label}**',
      'body': summary_body
    }
  }
