import numpy as np

def moving_average(data, window_size):
  cumsum = np.cumsum(data, dtype=float)
  cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
  return cumsum[window_size - 1:] / window_size

def detect_anomalies(data, window_size, threshold=2.0):
  moving_avg = moving_average(data, window_size)
  deviation = np.abs(data[window_size - 1:] - moving_avg)
  std_dev = np.std(deviation)
  lower_bound = moving_avg - threshold * std_dev
  upper_bound = moving_avg + threshold * std_dev
  anomalies = ((data[window_size - 1:] < lower_bound) | (data[window_size - 1:] > upper_bound)).astype(int)

  # Prepend empty values
  anomalies = np.concatenate([np.zeros(window_size - 1), anomalies])
  moving_avg = np.concatenate([np.zeros(window_size - 1), moving_avg])
  lower_bound = np.concatenate([np.zeros(window_size - 1), lower_bound])
  upper_bound = np.concatenate([np.zeros(window_size - 1), upper_bound])

  return anomalies, moving_avg, lower_bound, upper_bound

def process_anomalies(data):
  dates = data['dates']['data']
  date_label = data['dates']['label']
  values = data['values']['data']
  value_label = data['values']['label']

  prediction, moving_avg, lower_bound, upper_bound = detect_anomalies(values, window_size=6, threshold=3.269)

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

  anomalies_indexes = []

  for idx, predict in enumerate(prediction):
    if predict == 1:
      anomalies['dates']['data'].append(dates[idx])
      anomalies['values']['data'].append(values[idx])
      anomalies['count'] += 1
      anomalies_indexes.append(idx)

  summary = []
  anomal_summary_body = ''
  anomal_summary_title = ''
  if anomalies['count'] > 0:
    date = anomalies['dates']['data'][-1]
    y = anomalies['values']['data'][-1]
    l = np.round(lower_bound[anomalies_indexes[-1]], 2)
    h = np.round(upper_bound[anomalies_indexes[-1]], 2)

    anomal_summary_title = f'Recent anomalies in **{value_label}**'
    anomal_summary_body += f"The most recent anomaly was on **{date}** when **{value_label}** had value of **{y}**. "
    direction = 'increase' if y > h else 'decrease'
    anomal_summary_body += f"Which is out of the expected range of {l} - {h}, "
    anomal_summary_body += f"indicating a significant {direction} compared to the expected or historical patterns."

    summary.append(anomal_summary_title)
    summary.append(anomal_summary_body)


  cols = [date_label, 'Moving Average', 'Lower Bound', 'Upper Bound', 'Is Anomaly']
  values = list(zip(dates, moving_avg, lower_bound, upper_bound, prediction))
  return {
    'anomal_value': {
      'labels': cols,
      'values': values,
    },
    'anomal_metadata': {
      'direction': ''
    },
    'anomal_summary': summary,
    # 'moving_average': {
    #   'label': 'Moving Average',
    #   'data': moving_avg.tolist(),
    # },
    # 'lower_bound': {
    #   'label': 'Lower Bound',
    #   'data': lower_bound.tolist()
    # },
    # 'upper_bound': {
    #   'label': 'Upper Bound',
    #   'data': upper_bound.tolist()
    # },
    # 'anomalies': anomalies,
    # 'summary': summary
  }
