import urllib.request, json, time, subprocess, psutil

proc = subprocess.Popen(['python', r'D:\N4_K1\HTTM\Code\TieuLuan\RNN_CoPhieu\webapp\app.py'])
time.sleep(5)

try:
    p = psutil.Process(proc.pid)
    mem = p.memory_info().rss / 1e6
    print(f'Server RAM usage: {mem:.1f} MB')

    # Test predict_latest
    req = urllib.request.urlopen('http://127.0.0.1:5000/api/predict_latest?model=DeepStackedLSTM')
    data = json.loads(req.read().decode('utf-8'))
    print('Predict Latest:', data['model_used'], '->', data['prediction'], f"Prob Up: {data['probability_up']}%", f"Confidence: {data['confidence']}%")

    # Test predict_compare_all
    req2 = urllib.request.urlopen('http://127.0.0.1:5000/api/predict_compare_all')
    data2 = json.loads(req2.read().decode('utf-8'))
    print('Compare All:', [(x['model'], x['pred'], x['prob_up']) for x in data2['comparison']])

    print('ALL ENDPOINTS WORKING PERFECTLY!')
finally:
    proc.terminate()
