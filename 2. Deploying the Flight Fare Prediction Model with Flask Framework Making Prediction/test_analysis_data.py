import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import app

with app.app.app_context():
    client = app.app.test_client()
    res = client.get('/analysis/data')
    print('status', res.status_code)
    data = res.get_json()
    for k in ['airline_prices','monthly_prices','source_dest_prices','stops_prices','price_distribution']:
        v = data.get(k)
        if isinstance(v, dict):
            print(k, {kk: len(v.get(kk, [])) for kk in v.keys()})
        elif isinstance(v, list):
            print(k, len(v))
        else:
            print(k, type(v))
