import requests
import csv 
import json 
import pandas as pd


url = "http://api.coincap.io/v2/assets/bitcoin/history?interval=d1&start=1592585794000&end=1613753794000"

payload = {}
headers= {}

response = requests.request("GET", url, headers=headers, data = payload)

json_data = json.loads(response.text.encode('utf8'))

print(json_data)

bitcoin_data = json_data["data"]

import pandas as pd

df = pd.DataFrame(bitcoin_data)
df.to_csv('bitcoin-usd.csv', index=False)

print(df.sample)

import matplotlib.pyplot as plt
df.plot(x ='time', y='priceUsd', kind = 'line')
plt.show()

df.dtypes

df = pd.DataFrame(bitcoin_data, columns=['time', 'priceUsd'])
print(df.sample)

df['priceUsd'] = pd.to_numeric(df['priceUsd'], errors='coerce').fillna(0, downcast='infer')
df.dtypes
df.info()
df.plot(x ='time', y='priceUsd', kind = 'line')
plt.show()