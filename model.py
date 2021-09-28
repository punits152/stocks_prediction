import numpy as np
import yfinance as yf
from sklearn.svm import SVR
from  matplotlib.dates import date2num
df = yf.download("AMZN",period="12mo")
df.reset_index(inplace=True)
df["Date"] = df['Date'].map(date2num)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split(df["Date"], df["Close"], test_size=0.1, random_state=123)
svr = SVR()
svr.fit(np.array(X_train).reshape(-1,1),Y_train)


import pickle 
with open('model','wb') as f:
    pickle.dump(svr,f)