from sklearn.svm import SVR
import yfinance as yf


tkr = yf.Ticker("PFE")
df = yf.download("PFE",'2mo')

print(df)