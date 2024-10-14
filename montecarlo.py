import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + ".AX" for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

print(meanReturns)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# monte carlo method
# number of simulations
mc_sims = 100
T = 100 #timeframe in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(0, mc_sims):
    # mc loops
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T ) + 1) * initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('portfolio Value($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
# plt.show()

def mcVaR(returns, alpha=5):
    """
    Input: pandas series of returns
    Output: percentile on return distribution to a given confidence level alpha
    """

    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series")
    

def mcCVaR(returns, alpha=5):
    """
    Input: pandas series of returns
    Output: Conditional value at risk or expected shortfall to a given confidence level alpha
    """

    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha) 
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series")
    

portResults = pd.Series(portfolio_sims[-1,:])

VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

print(f'Value at Risk ${round(VaR, 2)}')
print(f'Conditional Value at Risk ${round(CVaR, 2)}')
