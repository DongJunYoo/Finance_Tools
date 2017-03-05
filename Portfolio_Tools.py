'''
Note!
If you don't have pandas_datareader, you have to install by
UNIX: sudo apt-get install python-pandas-datareader
Command: conda install pandas-datareader

'''
import pandas as pd 
import numpy as np 
import pandas_datareader.data as web 
import matplotlib.pyplot as plt 
import scipy.stats as scs 
import statsmodels.api as sm

#----------public interface--------------#
def Plotting(stocks, start, end):
  
    '''
    Plotting normalized Time Series of Portfolio stocks
    Stock quotes must be in a list and the stock quote must be in google 
    stock quote format. You need to input your start and end date
    e.g.
    ---------------Stock Quote Format-------------------
    Asiana = 'KRX:020560', Apple= 'NASDAQ:AAPL'
    
    ---------------Stock input for this function---------------
    Asiana = 'KRX:020560'
    Korean_Air = 'KRX:003490'
    S_Oil = 'KRX:010950'
    SK_Innovation = 'KRX:096770'
    Samsung_Elec = 'KRX:005930'
    LG_Elec = 'KRX:066570'
    Naver_Corp = 'KRX:035420'
    
    Stock_Quotes = [Asiana, Korean_Air,
                S_Oil, SK_Innovation,
                Samsung_Elec, LG_Elec,
                Naver_Corp]
    
    **you can just input stock quote into the list in string format 
    --------------Start & End input format---------------
    start = "1/1/2010", end = "1/1/2017"
    '''
    Stock_Quotes = stocks
    data = pd.DataFrame()
    Begin = start 
    End = end
    
    
    #Crawling
    for quote in Stock_Quotes:
        data[quote] = web.DataReader(quote, data_source= "google", 
                                     start = Begin, 
                                     end = End)['Close']
    data.columns = Stock_Quotes
    data = data.dropna()
    
    #Plotting Normalized Time Series Graph
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Time Series')
    ax1.plot(data / data.ix[0] *100) #normalized
    ax1.legend(Stock_Quotes, loc = 2)
    plt.grid()
    plt.show()

def Return_Covariance(stocks, start, end):
    '''
    Computing and Printing Yearly Log Return for input stocks and Covariance
    Stock quotes must be in a list and the stock quote must be in google 
    stock quote format. You need to input your start and end date
    e.g.
    ---------------Stock Quote Format-------------------
    Asiana = 'KRX:020560', Apple= 'NASDAQ:AAPL'
    
    ---------------Stock input for this function---------------
    Asiana = 'KRX:020560'
    Korean_Air = 'KRX:003490'
    S_Oil = 'KRX:010950'
    SK_Innovation = 'KRX:096770'
    Samsung_Elec = 'KRX:005930'
    LG_Elec = 'KRX:066570'
    Naver_Corp = 'KRX:035420'
    
    Stock_Quotes = [Asiana, Korean_Air,
                S_Oil, SK_Innovation,
                Samsung_Elec, LG_Elec,
                Naver_Corp]
    
    **you can just input stock quote into the list in string format 
    --------------Start & End input format---------------
    start = "1/1/2010", end = "1/1/2017"
    '''
    Stock_Quotes = stocks
    data = pd.DataFrame()
    Begin = start 
    End = end
    for quote in Stock_Quotes:
        data[quote] = web.DataReader(quote, data_source= "google", 
                                     start = Begin, 
                                     end = End)['Close']
    data.columns = Stock_Quotes
    data = data.dropna()
    Stock_Returns = np.log(data / data.shift(1))
    print("---Stock Yearly Return (252 trading days)---")
    print(Stock_Returns.mean() * 252)
    print()
    print("---Covariance Matrix of input Stocks---")
    print(Stock_Returns.cov() * 252)

def Plotting_Potential_Portfolios(stocks, start, end, rf):
    '''
    Plotting Investment Opportunity Set of Attainable Portfolios
    with random Portfolio Weights (50,000 Monte Carlo Simulations) 
    
    Note!:
    1. Stocks / Portfolio of Stocks in list [] format must be inserted 
    Stock quotes must be in google stock quote format
    2. Start & End Date must be inserted
       e.g.  start = "1/1/2010", end = "1/1/2017"
    3. Risk Free Rate must be inserted for Sharpe Ratio
    e.g. 5% --> 0.05 for input
    '''
    Stock_Quotes = stocks
    data = pd.DataFrame()
    Begin = start 
    End = end
    risk_free = rf
    for quote in Stock_Quotes:
        data[quote] = web.DataReader(quote, data_source= "google", 
                                     start = Begin, 
                                     end = End)['Close']
    data.columns = Stock_Quotes
    data = data.dropna()
    Stock_Returns = np.log(data / data.shift(1))
    no_assets = len(Stock_Quotes)
    
    p_returns = []
    p_vols = []
    for p in range(50000):
        weights = np.random.random(no_assets)
        weights /= np.sum(weights)
        p_returns.append(np.sum(Stock_Returns.mean()*weights)*252)
        p_vols.append(np.sqrt(np.dot(weights.T, 
                                     np.dot(Stock_Returns.cov()*252, weights))))
    p_returns = np.array(p_returns)
    p_vols = np.array(p_vols)
    plt.scatter(p_vols, p_returns, c = (p_returns - risk_free) 
                    / p_vols, marker = 'o')
    plt.grid(True)
    plt.title('Investment Opportunity Set of Attainable Portfolios')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label = 'Sharpe Ratio')
    plt.show()

def Optimize_Portfolio(stocks, start, end, rf):
    '''
     Note!:
    1. Stocks / Portfolio of Stocks in list [] format must be inserted 
    Stock quotes must be in google stock quote format
    e.g. 
    ---------------Stock Quote Format-------------------
    Asiana = 'KRX:020560', Apple= 'NASDAQ:AAPL'
    
    ---------------Stock input for this function---------------
    Asiana = 'KRX:020560'
    Korean_Air = 'KRX:003490'
    S_Oil = 'KRX:010950'
    SK_Innovation = 'KRX:096770'
    Samsung_Elec = 'KRX:005930'
    LG_Elec = 'KRX:066570'
    Naver_Corp = 'KRX:035420'
    
    Stock_Quotes = [Asiana, Korean_Air,
                S_Oil, SK_Innovation,
                Samsung_Elec, LG_Elec,
                Naver_Corp]
    
    **you can just input stock quote into the list in string format 
    ---------------------------------------------------------------
    
    2. Start & End Date must be inserted
       e.g.  start = "1/1/2010", end = "1/1/2017"
    3. Risk Free Rate must be inserted for Sharpe Ratio
    e.g. 5% --> 0.05 for input
    4. 10,000 Monte Carlo Simulations 
    Default 
    1. start date: "1/1/2010"
    2. end date: "1/1/2017"
    3. risk-free rate: 0%
    '''
    import scipy.optimize as sco
    Stock_Quotes = stocks
    data = pd.DataFrame()
    Begin = start 
    End = end
    for quote in Stock_Quotes:
        data[quote] = web.DataReader(quote, data_source= "google", 
                                     start = Begin, 
                                     end = End)['Close']
    data.columns = Stock_Quotes
    data = data.dropna()
    Stock_Returns = np.log(data / data.shift(1))
    no_assets = len(Stock_Quotes)
    p_returns = []
    p_vols = []
    for p in range(50000):
        weights = np.random.random(no_assets)
        weights /= np.sum(weights)
        p_returns.append(np.sum(Stock_Returns.mean()*weights)*252)
        p_vols.append(np.sqrt(np.dot(weights.T, 
                                     np.dot(Stock_Returns.cov()*252, weights))))
    p_returns = np.array(p_returns)
    p_vols = np.array(p_vols)
    def statistics(weights):
        weights = np.array(weights)
        p_returns = np.sum(Stock_Returns.mean()*weights)*252
        p_vols = np.sqrt(np.dot(weights.T, 
                                np.dot(Stock_Returns.cov()*252, weights)))
        return np.array([p_returns, p_vols, p_returns/p_vols])
    def min_func_sharpe(weights):
        return -statistics(weights)[2]
    
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnd = tuple((0,1) for x in range(no_assets))
    
    opts = sco.minimize(min_func_sharpe, no_assets*[1./no_assets],
                        method ='SLSQP', bounds = bnd, constraints = cons)
    print('#-----Asset Allocations for Max Sharpe Ratio -----#')
    print(opts['x'].round(3))
    print('---return, volatility (SD), Max sharpe ratio---')
    print(statistics(opts['x'].round(3)))
    print('*note: multiply 100 for return, volatility to get value in %')
    print()
    def min_func_variance(weights):
        return statistics(weights)[1] **2
    optv = sco.minimize(min_func_variance, no_assets*[1. / no_assets],
                    method = 'SLSQP', bounds = bnd, constraints = cons)
    print('#-----Asset Allocations for Minimum SD / Volatility-----#')
    print(optv['x'].round(3))
    print('---return, volatility (SD), sharpe ratio---')
    print(statistics(optv['x']).round(3))
    print('*note: multiply 100 for return, volatility to get value in %')


def Normality_Test(stocks, start, end):
    '''
    Conducts Normality Test of each asset in a portfolio
    1) computes size, min, max, mean, std, skew, kurtosis based on distribution
       of stock log returns 
    2) computes p-value to check normality of each asset. 
    *if p-value of Normality test is lower than 0.05, we reject the Null Hypothesis
    and say that the stock return does not follow normal distribution 
    *if p-value of Normality test is bigger than o.05, we accept the Null Hypothesis
    and say that the stock return follows like geometric brown motion, adn thus 
    follow normal distribution  
    
     Note!:
    1. Stocks / Portfolio of Stocks in list [] format must be inserted 
    Stock quotes must be in google stock quote format
    e.g. 
    ---------------Stock Quote Format-------------------
    Asiana = 'KRX:020560', Apple= 'NASDAQ:AAPL'
    
    ---------------Stock input for this function---------------
    Asiana = 'KRX:020560'
    Korean_Air = 'KRX:003490'
    S_Oil = 'KRX:010950'
    SK_Innovation = 'KRX:096770'
    Samsung_Elec = 'KRX:005930'
    LG_Elec = 'KRX:066570'
    Naver_Corp = 'KRX:035420'
    
    Stock_Quotes = [Asiana, Korean_Air,
                S_Oil, SK_Innovation,
                Samsung_Elec, LG_Elec,
                Naver_Corp]
    
    **you can just input stock quote into the list in string format 
    ---------------------------------------------------------------
    
    2. Start & End Date must be inserted
       e.g.  start = "1/1/2010", end = "1/1/2017"
    
    '''
    
    
    Stock_Quotes = stocks
    data = pd.DataFrame()
    Begin = start 
    End = end
    for quote in Stock_Quotes:
        data[quote] = web.DataReader(quote, data_source= "google", 
                                     start = Begin, 
                                     end = End)['Close']
    data.columns = Stock_Quotes
    data = data.dropna()
    
    def Stats_Print(array):
        sta = scs.describe(array)
        print("%14s %15s" %('statistic', 'value'))
        print('-'*40)
        print("%14s %15.5f" % ('size', sta[0]))
        print("%14s %15.5f" % ('min', sta[1][0]))
        print("%14s %15.5f" % ('max', sta[1][1]))
        print("%14s %15.5f" % ('mean', sta[2]))
        print("%14s %15.5f" % ('std', np.sqrt(sta[3])))
        print("%14s %15.5f" % ('skew', sta[1][1]))
        print("%14s %15.5f" % ('kurtosis', sta[5]))
    def Normality_Test_Result(arr):
        print("skew of data set %14.3f" %scs.skew(arr))
        print("skew test p-value %14.3f" %scs.skewtest(arr)[1])
        print("Kurt of data set %14.3f" %scs.kurtosistest(arr)[1])
        print("Kurt test p-value %14.3f" %scs.kurtosistest(arr)[1])
        print("Norm test p-value %14.3f" %scs.normaltest(arr)[1])
    
    log_returns = np.log(data / data.shift(1)) 
    for q in Stock_Quotes:
        print('\nResult for Stock %s' %q)
        print('-'*40)
        log_data = np.array(log_returns[q].dropna())
        Stats_Print(log_data)
    
    
    print()
    print('###############Normality Test Results################')
    for q in Stock_Quotes:
        print('\nResult for Stock %s' %q)
        print('-'*40)
        log_data = np.array(log_returns[q].dropna())
        Normality_Test_Result(log_data)
        
    log_returns.hist(bins=50, figsize = (9, 6)) #plotting histogram
    plt.show()


def Q_Q_PLOT(stocks, start, end, chosen_stock_quote):
    '''
    For Normality Test, plots Quantile - Quantile plot
    Note!:
    1. Stocks / Portfolio of Stocks in list [] format must be inserted 
    Stock quotes must be in google stock quote format
    e.g. 
    ---------------Stock Quote Format-------------------
    Asiana = 'KRX:020560', Apple= 'NASDAQ:AAPL'
    
    ---------------Stock input for this function---------------
    Asiana = 'KRX:020560'
    Korean_Air = 'KRX:003490'
    S_Oil = 'KRX:010950'
    SK_Innovation = 'KRX:096770'
    Samsung_Elec = 'KRX:005930'
    LG_Elec = 'KRX:066570'
    Naver_Corp = 'KRX:035420'
    
    Stock_Quotes = [Asiana, Korean_Air,
                S_Oil, SK_Innovation,
                Samsung_Elec, LG_Elec,
                Naver_Corp]
    
    **you can just input stock quote into the list in string format 
    ---------------------------------------------------------------
    
    2. Start & End Date must be inserted
       e.g.  start = "1/1/2010", end = "1/1/2017"
    3. The stock for Q-Q plot should be entered as stock quote in 
    "chosen_stock_quote parameter. It should be entered as string. 
    e.g.  'KRX:003490'
    '''
     
    Stock_Quotes = stocks
    data = pd.DataFrame()
    Begin = start 
    End = end
    for quote in Stock_Quotes:
        data[quote] = web.DataReader(quote, data_source= "google", 
                                     start = Begin, 
                                     end = End)['Close']
    data.columns = Stock_Quotes
    data = data.dropna()
    log_returns = np.log(data / data.shift(1)) 
    sm.qqplot(log_returns[chosen_stock_quote].dropna(), line = 's')
    plt.grid(True)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()
    print('Straight Line = Sign of Normal Distribution')
    print('Curved / not straight = sign of fat tail, Outlier, not normal')
    



