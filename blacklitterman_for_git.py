import yaml
from modules import date_lib_for_git
import pandas as pd
import numpy as np
import openpyxl


def index_setting_monthly(dataset: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Date Preprocessing.
    :param dataset: dataset will be adjusted
    :param path: the standard date dataset path
    :return: an adjusted dataset
    """
    date_dataset = pd.read_excel(path, sheet_name='price', usecols=[0])
    date_dataset.columns = ['Date']
    index_list = []
    for i in dataset.index:
        for j in date_dataset['Date']:
            if i.year == j.year and i.month == j.month:
                index_list.append(j)
                break
            else:
                pass
    if len(dataset.index) == len(index_list):
        dataset.index = index_list
        return dataset
    else:
        raise ValueError


class BlackLitterman:
    def __init__(self, config, period_start, period_end, kind='end', weight_accumulated=False, currency_open=False):
        """
        :param period_start: The start date for covariance matrix and risk aversion coefficient
        :param period_end: The end date for covariance matrix and risk aversion coefficient
        :param kind: What trading date you want in months
        :param weight_accumulated: Whether suppose that asset weight is affected by the growth of each asset
        :param weight_currency: Whether suppose that current is open or close
        """

        self._path = config['path']
        self.assets = config['assets']
        self.assets_currency = config['assets_currency']
        self._asset_columns = config['asset_columns']

        date_dataset = pd.read_excel(self._path, sheet_name='price', usecols=[0])
        date_dataset.columns = ['Date']

        self._start = date_lib_for_git.what_date_of_months(date_dataset['Date'], period_start, kind=kind)
        self._end = date_lib_for_git.what_date_of_months(date_dataset['Date'], period_end, kind=kind)
        self._divisor = 12  # meaning 'monthly'

        # Equity Block
        self.dy_us, self.dividend_growth_us, self.expected_inflation_us = None, None, None
        self.dy_kor, self.dividend_growth_kor, self.expected_inflation_kor = None, None, None
        # Fixed Income Block
        self.ytm_us, self.roll_down_us, self.credit_spread_us = None, None, None
        self.ytm_kor, self.roll_down_kor, self.credit_spread_kor = None, None, None
        # Risk Free Rate(CD)
        cd_dataset = pd.read_excel(self._path, sheet_name="CD91", usecols=[0, 1])
        cd_dataset.columns = ['Date', 'price']
        cd_dataset.set_index("Date", inplace=True)
        cd_dataset = index_setting_monthly(cd_dataset, self._path)
        self.rf = np.log(cd_dataset.loc[self._end, 'price'] + 1)

        self.weight = {}
        dataset = pd.read_excel(self._path, sheet_name="weight", usecols=[0] + self._asset_columns)
        dataset.columns = ['Date'] + self.assets
        dataset.set_index('Date', inplace=True)
        dataset = index_setting_monthly(dataset, self._path)
        self.currency_open = currency_open
        if weight_accumulated:  # Suppose that asset weight is affected by the growth of each asset
            price_dataset = pd.read_excel(self._path, sheet_name="price",
                                          usecols=[0] + self._asset_columns + [len(self.assets) + 1])
            price_dataset.columns = ['Date'] + self.assets + ['currency']
            price_dataset.set_index('Date', inplace=True)
            if self.currency_open:
                currency_constant = price_dataset.loc[self._end, 'currency'] / price_dataset.loc[self._start, 'currency']
            else:
                currency_constant = 1
            for i in self.assets:
                if i not in self.assets_currency:
                    self.weight[i] = dataset.loc[self._start, i] * price_dataset.loc[self._end, i] / \
                                     price_dataset.loc[self._start, i]
                else:
                    self.weight[i] = dataset.loc[self._start, i] * price_dataset.loc[
                        self._end, i] * currency_constant / price_dataset.loc[self._start, i]

        else:
            for i in self.assets:
                self.weight[i] = dataset.loc[self._end, i]
        for r in self.weight:
            self.weight[r] = self.weight[r] / sum(self.weight.values())

    def covariance_matrix(self, excess=True, currency_open=True):
        dataset = pd.read_excel(self._path, sheet_name="price", usecols=[0] + self._asset_columns + [len(self.assets) + 1])
        cd_dataset = pd.read_excel(self._path, sheet_name="CD91", usecols=[0, 1])
        dataset.columns = ['Date'] + self.assets + ['currency']
        dataset.set_index('Date', inplace=True)
        cd_dataset.columns = ['Date', 'price']
        cd_dataset.set_index('Date', inplace=True)
        cd_dataset = index_setting_monthly(cd_dataset, self._path)
        adjusted_dataset = []
        a = []

        for j in self.assets:
            tem = []
            for i in dataset[self._start:self._end].index:
                idx_i = np.searchsorted(dataset.index, i)
                yesterday = dataset.index[max(0, idx_i-1)]
                if excess:
                    risk_free_constant = (cd_dataset.loc[yesterday, 'price'] + 1)**(1/self._divisor) - 1
                else:
                    risk_free_constant = 0
                if currency_open:
                    currency_constant = dataset.loc[i, 'currency']/dataset.loc[yesterday, 'currency']
                else:
                    currency_constant = 1
                if j not in self.assets_currency:
                    tem.append(np.log(dataset.loc[i, j]/dataset.loc[yesterday, j] - risk_free_constant))
                else:
                    tem.append(np.log(dataset.loc[i, j]*currency_constant/dataset.loc[yesterday, j] - risk_free_constant))

            adjusted_dataset.append(tem)
        adjusted_dataset = pd.DataFrame(adjusted_dataset)

        for i in range(len(self.assets)):
            tem = []
            for j in range(len(self.assets)):
                tem.append(np.cov(adjusted_dataset.loc[i], adjusted_dataset.loc[j])[0, 1])
            a.append(tem)
        a = pd.DataFrame(a)
        # annual return
        a = a*self._divisor
        return a

    def risk_aversion_coefficient(self, amount):  # amount is about the risk aversion
        dataset = pd.read_excel(self._path, sheet_name="price", usecols=[0] + self._asset_columns + [len(self.assets) + 1])
        dataset.columns = ['Date'] + self.assets + ['currency']
        dataset.set_index("Date", inplace=True)
        cd_dataset = pd.read_excel(self._path, sheet_name="CD91", usecols=[0, 1])
        cd_dataset.columns = ['Date', 'price']
        cd_dataset.set_index("Date", inplace=True)
        tem = []
        rf = 0
        for i in cd_dataset.loc[self._start:self._end, 'price']:
            rf = rf + np.log(1 + ((i + 1)**(1/self._divisor) - 1))
        if self.currency_open:
            currency_constant = dataset.loc[self._end, 'currency']/dataset.iloc[self._start, 'currency']
        else:
            currency_constant = 1
        for j in self.assets:
            if j not in self.assets_currency:
                tem.append(np.log(dataset.loc[self._end, j]/dataset.loc[self._start, j]))
            else:
                tem.append(np.log(dataset.loc[self._end, j]*currency_constant/dataset.loc[self._start, j]))
        tem = pd.DataFrame(tem)

        a = (pd.DataFrame(list(self.weight.values())).transpose()).dot(tem)
        var = (pd.DataFrame(list(self.weight.values())).transpose()).dot(self.covariance_matrix(excess=True, currency_open=self.currency_open)).dot(pd.DataFrame(list(self.weight.values())))
        exponential = (self._end.year - self._start.year) + (1 / 12) * (self._end.month - self._start.month)
        excess_return = (a / exponential) - (rf / exponential)
        var = var*self._divisor
        return (excess_return.iloc[0, 0] + (var.iloc[0, 0] ** (1 / 2)) * amount) / var.iloc[0, 0]

    def implied_return(self):
        result_dict = {}
        for a in self.assets:
            result_dict[a] = 0
        return result_dict

    def expectation(self):  # expected return
        date_dataset = pd.read_excel(self._path, sheet_name='price', usecols=[0])
        date_dataset.columns = ['Date']
        dataset_us_stock = pd.read_excel(self._path, sheet_name="expectation_raw_stock", usecols=[i for i in range(4)])
        dataset_us_stock.drop([0], inplace=True)
        dataset_us_stock.columns = ['Date', 'DPS', 'Price', 'CPI']
        dataset_us_stock.set_index('Date', inplace=True)
        dataset_kor_stock = pd.read_excel(self._path, sheet_name="KOSPI200", usecols=[0, 1, 2, 3])
        dataset_kor_stock.columns = ['Date', 'Price', 'Dividend Yield', 'CPI']
        dataset_kor_stock.set_index('Date', inplace=True)
        dataset_kor_stock_lag = dataset_kor_stock.shift(4)
        dataset_kor_stock_lag['DPS'] = dataset_kor_stock_lag['Price']*dataset_kor_stock_lag['Dividend Yield']*0.01
        dataset_kor_stock['DPS'] = np.where(dataset_kor_stock_lag.index.month == 4, dataset_kor_stock_lag['DPS'], None)

        dataset_expected_inflation = pd.read_excel(self._path, sheet_name="expected inflation", usecols=[i for i in range(4)])
        dataset_expected_inflation.drop([0], inplace=True)
        dataset_expected_inflation.columns = ['Date', 'US', 'KOR', 'KOR_adjusted']  # KOR_adjusted는 발표 일자에 맞게 편향 생기지 않도록 조정
        dataset_expected_inflation.set_index('Date', inplace=True)
        dataset_expected_inflation = index_setting_monthly(dataset_expected_inflation, self._path)
        dataset_fi = pd.read_excel(self._path, sheet_name="expectation_raw_bond", usecols=[i for i in range(5)])
        dataset_fi.columns = ['Date', 'Domestic Treasury 5 years', 'Domestic Treasury 10 years', 'Global Treasury 5 years', 'Global Treasury 10 years']
        dataset_fi.set_index('Date', inplace=True)
        dataset_fi = index_setting_monthly(dataset_fi, self._path)
        dataset_credit_us = pd.read_excel(self._path, sheet_name="US IG SPREAD", usecols=[i for i in range(2)])
        dataset_credit_us.columns = ['Date', 'spread']
        dataset_credit_us.set_index('Date', inplace=True)
        dataset_credit_kor = pd.read_excel(self._path, sheet_name="KOBI Credit", usecols=[i for i in range(2)])
        dataset_credit_kor.columns = ['Date', 'spread']
        dataset_credit_kor.set_index('Date', inplace=True)

        ytm_dict = {
            'Domestic Treasury 5 years': dataset_fi.loc[self._end, 'Domestic Treasury 5 years'],
            'Domestic Treasury 10 years': dataset_fi.loc[self._end, 'Domestic Treasury 10 years'],
            'Global Treasury 5 years': dataset_fi.loc[self._end, 'Global Treasury 5 years'],
            'Global Treasury 10 years': dataset_fi.loc[self._end, 'Global Treasury 10 years'],
        }

        def forming(dataset):
            dataset = index_setting_monthly(dataset, self._path)
            five_years_ago = date_lib_for_git.the_number_date(self._end, -59)
            five_years_ago_timestamp = date_lib_for_git.what_date_of_months(date_dataset['Date'], five_years_ago)
            dataset['DPS Inflation Adjusted'] = dataset['DPS'] * dataset.loc[self._end, 'CPI'] / dataset['CPI']
            dataset['Price Inflation Adjusted'] = dataset['Price'] * dataset.loc[self._end, 'CPI'] / dataset['CPI']
            dataset['Price Inflation Adjusted'] = dataset['Price'] * dataset.loc[self._end, 'CPI'] / dataset['CPI']

            dividend_yield_inflation_adjusted = float(np.nanmean(dataset.loc[five_years_ago_timestamp:self._end, 'DPS Inflation Adjusted']) / dataset.loc[self._end, 'Price Inflation Adjusted'])  # DPS Inflation Adjusted은 5년 평균
            date_initial = dataset[(dataset['DPS Inflation Adjusted'].notnull())].index[0]
            date_last = dataset[(dataset['DPS Inflation Adjusted'].notnull())].index[-1]

            dividend_growth = (dataset.loc[date_last, 'DPS Inflation Adjusted']/dataset.loc[date_initial, 'DPS Inflation Adjusted']) ** (12/(len(dataset[date_initial:date_last]) - 1)) - 1
            return dividend_yield_inflation_adjusted, dividend_growth

        def roll_down(rate_10, rate_5):
            price_year10 = 100
            # Treasury 10 years which price and par value are same is discounted by 5 years yield
            price_year5 = 100 / ((1 + rate_5 / 100) ** 5) + 100 * ((rate_10 / 2) / 100) * (
                        1 - (1 / (1 + rate_5 / 100) ** 5)) / (
                                  (1 + rate_5 / 100) ** (1 / 2) - 1)
            return ((price_year5 / price_year10) ** (1 / 5) - 1) * 100

        expectation = {}
        # Equity: dividend yield + dividend growth + expected inflation
        self.dy_us = forming(dataset_us_stock)[0]
        self.dividend_growth_us = forming(dataset_us_stock)[1]
        self.dy_kor = forming(dataset_kor_stock)[0]
        self.dividend_growth_kor = forming(dataset_kor_stock)[1]
        self.expected_inflation_us = dataset_expected_inflation.loc[self._end, 'US']/100
        self.expected_inflation_kor = dataset_expected_inflation.loc[self._end, 'KOR_adjusted']/100
        expectation['Global Equity'] = np.log(1 + self.dy_us + self.dividend_growth_us + self.expected_inflation_us) - self.rf
        expectation['Domestic Equity'] = np.log(1 + self.dy_kor + self.dividend_growth_kor + self.expected_inflation_kor) - self.rf
        # Fixed Income: ytm + roll down spread + credit spread
        self.ytm_us = ytm_dict['Global Treasury 10 years']*0.01
        self.roll_down_us = roll_down(ytm_dict['Global Treasury 10 years'], ytm_dict['Global Treasury 5 years'])*0.01
        self.credit_spread_us = dataset_credit_us.loc[self._end, 'spread']*0.01
        self.ytm_kor = ytm_dict['Domestic Treasury 10 years']*0.01
        self.roll_down_kor = roll_down(ytm_dict['Domestic Treasury 10 years'], ytm_dict['Domestic Treasury 5 years'])*0.01
        self.credit_spread_kor = dataset_credit_kor.loc[self._end, 'spread']*0.01

        expectation['Global Corporate'] = np.log(1 + self.ytm_us + self.roll_down_us + self.credit_spread_us) - self.rf
        expectation['Global Treasury'] = np.log(1 + self.ytm_us + self.roll_down_us) - self.rf
        expectation['Domestic Corporate'] = np.log(1 + self.ytm_kor + self.roll_down_kor + self.credit_spread_kor) - self.rf
        expectation['Domestic Treasury'] = np.log(1 + self.ytm_kor + self.roll_down_kor) - self.rf

        result = {}
        for a in self.assets:
            result[a] = expectation[a]
        return result

    def correlation_matrix(self, excess=True):
        dataset = pd.read_excel(self._path, sheet_name="price", usecols=[0] + self._asset_columns + [len(self.assets) + 1])
        cd_dataset = pd.read_excel(self._path, sheet_name="CD91", usecols=[0, 1])

        dataset.columns = ['Date'] + self.assets + ['Currency']
        dataset.set_index("Date", inplace=True)
        cd_dataset.columns = ['Date', 'price']
        cd_dataset.set_index("Date", inplace=True)
        adjusted_dataset = []
        a = []

        t1 = 0
        t2 = 0
        number1 = 0
        number2 = 0
        for t in range(len(dataset)):
            if dataset.index[t] == self._start:
                t1 = t
                number1 = number1 + 1
            elif dataset.index[t] == self._end:
                t2 = t
                number2 = number2 + 1
            else:
                pass

        for j in range(len(self.assets)):
            tem = []
            for i in range(t1, t2+1):
                if excess:
                    risk_free_constant = (cd_dataset.iloc[i - 1, 0] + 1)**(1/self._divisor) - 1
                else:
                    risk_free_constant = 0
                tem.append(np.log(dataset.iloc[i, j]/dataset.iloc[i - 1, j] - risk_free_constant))
            adjusted_dataset.append(tem)
        adjusted_dataset = pd.DataFrame(adjusted_dataset)
        for i in range(len(self.assets)):
            tem = []
            for j in range(len(self.assets)):
                tem_mat = np.cov(adjusted_dataset.iloc[i], adjusted_dataset.iloc[j])
                tem.append(tem_mat[0, 1]/(tem_mat[0, 0]**(1/2))/(tem_mat[1, 1]**(1/2)))
            a.append(tem)
        a = pd.DataFrame(a)
        a.columns = self.assets
        a.index = self.assets
        return a


def projection_matrix(size):
    a = [[0 for i in range(size)] for j in range(size)]
    a_matrix = pd.DataFrame(a)
    for i in range(size):
        for j in range(size):
            if i == j:
                a_matrix.iloc[i, j] = 1
            else:
                pass
    return a_matrix


def tuning_constant():
    a = 0.05
    return a


def expected_return(config, period_start, period_end, kind='end', currency_open=False):
    result_dict = {}
    tem = BlackLitterman(config, period_start, period_end, kind=kind)
    p = projection_matrix(len(tem.assets))
    r = list(tem.implied_return().values())
    expectation = list(tem.expectation().values())

    if currency_open:
        u = (p.dot(tuning_constant()*tem.covariance_matrix(excess=True, currency_open=True)).dot(p.transpose()))*np.eye(len(tem.assets))  # uncertainty matrix of views
        s = tem.covariance_matrix(excess=True, currency_open=True)
    else:
        u = (p.dot(tuning_constant()*tem.covariance_matrix(excess=True, currency_open=False)).dot(p.transpose()))*np.eye(len(tem.assets))  # uncertainty matrix of views
        s = tem.covariance_matrix(excess=True, currency_open=False)

    cal = r + tuning_constant()*s.dot(p.transpose()).dot(np.linalg.inv(p.dot(tuning_constant()*s).dot(p.transpose()) + u)).dot(expectation - p.dot(r))
    for a in range(len(tem.assets)):
        result_dict[tem.assets[a]] = cal[a]
    return result_dict


def expected_weight(config, period_start, period_end, kind='end', currency_open=False, tracking_error=0.02):
    result_dict = {}
    tem = BlackLitterman(config, period_start, period_end, kind=kind)
    p = projection_matrix(len(tem.assets))
    if currency_open:
        u = (p.dot(tuning_constant()*tem.covariance_matrix(excess=True, currency_open=True)).dot(p.transpose()))*np.eye(len(tem.assets))
        ex_covariance = tem.covariance_matrix(excess=True, currency_open=True)
        covariance = ex_covariance.add(np.linalg.inv((np.linalg.inv(tuning_constant()*ex_covariance) + (p.transpose()).dot(u).dot(p))))
        active_risk = active_risk_aversion(config, period_start, period_end, currency_open=True, tracking_error=tracking_error)
        r = pd.DataFrame(list(expected_return(config, period_start, period_end, currency_open=True).values()))
    else:
        u = (p.dot(tuning_constant()*tem.covariance_matrix(excess=True, currency_open=False)).dot(p.transpose()))*np.eye(len(tem.assets))
        ex_covariance = tem.covariance_matrix(excess=True, currency_open=False)
        covariance = ex_covariance.add(np.linalg.inv((np.linalg.inv(tuning_constant() * ex_covariance) + (p.transpose()).dot(u).dot(p))))
        active_risk = active_risk_aversion(config, period_start, period_end, currency_open=False,tracking_error=tracking_error)
        r = pd.DataFrame(list(expected_return(config, period_start, period_end, currency_open=False).values()))
    one = pd.DataFrame(np.ones(len(tem.assets)))
    theta = ((one.transpose()).dot(np.linalg.inv(covariance)).dot(r)/((one.transpose()).dot(np.linalg.inv(covariance)).dot(one))).loc[0, 0]
    b = (np.linalg.inv(covariance).dot(r - theta*one))/(2*active_risk)
    for i in range(len(tem.assets)):
        result_dict[tem.assets[i]] = b[i][0]
    return result_dict


def active_risk_aversion(config, period_start, period_end, kind='end', currency_open=False, tracking_error=0.02):
    tem = BlackLitterman(config, period_start, period_end, kind=kind)
    p = projection_matrix(len(tem.assets))
    u = (p.dot(tuning_constant() * tem.covariance_matrix(excess=True, currency_open=currency_open)).dot(p.transpose())) * np.eye(len(tem.assets))
    r = pd.DataFrame(list(expected_return(config, period_start, period_end, currency_open=True).values()))
    ex_covariance = tem.covariance_matrix(excess=True, currency_open=True)

    covariance = ex_covariance.add(np.linalg.inv((np.linalg.inv(tuning_constant() * ex_covariance) + (p.transpose()).dot(u).dot(p))))
    inverse_covariance = np.linalg.inv(covariance)
    one = pd.DataFrame(np.ones(len(tem.assets)))
    theta = ((one.transpose()).dot(inverse_covariance).dot(r)/((one.transpose()).dot(inverse_covariance).dot(one))).loc[0, 0]
    a = (((inverse_covariance.dot(r - theta*one)).transpose()).dot(covariance).dot(inverse_covariance.dot(r - theta*one)))**(1/2)/(2*tracking_error)
    return a[0, 0]


if __name__ == "__main__":
    with open('../dataset/config_for_git.yml', encoding='utf-8') as f:
        configuration = yaml.load(f, Loader=yaml.SafeLoader)

    test = BlackLitterman(configuration, '2012-02', '2024-03', kind='end')
    print("covariance matrix: ", test.covariance_matrix())
    print(test.weight)
    print(test.risk_aversion_coefficient(0))
    print("implied return: ", test.implied_return())
    print(test.expectation())
    print(test.covariance_matrix())
    print(projection_matrix(len(configuration['assets'])))
    print("expected return: ", expected_return(configuration, '2012-02', '2024-03'))
    print("proposed weight: ", expected_weight(configuration, '2012-02', '2024-03'))
    print("active risk aversion: ", active_risk_aversion(configuration, '2012-02', '2024-03'))
