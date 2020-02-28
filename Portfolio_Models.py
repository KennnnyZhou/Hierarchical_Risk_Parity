from HRP import *
from MVP import *
import matplotlib.pyplot as plt
import seaborn as sns
import pyfolio as pf
import matplotlib
sns.set_context("talk")
sns.set_style("darkgrid")
sns.set_palette(sns.color_palette(palette='Set2'))
matplotlib.rcParams.update({'font.family': 'Arial',
                            'font.size': 25})


class models:
    def __init__(self):
        return

    def get_weight(self, data, model):
        if model == 'HRP':
            model_weight = self.get_HRP_weights(data)
        elif model == 'EW':
            model_weight = self.get_EW_weights(data)
        elif model == 'MVP':
            model_weight = self.get_MVP_weights(data)
        elif model == 'IVP':
            model_weight = self.get_IVP_weights(data)
        elif model == 'RDM':
            model_weight = self.get_RDM_weights(data)
        elif model == 'UMVP':
            model_weight = self.get_UMVP_weights(data)
        else:
            raise ValueError('No such Model!')

        return model_weight

    def get_vol(self, data, weights):
        # normal volatility calculation
        cov = data.cov().values
        weights = weights.values
        var = np.dot(weights.T.dot(cov), weights)
        return np.sqrt(var * 252)

    def get_HRP_weights(self, price_data):
        HRP_result = get_HRP_result(price_data)
        return HRP_result[0]

    def get_EW_weights(self, price_data):
        N = price_data.shape[1]
        EW_weights = [1 / N] * N
        return pd.Series(EW_weights)

    def get_IVP_weights(self, price_data):
        return_data = price_data.pct_change().dropna()
        cov = return_data.cov().values
        ivp_weights = 1. / np.diag(cov)
        ivp_weights /= ivp_weights.sum()
        return pd.Series(ivp_weights)

    def get_MVP_weights(self, price_data):
        mvp_weights = MVP(price_data)[0]
        return mvp_weights

    def get_UMVP_weights(self, price_data):
        umvp_weights = UMVP(price_data)
        return umvp_weights

    def get_RDM_weights(self, price_data, seed=1):
        np.random.seed(seed)
        N = price_data.shape[1]
        rdm = np.random.randint(1, 100, N)
        rdm = rdm / rdm.sum()
        return pd.Series(rdm)


class out_test(models):
    def __init__(self, full_p_data, cal_period=0):
        models.__init__(self)
        self.full_p_data = full_p_data
        self.period = cal_period
        self.model_weights = {}

    def cal_weight(self, price_data, model):
        last_index = price_data.index[-1]
        last_loc = self.full_p_data.index.get_loc(last_index)
        data = self.full_p_data.iloc[last_loc - self.period:last_loc, :]
        weights = self.get_weight(data, model)
        return weights
        # how to realize rolling with steps in Python ?

    def rebalance_weights(self, price_data, model, freq='BM'):
        tickers = price_data.columns.tolist()
        weight_df = price_data.resample(freq).apply(
            self.cal_weight, model=model)
        weight_df.columns = tickers
        weights = weight_df.shift().dropna()  # 将利用前一个月算出的weight数据对齐到下一个月的return
        return weights

    def rebalance_test(self, price_data, model, freq='BM'):
        weights_mon = self.rebalance_weights(price_data, model, freq='BM')
        return_mon = price_data.resample(freq).last().pct_change().dropna()
        self.model_weights[model] = weights_mon

        model_return = return_mon * weights_mon
        mon_return = pd.DataFrame(model_return.sum(axis=1), columns=[model])
        return mon_return

    def run_test(self, models_list):
        print('Out-of-sample test starts:')
        for i, model in enumerate(models_list):
            result = self.rebalance_test(
                self.full_p_data.iloc[self.period + 1:, :], model=model)
            print('%s is finished' % model)
            if i == 0:
                self.r_result = result
            else:
                self.r_result = pd.merge(
                    self.r_result, result, left_index=True, right_index=True)

        self.r_result.columns = models_list
        return self.r_result, self.model_weights

    def plot_cum_return(self, r=None):
        if r is not None:
            self.r_result = r

        cum_return = (1 + self.r_result).cumprod()
        plt.figure()
        colors = sns.color_palette(
            palette='Set3', n_colors=self.r_result.shape[1])
        for i, n in enumerate(self.r_result.columns):
            cum_return[n].plot(color=colors[i])
        plt.legend(loc=0)
        plt.title('Cumulative returns')

    def plot_SR(self, r=None, Rf=0):
        if r is not None:
            self.r_result = r

        fig, ax = plt.subplots()
        SR = pf.timeseries.sharpe_ratio(
            self.r_result, risk_free=Rf, period='monthly')

        sns.barplot(x=SR, y=self.r_result.columns, ax=ax)
        ax.get_yticklabels()[1].set_color("red")
        plt.title('Sharpe Ratio')

    def plot_ann_vol(self, r=None):
        if r is not None:
            self.r_result = r

        fig, ax = plt.subplots()
        ann_vol = pf.timeseries.annual_volatility(
            self.r_result, period='monthly')
        sns.barplot(x=ann_vol, y=self.r_result.columns, ax=ax)
        ax.get_yticklabels()[1].set_color("red")
        plt.title('Annualized Volatility')


class in_test(models):
    def __init__(self, full_p_data, models_list):
        models.__init__(self)
        self.full_p_data = full_p_data
        self.return_data = self.full_p_data.pct_change().dropna()
        self.r_result = []
        self.vol_result = []
        self.weights = {}
        self.models_list = models_list

    def run_test(self, models_list):
        print('In-sample test starts:')
        for i, model in enumerate(models_list):
            w = self.get_weight(self.full_p_data, model)
            ann_r = w.values.dot(self.return_data.mean().values * 252)
            ann_vol = self.get_vol(self.return_data, w)

            self.r_result.append(ann_r)
            self.vol_result.append(ann_vol)
            self.weights[model] = w

            print('%s is finished' % model)

        return self.r_result, self.vol_result, self.weights

    def plot_SR(self, Rf=0):
        fig, ax = plt.subplots()
        SR = (np.array(self.r_result) - Rf) / np.array(self.vol_result)
        sns.barplot(x=SR, y=self.models_list, ax=ax)
        ax.get_yticklabels()[1].set_color("red")
        plt.title('Sharpe Ratio')

    def plot_r_vol(self):
        # plot annualized return plot
        fig, ax = plt.subplots(ncols=2)
        sns.barplot(x=self.r_result, y=self.models_list, ax=ax[0])
        ax[0].get_yticklabels()[1].set_color("red")
        ax[0].set(title='Annualized Return')

        # plot annualized volatility plot
        sns.barplot(x=self.vol_result, y=self.models_list, ax=ax[1])
        ax[1].get_yticklabels()[1].set_color("red")
        ax[1].set(title='Annualized Volatility')

    def plot_frontier(self):
        plot_frontier(self.full_p_data)
        colors = sns.color_palette(
            palette='Set2', n_colors=len(
                self.models_list))

        for i in range(len(self.models_list)):
            if self.models_list[i] == 'HRP':
                plt.plot(
                    self.vol_result[i],
                    self.r_result[i],
                    'D',
                    color=colors[i],
                    label="point")
                plt.text(
                    self.vol_result[i] + 0.003,
                    self.r_result[i] - 0.004,
                    self.models_list[i],
                    fontsize=18,
                    color='red')
            elif self.models_list[i] != 'MVP':
                plt.plot(
                    self.vol_result[i],
                    self.r_result[i],
                    'ro',
                    color=colors[i],
                    label="point")
                plt.text(
                    self.vol_result[i] + 0.003,
                    self.r_result[i] - 0.004,
                    self.models_list[i],
                    fontsize=18)
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
