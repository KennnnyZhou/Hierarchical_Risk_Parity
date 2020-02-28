import matplotlib.pyplot as plt
from Portfolio_Models import *
import pandas as pd


if __name__ == '__main__':

    file = 'ETF_Index.csv'
    price_data = pd.read_csv(file, index_col=0)
    price_data.index = pd.to_datetime(price_data.index, format="%Y-%m-%d")
    models_list = ['IVP', 'HRP', 'EW', 'MVP', 'UMVP', 'RDM']

    ########### in-sample ############

    in_start_date = '2012-01-01'
    in_end_date = '2016-01-01'
    in_sample = price_data[(price_data.index >= in_start_date ) & (price_data.index < in_end_date)]

    in_test = in_test(in_sample, models_list)
    r, vol, weights = in_test.run_test(models_list)
    in_test.plot_SR()
    in_test.plot_r_vol()
    in_test.plot_frontier()


    ########### out-of-sample ##############
    out_start_date = '2012-01-01'
    out_end_date = '2019-01-01'
    out_sample = price_data[(price_data.index >= out_start_date ) & (price_data.index < out_end_date)]

    period = 60
    out_test = out_test(out_sample, period)

    out_r, out_weights = out_test.run_test(models_list)

    out_test.plot_cum_return()
    out_test.plot_SR()
    out_test.plot_ann_vol()

    plt.show()


