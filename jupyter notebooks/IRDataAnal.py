# this is the site file

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from typing import Optional, Union, Callable, Tuple, List
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.cross_decomposition._pls import _center_scale_xy
from scipy.signal import find_peaks

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg as linalg
import copy
from pathlib import Path
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter

import math


def autoBaselinePoints(wavenumber, intensity, peak_loc_wavenumber, isosbestic, n_sim): 

    # compute the two baseline points by looking for isosbestic points
    # alternatively, compute baseline points thats in the valley of the spectra after first or second derivative transformation
    # intensity should have shape (nspectra x nfeatures), and should be from a single experiment
    # wavenumber should be an array of increasing value (need to manual invert the wavenumber before feeding into this function)

    # generate fake ir datapoints by linearly extrapolating from the nearby real data points
    fake_intensity_all = []
    fake_wavenumbers = np.linspace(wavenumber[0], wavenumber[-1], n_sim)

    for j in range(intensity.shape[0]): 

        all_slopes = np.diff(intensity[j, :]) / np.diff(wavenumber)
        all_intercept = intensity[j][1:] - wavenumber[1:]*all_slopes
        
        slope_counter = 0
        fake_intensity = []
        for i in range(len(fake_wavenumbers)): 
            # print(slope_counter)
            if fake_wavenumbers[i] >= wavenumber[slope_counter] and fake_wavenumbers[i] < wavenumber[slope_counter+1]: 
                fake_intensity.append(fake_wavenumbers[i]*all_slopes[slope_counter] + all_intercept[slope_counter])
            else: 
                slope_counter += 1
                try: 
                    fake_intensity.append(fake_wavenumbers[i]*all_slopes[slope_counter] + all_intercept[slope_counter])
                except: 
                    fake_intensity.append(fake_wavenumbers[i]*all_slopes[slope_counter-1] + all_intercept[slope_counter-1])

        fake_intensity_all.append(fake_intensity)

    fake_intensity_all = np.array(fake_intensity_all)

    # compute the abs values of std in the fake intensity
    # valley corresponds to isosbestic points and peaks corresponds to valley in the original IR data
    absVar = np.abs(fake_intensity_all.std(axis=0))
    peaks, _ = find_peaks(absVar)  
    valleys, _ = find_peaks(-absVar)  

    # identify the correct peak location 
    peaks_wavenumber = fake_wavenumbers[peaks]
    # print(peaks_wavenumber)
    diff = np.abs(peak_loc_wavenumber - peaks_wavenumber)
    desired_peak_location = np.where(diff==diff.min())[0][0] 

    if isosbestic: 
        distances = np.abs(valleys - peaks[desired_peak_location])   # Distance from the peak to each valley
        nearest_valley_indices = np.argsort(distances)[:2]  # Indices of the two nearest valleys
        nearest_valleys = valleys[nearest_valley_indices]

        return fake_wavenumbers, fake_intensity_all, np.sort(nearest_valleys)

    else: 
        distances = np.abs(peaks - peaks[desired_peak_location])   # Distance from the peak to each valley
        nearest_peaks_indices = np.argsort(distances)[1:3]  # Indices of the two nearest valleys
        nearest_peaks = peaks[nearest_peaks_indices]

        return fake_wavenumbers, fake_intensity_all, np.sort(nearest_peaks)


# def GridSearchCVbyExp(rgr, paramters, scoring, cv): 

#     # perform cv, instead on random subset of the training data, on different experiment

#     s, e = 0, ir.train_dataLList[0]
#     for i in range(len(ir.train_dataLList)): 

#         x_val_s = ir.train_ir[:s, :]
#         x_val_e = ir.train_ir[e:, :]
#         x_val = np.vstack((x_val_s, x_val_e))

#         y_val_s = ir.train_lc[:s, :]
#         y_val_e = ir.train_lc[e:, :]
#         y_val = np.vstack((y_val_s, y_val_e))

#         x_train = ir.train_ir[s:e, :]
#         y_train = ir.train_lc[s:e, :]


# functions for normalizing data
def minmax(a, min_, max_): 

    return (a-min_) / (max_-min_)


def unnormalizeMinmax(a, min_, max_): 
    return a * (max_ - min_) + min_


# functions for calculating IR peak heights and area
def heightTwoPoints(baseline_wavenumber, baseline_intensity, spec): 

    fp = (baseline_wavenumber[0], baseline_intensity[0])
    sp = (baseline_wavenumber[1], baseline_intensity[1])

    real_fp_wavenumber = np.where(np.abs(fp[0]-spec[:, 0])==np.abs(fp[0]-spec[:, 0]).min())[0][0]
    real_sp_wavenumber = np.where(np.abs(sp[0]-spec[:, 0])==np.abs(sp[0]-spec[:, 0]).min())[0][0]

    wavenumber = spec[:, 0]
    Abs = spec[:, 1]

    peak_loc = np.where(Abs==Abs[real_fp_wavenumber:real_sp_wavenumber].max())[0][0]
    peak_pos = [wavenumber[peak_loc], Abs.max()]
    n = math.dist(peak_pos, fp)
    m = math.dist(peak_pos, sp)
    l = math.dist(fp, sp)

    x = (n**2 + l**2 - m**2) / (2*l)
    h2 = n**2 - x**2
    if h2 >= 0: 
        return np.sqrt(h2)
    else: 
        return 0

def peakAreaTwoPoints(baseline_wavenumber, baseline_intensity, spec):

    fp = (baseline_wavenumber[0], baseline_intensity[0])
    sp = (baseline_wavenumber[1], baseline_intensity[1])

    # correct baseline by applying two point methods
    wavenumber = spec[:, 0]
    Abs = spec[:, 1]

    s = (sp[1]-fp[1]) / (sp[0]-fp[0])
    b = sp[1] - s*sp[0]

    ref_line = wavenumber * s + b
    corrected_spec = Abs - ref_line

    return np.trapz(corrected_spec, wavenumber)

def heightToZero(_, _2, spec): 
    return spec[:, 1].max()


# function for generating peak height / area trends
def generateTrends(trend_method, ir_data, ir_data_length, ir_wavenumber, peaks_wavenumber, isosbestic=True, diagnostic=False, n_sim=700): 
    s, e = 0, ir_data_length[0]
    trend_list = []

    for i in range(len(ir_data_length)): 

        temp_abs = ir_data[s:e, :]
        fake_wavenumbers, fake_intensity_all, baseline_index = autoBaselinePoints(ir_wavenumber, temp_abs, peaks_wavenumber, isosbestic, n_sim)
        # print(baseline_index)

        if diagnostic: 
            plt.figure()
            plt.plot(fake_wavenumbers, fake_intensity_all.T);
            plt.scatter(fake_wavenumbers[baseline_index], fake_intensity_all[diagnostic, baseline_index])


        temp_trend = []
        
        for j in range(temp_abs.shape[0]):
            spec = np.hstack((ir_wavenumber[:, np.newaxis], temp_abs[j, :][:, np.newaxis]))
            temp_trend.append(trend_method(fake_wavenumbers[baseline_index], fake_intensity_all[j, baseline_index], spec))

        trend_list.append(temp_trend)

        s = e
        if i == len(ir_data_length) - 1: 
            e = -1
        else: 
            e = e + ir_data_length[i+1]

    return trend_list


# function for prep univariate calibration data
def prepDataUnivariateCal(ir_trend_list, ir_data_length, lc_data, lc_convert_prod=9.009e-5): 

    s, e = 0, ir_data_length[0]

    univariate_cal_data = []

    for i in range(len(ir_data_length)): 

        univariate_cal_data.append([lc_data[s:e, :]*lc_convert_prod, ir_trend_list[i]])

        s = e
        if i == len(ir_data_length) - 1: 
            e = -1
        else: 
            e = e + ir_data_length[i+1]

    return univariate_cal_data


# custom regressors for sklearn
class PCRegression(BaseEstimator):

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y): 
        self.pca = PCA(self.n_components).fit(X)
        X_transform = self.pca.transform(X)
        self.regr = LinearRegression()
        self.regr.fit(X_transform, y)

        return 

    def predict(self, x): 
        x_transform = self.pca.transform(x)

        return self.regr.predict(x_transform)

class ScaledSVR(BaseEstimator): 

    def __init__(self, kernel, gamma, C, epsilon):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y): 
        self.regr = SVR(kernel=self.kernel, gamma=self.gamma, C=self.C, epsilon=self.epsilon)
        scaled_X, scaled_y, self.x_mean, self.y_mean, self.x_std, self.y_std = _center_scale_xy(X, y, scale=True)
        self.regr.fit(scaled_X,scaled_y.ravel())

        return

    def predict(self, x_test): 
        x_test_scaled = (x_test-self.x_mean) / self.x_std
        pred_scaled = self.regr.predict(x_test_scaled)

        return pred_scaled * self.y_std + self.y_mean


class calibrationData:

    def __init__(self, raw_ir_dir, lc_dir, comp_names):
        
        self.dataLList = []
        self.ir_time_stamp = []
        self.lc_time_stamp = []
        self.bg_spec_list = []

        # load IR data
        entries = Path(raw_ir_dir)
        self.raw_ir_data_full = []

        print('Loading IR data......')
        for entry in sorted(entries.iterdir()):

            if entry.name[0] == '.':
                continue        

            print(entry.name)

            self.raw_ir_data_full.append(self.loadSpec(raw_ir_dir + entry.name + '/'))

        self.raw_ir_data_full = np.concatenate(self.raw_ir_data_full, axis=0)

        # load lc data
        entries = Path(lc_dir)
        self.lc_data = []
        self.lc_data_plot = {}
        self.lc_data_name = []

        print('Loading LC data......')
        for entry in sorted(entries.iterdir()):

            if entry.name[0] == '.' or entry.name[0] == '~':
                continue        
            
            print(entry.name)
            all_lc_data = pd.read_excel(lc_dir + entry.name)
            self.lc_data_plot[entry.name[:-5]] = all_lc_data
            self.lc_data_name.append(entry.name[:-5])
            self.lc_data.append(all_lc_data[comp_names].to_numpy())
            self.lc_time_stamp.append(all_lc_data[['DateTime', 'Events']])

        self.lc_data = np.concatenate(self.lc_data, axis=0)

        # sync lc and IR time stamp
        # ir_index_match_lc has a list of start and end index that matches the time for each lc experiment
        self.new_dataLList = []
        self.ir_time = []
        self.ir_index_match_lc = []

        s, e = 0, self.dataLList[0]
        temp_ir_data = []
        for i in range(len(self.lc_time_stamp)):
            lc_t = self.setTimeZero(self.lc_time_stamp[i])
            ir_t = self.setTimeZero(self.ir_time_stamp[i])
            self.lc_data_plot[self.lc_data_name[i]]['DateTime'] = lc_t
            

            match_index = []
            for j in range(len(lc_t)): 
                
                diff = np.abs(ir_t-lc_t[j])
                temp_match_index = np.where(diff == diff.min())[0][0]
                match_index.append(temp_match_index)

            self.ir_time.append(ir_t)
            self.ir_index_match_lc.append([match_index[0], match_index[-1]])

            temp_ir_data.append(self.raw_ir_data_full[s:e, :][match_index, :])
            s = e
            if i == len(self.lc_time_stamp) - 1:
                e = -1
            else: 
                e = e + self.dataLList[i+1] 

            self.new_dataLList.append(temp_ir_data[i].shape[0])

        self.raw_ir_data = np.concatenate(temp_ir_data, axis=0)


    def setTimeZero(self, time_stamp, event='ir sync'):

        # zero the DateTime column of the pd dataframe by looking for the event
        # return a 1-d time array in hours

        temp_t = np.zeros((time_stamp.shape[0], ))
        t_0 = time_stamp[time_stamp['Events'] == 'ir sync']['DateTime'].values[0]
        dt = time_stamp['DateTime'].to_numpy()
        for j in range(len(temp_t)): 
            temp_t[j] = dt[j].hour * 60 * 60 + dt[j].minute * 60 + dt[j].second
        temp_t = (temp_t - t_0.hour * 60 * 60 - t_0.minute * 60 - t_0.second)/60/60

        return temp_t

    def loadSpec(self, folder_name):

        entries = Path(folder_name)
        spectra_train = []

        for entry in sorted(entries.iterdir()):

            if entry.name[0] == '.':
                continue
            try: 
                d = pd.read_csv(folder_name + entry.name).to_numpy()
                if d.shape[0] > 589: 
                    # the length of 589 corresponds to batch IR raw data size, for flow ir, the size is 839
                    # for calibration purposes, it needs to be truncated
                    d = d[250:, :]
                spectra_train.append(d)
            except UnicodeDecodeError: 
                # this error will take place when trying to use read_csv to load xlsx
                # which is convient way to read time stamp
                ts = pd.read_excel(folder_name + entry.name)
                self.ir_time_stamp.append(ts[['DateTime', 'Events']])

        # spectra_train should have shape ndata x nfeature
        spec = np.array(spectra_train)[:, :, -1]
        self.wave_number = spectra_train[-1][:, 0]
        self.bg_spec_list.append(spec[0, :])
        self.dataLList.append(spec.shape[0])

        return spec


    def TrainTestQTscores_PLS(self, PLSR): 

        train_ir_scale = (self.train_ir - PLSR.x.mean_) / PLSR.x.std_
        test_ir_scale = (self.test_ir - PLSR.x.mean_) / PLSR.x.std_

        T_mat_train = train_ir_scale @ PLSR.x_rotations_
        T_mat_test = test_ir_scale @ PLSR.x_rotations_
        sa = np.var(T_mat_train, axis=0)
        Q_mat = np.eye(self.train_ir.shape[-1]) - PLSR.x_rotations_ @ PLSR.x_rotations_.T

        # compute T2 and Q score for each training ir spectrum
        T2_training = []
        Q_training = []

        for i in range(self.train_ir.shape[0]): 
            T2_training.append(np.sum((T_mat_train[i, :]**2 / sa)))
            Q_training.append(train_ir_scale[i, :].reshape(1, -1) @ Q_mat @ train_ir_scale[i, :].reshape(-1, 1))

        # compute T2 score for each testing ir spectrum
        T2_testing = []
        Q_testing = []
        for i in range(self.test_ir.shape[0]): 
            T2_testing.append(np.sum((T_mat_test[i, :]**2 / sa)))
            Q_testing.append(test_ir_scale[i, :].reshape(1, -1) @ Q_mat @ test_ir_scale[i, :].reshape(-1, 1))

        return np.array(T2_training), np.array(T2_testing), np.array(Q_training), np.array(Q_testing)


    def TrainTestQTscores(self, pca_rank): 

        pca = PCA(10).fit(self.train_ir)
        cumexpratio = np.cumsum(pca.explained_variance_ratio_)
        first99 = np.where(cumexpratio >= 0.99)[0][0]
        print(first99+1)
        pca = PCA(first99+1).fit(self.train_ir)
        transformed_data = pca.transform(self.train_ir)
        tdata_test = pca.transform(self.test_ir)

        # variances of each pca component from training dataset
        sa = np.var(transformed_data, axis=0)

        # pca component matrix used to compute Q values
        Q_mat = np.eye(self.train_ir.shape[-1]) - pca.components_.T @ pca.components_

        # compute T2 and Q score for each training ir spectrum
        T2_training = []
        Q_training = []
        for i in range(transformed_data.shape[0]): 
            T2_training.append(np.sum((transformed_data[i, :]**2 / sa)))
            Q_training.append(self.train_ir[i, :].reshape(1, -1) @ Q_mat @ self.train_ir[i, :].reshape(-1, 1))

        # compute T2 score for each testing ir spectrum
        T2_testing = []
        Q_testing = []
        for i in range(tdata_test.shape[0]): 
            T2_testing.append(np.sum((tdata_test[i, :]**2 / sa)))
            Q_testing.append(self.test_ir[i, :].reshape(1, -1) @ Q_mat @ self.test_ir[i, :].reshape(-1, 1))

        return np.array(T2_training), np.array(T2_testing), np.array(Q_training).reshape(-1, ), np.array(Q_testing).reshape(-1, ), pca

    def TrainTestQTscores_scaled(self, pca_rank): 

        train_ir_scale = (self.train_ir - self.train_ir.mean(axis=0)) / self.train_ir.std(axis=0, ddof=1)
        test_ir_scale = (self.test_ir - self.train_ir.mean(axis=0)) / self.train_ir.std(axis=0, ddof=1)

        pca = PCA(pca_rank).fit(train_ir_scale)
        transformed_data = pca.transform(train_ir_scale)
        tdata_test = pca.transform(test_ir_scale)

        # variances of each pca component from training dataset
        sa = np.var(transformed_data, axis=0)

        # pca component matrix used to compute Q values
        Q_mat = np.eye(self.train_ir.shape[-1]) - pca.components_.T @ pca.components_

        # compute T2 and Q score for each training ir spectrum
        T2_training = []
        Q_training = []
        for i in range(transformed_data.shape[0]): 
            T2_training.append(np.sum((transformed_data[i, :]**2 / sa)))
            Q_training.append(train_ir_scale[i, :].reshape(1, -1) @ Q_mat @ train_ir_scale[i, :].reshape(-1, 1))

        # compute T2 score for each testing ir spectrum
        T2_testing = []
        Q_testing = []
        for i in range(tdata_test.shape[0]): 
            T2_testing.append(np.sum((tdata_test[i, :]**2 / sa)))
            Q_testing.append(test_ir_scale[i, :].reshape(1, -1) @ Q_mat @ test_ir_scale[i, :].reshape(-1, 1))

        return np.array(T2_training), np.array(T2_testing), np.array(Q_training), np.array(Q_testing), pca


    def TrainTestPCA(self, comp_to_visual, train_labels, test_labels, skip=1): 

        llist = self.train_dataLList
        testllist = self.test_dataLList
        print(self.train_dataLList)

        pca = PCA(20).fit(self.train_ir)
        transformed_data = pca.transform(self.train_ir)

        s, e = 0, llist[0]
        ind1, ind2 = comp_to_visual

        for i in range(len(llist)): 

            plt.scatter(transformed_data[s:e:skip, ind1], transformed_data[s:e:skip, ind2], marker='o', label=train_labels[i], alpha=0.15)
            pp = 20

            plt.annotate(i+1, (transformed_data[s+pp, ind1], transformed_data[s+pp, ind2]))

            s = e
            if i == len(llist) - 1: 
                e = -1
            else: 
                e = e + llist[i+1]

        transformed_data_test = pca.transform(self.test_ir)

        s, e = 0, testllist[0]

        for i in range(len(testllist)): 

            plt.scatter(transformed_data_test[s:e:skip, ind1], transformed_data_test[s:e:skip, ind2], marker='x', label=test_labels[i])
            # plt.annotate(i+1, (transformed_data_test[s, ind1], transformed_data_test[s, ind2]))

            s = e
            if i == len(testllist) - 1: 
                e = -1
            else: 
                e = e + testllist[i+1]

        plt.xlabel(f'PC-{ind1+1} ({str(self.pca.explained_variance_ratio_[ind1]*100)[:4]}%)')
        plt.ylabel(f'PC-{ind2+1} ({str(self.pca.explained_variance_ratio_[ind2]*100)[:4]}%)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True)

        # an = np.linspace(0, 2 * np.pi, 100)
        # plt.plot(transformed_data[:, ind1].max()*np.cos(an), transformed_data[:, ind2].max()*np.sin(an), c='k', lw=0.5) 
        # plt.axvline(x=0, c='k', lw=0.3)
        # plt.axhline(y=0, c='k', lw=0.3)

    def visualizeComponents(self, comp_to_visual, exp_labels, skip=1): 



        llist = self.new_dataLList

        shape = ['o', 'x', '>', 'p', '*', '+', '<', '1', '2', '3', '4', '8', '|']
        clist = ['b', 'orange', 'g', 'r', 'm', 'brown', 'cyan']
        pca = PCA(20).fit(self.raw_ir_data_preprocess)
        transformed_data = pca.transform(self.raw_ir_data_preprocess)

        s, e = 0, llist[0]
        ind1, ind2 = comp_to_visual

        for i in range(len(llist)): 

            alpha_list = np.linspace(0.1, 1, e-s)[::-1]

            plt.scatter(transformed_data[s:e:skip, ind1], transformed_data[s:e:skip, ind2], label=exp_labels[i], marker=shape[i], c=clist[i], alpha=alpha_list)
            # plt.annotate(i+1, (transformed_data[s, ind1], transformed_data[s, ind2]), c=clist[i])

            s = e
            if i == len(llist) - 1: 
                e = -1
            else: 
                e = e + llist[i+1]

        plt.xlabel(f'PC-{ind1+1} ({str(pca.explained_variance_ratio_[ind1]*100)[:4]}%)')
        plt.ylabel(f'PC-{ind2+1} ({str(pca.explained_variance_ratio_[ind2]*100)[:4]}%)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=4, fancybox=True, shadow=True)

        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(transformed_data[:, ind1].max()*np.cos(an), transformed_data[:, ind2].max()*np.sin(an), c='k', lw=0.5) 
        plt.axvline(x=0, c='k', lw=0.3)
        plt.axhline(y=0, c='k', lw=0.3)
        

    def loadExtraSpectra(self, extra_spectra):

        # loading spectra from a csv instead of a folder
        self.dataLList.append(extra_spectra.shape[0])
        self.raw_ir_data = np.vstack((self.raw_ir_data, extra_spectra))

    def matchSpectra(self, ind): 
        self.raw_ir_data = self.raw_ir_data[ind, :]

    def preprocess(self, start_list, end_list, skip=False, isDerivative=False, isSNV=False,
                        isMSC=False, isSmooth=False, isBaselineCorr=False, isTruncateSS=np.array([]), 
                        isTruncateData=False, isSubtractFirst=False, isSubtractAny=False,truncateWaveNUmber=[]):

        # raw_ir_data is the raw ir data that matches LC time but without any preprocessing
        # real_raw is the original raw ir data without mactching LC times and without preprocessing
        # raw_ir_preprocess is the ir data that matches LC time and with preprocessing

        # dataLList is a list that contains the original # of IR spectra within each experiment
        # new_dataLList is the list that contains the # of IR spectra that matches LC data within each experiment

        # as for now (111924), only subtract any and 2nd derivative preprocessing are implemented on both matched IR and raw IR

        self.sel_ind = np.concatenate([np.arange(start_list[i], end_list[i]) for i in range(len(start_list))])
        self.wave_number_preprocess = self.wave_number[self.sel_ind]
        self.raw_ir_data_preprocess = self.raw_ir_data[:, self.sel_ind]
        # self.raw_ir_data_preprocess = self.raw_ir_data_preprocess[:, ::skip]

        self.real_raw = self.raw_ir_data_full[:, self.sel_ind]
        # self.real_raw = self.real_raw[:, ::skip]

        if isSubtractAny: 
            # if isSubstractAny == True, then substract the selected spectrum from each experiment

            s, e, s2, e2 = 0, self.new_dataLList[0], 0, self.dataLList[0]
            for i in range(len(self.new_dataLList)): 

                selected_spectrum = self.real_raw[s2:e2, :][isSubtractAny[i], :] 

                self.raw_ir_data_preprocess[s:e, :] = self.raw_ir_data_preprocess[s:e, :] - selected_spectrum
                self.real_raw[s2:e2, :] = self.real_raw[s2:e2, :] - selected_spectrum

                s, s2 = e, e2

                if i == len(self.new_dataLList) - 1: 
                    e = -1
                    e2 = -1
                else: 
                    e = e + self.new_dataLList[i+1]
                    e2 = e2 + self.dataLList[i+1]


        if isSubtractFirst: 
            # if isSubstractFirst == True, then substract the first spectrum from each experiment
            s, e, s2, e2 = 0, self.new_dataLList[0], 0, self.dataLList[0]
            for i in range(len(self.new_dataLList)): 

                self.raw_ir_data_preprocess[s:e, :] = self.raw_ir_data_preprocess[s:e, :] - self.bg_spec_list[i][self.sel_ind]
                self.real_raw[s2:e2, :] = self.real_raw[s2:e2, :] - self.bg_spec_list[i][self.sel_ind]

                s = e
                s2 = e2
                if i == len(self.new_dataLList) - 1: 
                    e = -1
                    e2 = -1
                else: 
                    e = e + self.new_dataLList[i+1]
                    e2 = e2 + self.dataLList[i+1]

        if isSmooth:
            self.raw_ir_data_preprocess = savgol_filter(x=self.raw_ir_data_preprocess, window_length=isSmooth[0], polyorder=isSmooth[1], axis=-1)
            self.real_raw = savgol_filter(x=self.real_raw, window_length=isSmooth[0], polyorder=isSmooth[1], axis=-1)

        if isBaselineCorr: 

            temp = []
            for i in range(self.raw_ir_data_preprocess.shape[0]): 
                temp.append(self.twoPointBaseline([self.wave_number_[0], self.raw_ir_data_preprocess[i, 0]], 
                    [self.wave_number_[-1], self.raw_ir_data_preprocess[i, -1]], self.wave_number_, self.raw_ir_data_preprocess[i, :]))
            self.raw_ir_data_preprocess = np.array(temp)

        # if isTruncateData:  # not useful anymore
        #     # if isTruncateData is not false, then only keep [isTruncateData[0]:isTruncateData[1]] for each experiment

        #     s, e = 0, self.dataLList[0]
        #     temp_data = []
        #     for i in range(len(self.dataLList)): 
        #         temp_data.append(self.raw_ir_data_preprocess[s:e, :][isTruncateData[0]:isTruncateData[1], :])
        #         self.new_dataLList[i] = temp_data[i].shape[0]

        #         s = e 
        #         if i == len(self.dataLList) - 1: 
        #             e = -1
        #         else: 
        #             e = e + self.dataLList[i+1]

        #     self.raw_ir_data_preprocess = np.vstack(temp_data)
        
        if isSNV:
            self.raw_ir_data_preprocess = normalize(self.raw_ir_data_preprocess)

        if isMSC: 
            self.mean_ref = np.mean(self.raw_ir_data_preprocess, axis=0)
            for i in range(self.raw_ir_data_preprocess.shape[0]): 
                fit = np.polyfit(self.mean_ref, self.raw_ir_data_preprocess[i, :], 1)
                self.raw_ir_data_preprocess[i, :] = (self.raw_ir_data_preprocess[i, :] - fit[1]) / fit[0]

        if isDerivative == 1:
            _1st = [np.gradient(self.raw_ir_data_preprocess[i, :]) for i in range(self.raw_ir_data.shape[0])]
            self.raw_ir_data_preprocess = np.array(_1st)
            _1st = [np.gradient(self.real_raw[i, :]) for i in range(self.real_raw.shape[0])]
            self.real_raw = np.array(_1st)
        if isDerivative == 2:
            _2nd = [-np.gradient(np.gradient(self.raw_ir_data_preprocess[i, :])) for i in range(self.raw_ir_data.shape[0])]
            self.raw_ir_data_preprocess = np.array(_2nd)
            _2nd = [-np.gradient(np.gradient(self.real_raw[i, :])) for i in range(self.real_raw.shape[0])]
            self.real_raw = np.array(_2nd)

        if isTruncateSS.any(): 
            self.raw_ir_data_preprocess = self.raw_ir_data_preprocess[isTruncateSS, :]
            self.lc_data = self.lc_data[isTruncateSS, :]

        if truncateWaveNUmber: 
            new_ir = []
            new_ir_wavenumber = []
            for i in range(len(truncateWaveNUmber)): 
                new_ir.append(self.raw_ir_data_preprocess[:, truncateWaveNUmber[i][0]:truncateWaveNUmber[i][1]])
                new_ir_wavenumber.append(self.wave_number_preprocess[truncateWaveNUmber[i][0]:truncateWaveNUmber[i][1]])
            self.raw_ir_data_preprocess = np.hstack(new_ir)
            self.wave_number_preprocess = np.hstack(new_ir_wavenumber)

        if skip: 

            self.raw_ir_data_preprocess = self.raw_ir_data_preprocess[::skip, :]


        # self.pca = PCA(20).fit(self.raw_ir_data_preprocess)


    def twoPointBaseline(self, p1, p2, wavenum, absorbance): 
        s = (p2[1]-p1[1]) / (p2[0]-p1[0])
        b = p1[1] - s*p1[0]
        
        ref_line = wavenum * s + b
        corr = ref_line - 0
    
        return absorbance - corr

    def trainTestSplit(self, test_exp_inds, ignore=[]): 

        # exp_ind is a list of experiment index that will be used as testing data
        # the remaining data will used for training

        self.train_ir, self.train_lc, self.test_ir, self.test_lc = [], [], [], []

        s, e = 0, self.new_dataLList[0]
        self.train_dataLList = []
        self.test_dataLList = []
        for i in range(len(self.new_dataLList)): 

            if i in test_exp_inds: 
                self.test_ir.append(self.raw_ir_data_preprocess[s:e, :])
                self.test_lc.append(self.lc_data[s:e, :])
                self.test_dataLList.append(e-s)
            elif i in ignore: 
                pass
            else: 
                self.train_ir.append(self.raw_ir_data_preprocess[s:e, :])
                self.train_lc.append(self.lc_data[s:e, :])
                self.train_dataLList.append(e-s)

            s = e
            if i == len(self.new_dataLList) - 1: 
                e = -1
            else: 
                e = e + self.new_dataLList[i+1]

        self.train_ir = np.concatenate(self.train_ir, axis=0)
        self.train_lc = np.concatenate(self.train_lc, axis=0)
        self.test_ir = np.concatenate(self.test_ir, axis=0)
        self.test_lc = np.concatenate(self.test_lc, axis=0)

    # not useful any more
    # def trainValidateSplit(self, ratio, rs):

    #     s, e = 0, self.train_dataLList[0]
    #     X_train, X_test, y_train, y_test = [], [], [], []
    #     for i in range(len(self.train_dataLList)):         
    #         X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(self.train_ir[s:e, :],self.train_lc[s:e, :], 
    #                                                 test_size=ratio, random_state=rs, shuffle=True)
    #         X_train.append(X_train_t)
    #         X_test.append(X_test_t)
    #         y_train.append(y_train_t)
    #         y_test.append(y_test_t)

    #         s = e
    #         if i == len(self.train_dataLList) - 1: 
    #             e = -1
    #         else: 
    #             e = e + self.train_dataLList[i+1]

    #     return np.concatenate(X_train), np.concatenate(X_test), np.concatenate(y_train), np.concatenate(y_test)


    def avgModelR2(self, x_, model, sp, n_splits=5, n_repeats=500):

        r2_ = []

        for i in range(n_repeats):
            kf = KFold(n_splits=n_splits, shuffle=True)
            split_indices = kf.split(X=self.raw_ir_data_preprocess)

            for train_index, test_index in split_indices:
                xtrain = x_[train_index]
                ytrain = self.lc_data[train_index, sp]/np.sum(self.lc_data[train_index, :], axis=1)
                xtest = x_[test_index]
                ytest = self.lc_data[test_index, sp]/np.sum(self.lc_data[test_index, :], axis=1)

                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)
                r2_.append(r2_score(ytest, ypred))

        return r2_

    def evaluateModel(self, model_list, name_list, color_list):

        fig, ax = plt.subplots(len(model_list), 1, figsize=(12, 10), sharex=True)

        for i, model in enumerate(model_list):

            r2_hist = self.avgModelR2(model)
            ax[i].hist(r2_hist, bins=80, alpha=0.25, color=color_list[i], label=name_list[i])
            ax[i].axvline(x=np.mean(r2_hist), ls='--', c=color_list[i], label=f'r2_avg={np.mean(r2_hist)}')
            ax[i].legend()
        ax[0].set_xlim(0, )

        plt.show()