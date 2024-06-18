'''
18/06/2024
This Code was written by Rawan S. Abdulsadig (https://github.com/RawanAbdulsadig), corresponding to the work presented in the paper: 'A Novel Computational Signal Processing Framework Towards Multimodal Vital Signs Extraction Using Neck-Worn Wearable Devices'
Please cite the paper if you found any of the code below useful, thanks.
'''

import numpy as np
import pandas as pd
import scipy

def Standardise(sig):
    return (sig - np.mean(sig))/np.std(sig)

def Normalise(sig):
    return (sig - np.min(sig))/(np.max(sig) - np.min(sig))

def getDominanceScores(segment, DS_df, thresh, level, low_fftfreq, high_fftfreq):

    Obs_Rates, Prob_Est = getProbaEstimations(segment, low_fftfreq, high_fftfreq)
    rate_estimations_DF = pd.DataFrame({'Rate':np.around(Obs_Rates[Prob_Est >= thresh]),
                                        'p':  Prob_Est[Prob_Est >= thresh] / level})
    DS_df = pd.concat([DS_df, rate_estimations_DF], axis=0, ignore_index=True)

    sub_segments, noisy = get_cleanChuncks(segment, gab = 1, min_chunckLen = 5*100)
#     print(sub_segments)
    if (len(sub_segments) > 0) & noisy:
        for sub_segment in sub_segments:

            if (level+1 <= 20) :
                sub_segment = Standardise(sub_segment)
                DS_df = getDominanceScores(sub_segment, DS_df, thresh, level+1, low_fftfreq, high_fftfreq)
    return DS_df

def applyDoublePeakAdjustment(df, last_confRate, slack):
    try:
        if np.percentile(df.Rate , 75) > np.percentile(df.Rate , 50)*2 - slack*3:
            df.loc[df.Rate >= np.percentile(df.Rate , 75),'Rate'] = df[df.Rate >= np.percentile(df.Rate , 75)].Rate/2
            print('peaks adjusted')
        elif (np.percentile(df.Rate , 0) > last_confRate*2 - slack*3) & (not np.isnan(last_confRate)):
            df.loc[df.Rate >= np.percentile(df.Rate , 0),'Rate'] = df[df.Rate >= np.percentile(df.Rate , 0)].Rate/2
            print('peaks adjusted')
    except:
        pass
    return df

def getProbaEstimations(segment, low_fftfreq, high_fftfreq):
    fft_f = np.fft.fftfreq(segment.shape[0], d=1/100)
    pfft_freqs = fft_f[(fft_f>low_fftfreq) & (fft_f<=high_fftfreq)]

    prr = np.fft.fft(segment) / len(segment)
    pfft = np.abs(prr[(fft_f>low_fftfreq) & (fft_f<=high_fftfreq)])

    pfft_probs = scipy.special.softmax(Standardise(pfft))

    if pfft_probs[0] == pfft_probs.max():
        pfft_probs = np.delete(pfft_probs, 0)
        pfft_freqs = np.delete(pfft_freqs, 0)

    return pfft_freqs*60, pfft_probs

def get_cleanChuncks(r_val, gab = 1, min_chunckLen = 5*100):
    outlier_reigons = (r_val>3) | (r_val < -3)*1
    Acceptable_reigons = (r_val<=3) & (r_val >= -3)*1
    indexes = np.arange(len(r_val))
    noisy = sum(outlier_reigons) != 0
    Cleaning_DataFrame = pd.DataFrame({'indexes': indexes, 'Acceptable_reigons': Acceptable_reigons})
    if Cleaning_DataFrame.Acceptable_reigons.values[0] == 1:
        Cleaning_DataFrame.loc[:,'Acceptable_reigons'].values[0] = 0
    if Cleaning_DataFrame.Acceptable_reigons.values[-1] == 1:
        Cleaning_DataFrame.loc[:,'Acceptable_reigons'].values[-1] = 0

    Cleaning_DataFrame['ChunckChange'] = np.nan
    Cleaning_DataFrame.loc[1:,'ChunckChange']= Cleaning_DataFrame.Acceptable_reigons[1:].values - Cleaning_DataFrame.Acceptable_reigons[0:-1].values
    ChunckStartIndex = Cleaning_DataFrame[Cleaning_DataFrame['ChunckChange'].values > 0].indexes.values
    ChunckEndIndex = Cleaning_DataFrame[Cleaning_DataFrame['ChunckChange'].values < 0].indexes.values
    Chunck_StartEnd_Index = []
    for st,en in zip(ChunckStartIndex, ChunckEndIndex):
        Chunck_StartEnd_Index.append((st,en))

    r_val_chuncks = []
    for (chunck_start, chunck_end) in Chunck_StartEnd_Index:
        if chunck_start == 1:
            if chunck_end >= len(r_val)-3:
                 r_val_chunck = r_val[int(100*gab/2):-int(100*gab/2)]
            else:
                 r_val_chunck = r_val[int(100*gab/2):chunck_end-int(100*gab)]
        else:
            if chunck_end >= len(r_val)-3:
                r_val_chunck = r_val[chunck_start+int(100*gab):]
            else:
                r_val_chunck = r_val[chunck_start+int(100*gab):chunck_end-int(100*gab)]

        if len(r_val_chunck) >= min_chunckLen:
            r_val_chuncks.append(r_val_chunck)
        else:
            noisy = True
    return r_val_chuncks, noisy

def getRateEstimationBand(Processed_segments, EWMA, EWMA_Estimation, EWMA_ConfEstimation, lastConfRate, Rate_Range, slack, task, mu , std, low_fftfreq, high_fftfreq, alpha=0.3, thresh = 0.1, low_percentile=25, high_percentile=75):
    rl = np.floor(slack/2)
    ru = np.ceil(slack/2)

    DF_df = pd.DataFrame([])
    level = 1
    for Processed_segment in Processed_segments:
        DF_df_ = pd.DataFrame([])
        DF_df_ = getDominanceScores(Processed_segment, DF_df_, thresh, level, low_fftfreq, high_fftfreq)
        DF_df = pd.concat([DF_df, DF_df_], axis=0, ignore_index=True)
#     print(DF_df)
    if task == 'PR':
        DF_df = applyDoublePeakAdjustment(DF_df, lastConfRate, slack)

    DF_df = DF_df.groupby(['Rate']).sum()

    DF_df['p_band'] = DF_df.p
    for r in list(DF_df.index):
        DF_df.loc[r , 'p_band'] = DF_df[(DF_df.index >= r-rl) & (DF_df.index <= r+ru)].p.sum()

    EWMA['newProbs'] = 0
    EWMA.loc[DF_df.index.values.astype('int'),'newProbs'] = DF_df.p_band.values
    if sum(EWMA.newProbs)>0:
        EWMA.ProbDist = Normalise(EWMA.ProbDist)*(1-alpha) + Normalise(EWMA.newProbs)*alpha
    else:
        EWMA.ProbDist = Normalise(EWMA.ProbDist)*(1-alpha)

    try:
        highest_probs = EWMA[EWMA.ProbDist > thresh]
        perc_low  = np.percentile(highest_probs.index, low_percentile)
        perc_high = np.percentile(highest_probs.index, high_percentile)
        median = np.percentile(highest_probs.index,50)

        if ((median - perc_low) <= slack) & ((perc_high - median) <= slack):
            confident = True
        else:
            confident = False

        if not np.isnan(EWMA_Estimation[-1]):
            EWMA_Estimation.append(highest_probs.index[np.argmin(abs(highest_probs.index - EWMA_Estimation[-1]).values)])
        else:
            EWMA_Estimation.append(highest_probs[(highest_probs.ProbDist.values == highest_probs.ProbDist.max())].index[0])

    except:
            confident = False
            EWMA_Estimation.append(np.nan)

    if confident:
        EWMA_ConfEstimation.append(EWMA_Estimation[-1])
        last_confRate = EWMA_Estimation[-1]
    else: EWMA_ConfEstimation.append(np.nan)

    return EWMA_Estimation, EWMA_ConfEstimation, lastConfRate, EWMA
