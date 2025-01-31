#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karajohnson
"""


import os
import numpy as np
from scipy import signal
from fooof import FOOOF, utils
from scipy.stats import sem, pearsonr, ttest_ind, fisher_exact
import math
import pandas as pd
import copy
pd.options.display.max_columns = None
pd.options.display.max_rows = None

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams.update({'text.usetex':False})
plt.rcParams.update({"svg.fonttype":'none'})
plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Arial'})
import json

import warnings
warnings.filterwarnings('ignore')

import GraphingUtility
import SignalProcessingUtility

# =============================================================================
# %% Load clinical data spreadsheet + clean data
# =============================================================================

workDir = "/Users/karajohnson/UFL Dropbox/Kara Johnson/Postdoc/Projects/Intraop/Analyses/RestLFP/PD_GPi_STN/Depression"
outDir = os.path.join(workDir,"Processed")

if not os.path.exists(outDir):
    os.makedirs(outDir)
  
Scores = pd.read_excel(os.path.join(workDir,"CohortData","Neurophysiology of depression in PD - patient data updated.xlsx"),dtype={"MRN":"str"},
                       na_values=['',' '],keep_default_na=False).dropna(subset=["INFORM_ID"])
Scores["Hemi"] = np.zeros(len(Scores))
Scores["Hemi"].loc[Scores["Side 1 surg"]=="L"] = "Left"
Scores["Hemi"].loc[Scores["Side 1 surg"]=="R"] = "Right"
Scores["Target"].loc[Scores["Target"]=="GPI"] = "GPi"
Scores["INFORM_ID"] = Scores["INFORM_ID"].astype(np.int64)

renameCol = {"UPDRS OFF":"UPDRS_OFF", 
             "UPDRS ON":"UPDRS_ON",
             "Levodopa dose":"LEDD",
             "Anti-depressant":"Antidepressant", 
             "Benzodiazepine ":"Benzodiazepine",
             "Non-benzodiazepine hypnotic ":"NonBenzo",
             "MOA antidepressant":"AntidepressantType",
             "Duration of disease":"DiseaseDuration",
             "time between eval and recording":"EvalTimePoint"}
Scores = Scores.rename(columns=renameCol)

# =============================================================================
# %% Load LFP data + merge with clinical data + clean data
# =============================================================================

score = "BDI"

LFP = pd.read_csv(os.path.join(workDir,"PD_RestLFPNormDatabase_30s_20240325_FOOOF_HiNormExp2_AutoNoiseDetect_AddedMax.csv"),na_values=['',' '],keep_default_na=False)

LFP["SurgDescription"] = LFP["Hemi"].str[0]+" "+LFP["Target"]
LFP.rename(columns={"INFORM_ID":"idPatient"},inplace=True)
Scores.rename(columns={"INFORM_ID":"idPatient"},inplace=True)
LFP["idPatient"] = LFP["idPatient"].astype(np.int64)
Scores["idPatient"] = Scores["idPatient"].astype(np.int64)

Data = pd.merge(Scores[[x for x in Scores.columns if not "Target" in x]],LFP[LFP["Dx"] == "PD"],how="outer",on=["idPatient","Hemi"]).dropna(subset=["PtID"]) #[[x for x in Scores.columns if not "Target" in x]]

Data["Depression"] = np.zeros(len(Data))
Data["Depression"].loc[Data["BDI"]>=14] = 1
Data["Depression"].loc[pd.isna(Data["BDI"])] = np.nan

Data["Anxiety"] = np.zeros(len(Data))
Data["Anxiety"].loc[Data["STAITrait"]>=40] = 1
Data["Anxiety"].loc[pd.isna(Data["STAITrait"])] = np.nan

Data["Apathy"] = np.zeros(len(Data))
Data["Apathy"].loc[Data["AS"]>=14] = 1
Data["Apathy"].loc[pd.isna(Data["AS"])] = np.nan

Data["NumContacts"] = Data.groupby(["PtID"])["PtID"].transform("count")

print("Number of hemis with baseline scores and LFP:",len(Data.dropna(subset=[score]).drop_duplicates(subset=["PtID"])))
print("Number of patients with baseline scores and LFP:",len(Data.dropna(subset=[score]).drop_duplicates(subset=["idPatient"])))

### Average power over segments
C0 = (Data["Contact"]=="Contact_0")
C1q = (Data["NumContacts"]==4)&((Data["Contact"]=="Contact_1"))
C2q = (Data["NumContacts"]==4)&((Data["Contact"]=="Contact_2"))
C3q = (Data["NumContacts"]==4)&((Data["Contact"]=="Contact_3"))
C1d = (Data["NumContacts"]==8)&((Data["Contact"]=="Contact_1")|(Data["Contact"]=="Contact_2")|(Data["Contact"]=="Contact_3"))
C2d = (Data["NumContacts"]==8)&((Data["Contact"]=="Contact_4")|(Data["Contact"]=="Contact_5")|(Data["Contact"]=="Contact_6"))
C3d = (Data["NumContacts"]==8)&(Data["Contact"]=="Contact_7")

Data.loc[C0,"Ring"] = "Contact_0"
Data.loc[C1d,"Ring"] = "Contact_1"
Data.loc[C2d,"Ring"] = "Contact_2"
Data.loc[C3d,"Ring"] = "Contact_3"
Data.loc[C1q,"Ring"] = "Contact_1"
Data.loc[C2q,"Ring"] = "Contact_2"
Data.loc[C3q,"Ring"] = "Contact_3"

bands = [x for x in Data.columns if "Delta" in x or "Theta" in x or "Alpha" in x or "Beta" in x or "Gamma" in x or "HFO" in x]
for b in bands:
    Data.loc[(Data["Ring"]=="Contact_0"),b+"Ring"] = Data[Data["Ring"]=="Contact_0"].groupby("PtID")[b].transform('mean')
    Data.loc[(Data["Ring"]=="Contact_1"),b+"Ring"] = Data[Data["Ring"]=="Contact_1"].groupby("PtID")[b].transform('mean')
    Data.loc[(Data["Ring"]=="Contact_2"),b+"Ring"] = Data[Data["Ring"]=="Contact_2"].groupby("PtID")[b].transform('mean')
    Data.loc[(Data["Ring"]=="Contact_3"),b+"Ring"] = Data[Data["Ring"]=="Contact_3"].groupby("PtID")[b].transform('mean')

# =============================================================================
# %% Write source data - clinical vars + band powers
# =============================================================================

bands = ["DeltaNormRing","ThetaNormRing","AlphaNormRing","BetaNormRing",
         "LowBetaNormRing","HighBetaNormRing","GammaNormRing","LowGammaNormRing","HighGammaNormRing",
         "HFONormRing","FOOOFExp","FOOOFOffset"]
subDF = copy.deepcopy(Data[Data["Target"]=="GPi"].dropna(subset=["BDI"]))
for band in bands:
    subDF[band+"Mean"] = subDF.groupby(["PtID"])[band].transform("mean")

outDF = subDF.drop_duplicates(subset=["PtID"]).sort_values(by=["idPatient"])
outbands = [b+"Mean"  for b in bands]
outDF["PtNum"] = np.arange(1,len(outDF)+1)

outcols = ["idPatient","PtNum","Hemi","Sex","Age","EvalTimePoint",
           "BDI","STAITrait","AS",
           "UPDRS_OFF","UPDRS_ON","DiseaseDuration","LEDD",
           "Antidepressant","Benzodiazepine","NonBenzo"]+outbands

print(outDF[outcols].head(),"\n")
print("N=",len(outDF))

# outDF.to_excel(os.path.join(outDir,"PDDepression_SourceData_BandPowers_INFORM.xlsx"),columns=outcols,index=False,na_rep="NaN")

# =============================================================================
# %% Test for differences in clinical variables x depression
# =============================================================================

target = "GPi"
procDF = Data[Data["Target"]==target].dropna(subset=["BDI"]).drop_duplicates(subset=["idPatient"])

## t-test for continuous vars
clinVars = ["Age",
            "DiseaseDuration",
            "EvalTimePoint",
            "BDI","AS","STAITrait",
            "UPDRS_OFF","UPDRS_ON",
            "LEDD"]
for v in clinVars:
    print(v)
    print(procDF.groupby("Depression")[v].describe(),"\n")
    t,p = ttest_ind(procDF[procDF["Depression"]==1].dropna(subset=[v])[v],
                    procDF[procDF["Depression"]==0].dropna(subset=[v])[v])
    print(t,p,"\n")
    
## Fisher exact test for discrete vars
catVars = ["Sex",
           "Hemi",
           "Antidepressant",
           "Benzodiazepine",
           "NonBenzo"]
for c in catVars:
    print(c)
    print(procDF.groupby("Depression")[c].value_counts(),"\n")
    table = sm.stats.Table.from_data(procDF[[c,"Depression"]])
    oddsratio, pvalue = fisher_exact(table.table)
    print(oddsratio,pvalue,"\n")
    
# =============================================================================
# %% Test for differences in band powers x depression
# =============================================================================

target = "GPi"
score = "BDI"
sx = "Depression"
cutoff = 14

plt.close("all")

subDF = Data[Data["Target"]==target].dropna(subset=[score])
subDF[sx] = np.empty(len(subDF))
subDF.loc[subDF[score]>=cutoff,sx] = 1
subDF.loc[subDF[score]<cutoff,sx] = 0

print("Num patients in subDF",len(subDF.drop_duplicates(subset=["idPatient"])))

bands = ["Delta","Theta","Alpha",
         "Beta","LowBeta","HighBeta",
         "Gamma","LowGamma","HighGamma","HFO"]

ps = []
prs = []
ts = []
rs = []

for band in bands:
    
    if not "FOOOF" in band:
        band = band+"NormRing"
    
    ### All contacts
    # procDF = subDF
    
    ### Max contact
    # procDF = subDF.loc[subDF.groupby(['PtID'])[band].idxmax()][subDF.columns]
    
    ### Mean across all contacts
    subDF[band+"Mean"] = subDF.groupby(["PtID"])[band].transform("mean")
    procDF = subDF.drop_duplicates(subset=["PtID"])
    band = band+"Mean"   
    
    ### Depressed vs not depressed
    t,p = ttest_ind(procDF[procDF[score]>=cutoff][band],
                    procDF[procDF[score]<cutoff][band])
    
    print(len(procDF[procDF[score]>=cutoff][band]),"vs",len(procDF[procDF[score]<cutoff][band]))
    
    print(band,t,p)
    ps.append(p)
    ts.append(t)

    r,pr = pearsonr(procDF[band],procDF[score])
    print(band,score,r,pr,'\n')
    prs.append(pr)
    rs.append(r)

rej,pcorr = fdrcorrection(ps,alpha=0.05,method="indep",is_sorted=False)

rej,prcorr = fdrcorrection(prs,alpha=0.05,method="indep",is_sorted=False)
print("corr",prcorr,rej)

dfdict = {"Variable":[x+"NormRing" for x in bands],
          "T":ts,"RawP_Ttest":ps,"FdrP_Ttest":pcorr,
          "R":rs,"RawP_Corr":prs,"FdrP_Corr":prcorr}
statdf = pd.DataFrame(dfdict)
print(statdf)

# =============================================================================
# %% GLMs
# =============================================================================

target = "GPi"
score = "BDI"

subDF = Data[Data["Target"]==target].dropna(subset=[score])
print("Num patients in subDF",len(subDF.drop_duplicates(subset=["idPatient"])))

### Beta
allbands = ["Delta","Theta","Alpha","Beta","Gamma","HFO",
            "FOOOFExp","FOOOFOffset"]

### Low Beta
# allbands = ["Delta","Theta","Alpha","LowBeta","Gamma","HFO",
#             "FOOOFExp","FOOOFOffset"]

### High Beta
# allbands = ["Delta","Theta","Alpha","HighBeta","Gamma","HFO",
#             "FOOOFExp","FOOOFOffset"]

allbands = [x+"NormRing" for x in allbands if not "FOOOF" in x]+["FOOOFExp","FOOOFOffset"]
for band in allbands:
    subDF[band+"Mean"] = subDF.groupby(["PtID"])[band].transform("mean")
    procDF = subDF.drop_duplicates(subset=["PtID"],keep="first")

# bands = [x+"NormRing" for x in allbands if not "FOOOF" in x]+["FOOOFExp","FOOOFOffset"]
bandVars = [x+"Mean" for x in allbands]

### Center continuous vars
ctrVars = ["EvalTimePoint","UPDRS_OFF","UPDRS_ON","DiseaseDuration","Age","LEDD","BDI","STAITrait","AS"]+bandVars
for c in ctrVars:
    procDF[c+"Ctr"] = (procDF[c]-procDF[c].mean())/procDF[c].std()

bandVarsCtr = [x+"Ctr" for x in bandVars]
bandsf = " + ".join(bandVarsCtr)
clinVars = "EvalTimePointCtr + C(Sex) + C(Hemi) + UPDRS_OFFCtr + UPDRS_ONCtr + AgeCtr + DiseaseDurationCtr + LEDDCtr + C(Antidepressant) + C(Benzodiazepine) + C(NonBenzo) + STAITraitCtr + ASCtr"

md = smf.glm(score+" ~ "+clinVars+" +  "+bandsf, procDF,missing="drop")
mdf = md.fit()
print(mdf.summary())
print("AIC",mdf.aic)

#%% Coefficient figure

fig = GraphingUtility.largeFigure(0, resolution=[900,1200])
ax = fig.subplots(1, 1)

html = mdf.summary().tables[1].as_html()
summaryDF = pd.read_html(html, header=0, index_col=0)[0].reset_index()

order = np.array(["Intercept",
         "C(Sex)[T.M]",
         "AgeCtr",
         "C(Target)[T.STN]",
         "C(Hemi)[T.Right]",
         "EvalTimePointCtr",
         "DiseaseDurationCtr",
         "UPDRS_OFFCtr",
         "UPDRS_ONCtr",
         "LEDDCtr",
         "STAITraitCtr",
         "ASCtr",
         "C(Antidepressant)[T.YES]",
         "C(Benzodiazepine)[T.YES]",
         "C(NonBenzo)[T.YES]",
         "DeltaNormRingMeanCtr",
         "ThetaNormRingMeanCtr",
         "AlphaNormRingMeanCtr",
         "BetaNormRingMeanCtr",
         "GammaNormRingMeanCtr",
         "HFONormRingMeanCtr",
         "FOOOFExpMeanCtr",
         "FOOOFOffsetMeanCtr"])

ordernums = [0]*len(summaryDF)
summaryDF["order"] = ordernums
for i,var in enumerate(order):
    summaryDF.loc[summaryDF["index"]==var,"order"] = np.abs(21-i)

summaryDF.sort_values("order",inplace=True)
y = summaryDF["index"].tolist()
x = summaryDF["coef"].tolist()
yerrs = []

for val in summaryDF["index"]:
    coef = summaryDF[summaryDF["index"]==val]["coef"].values[0]
    lower = summaryDF[summaryDF["index"]==val]["[0.025"].values[0]
    upper = summaryDF[summaryDF["index"]==val]["0.975]"].values[0]

    yerrs.append((np.abs(coef-lower),np.abs(coef-upper)))
    print(val,lower,upper)

yerr = np.array(yerrs).T
# yerr = [summaryDF["[0.025"].values,summaryDF["0.975]"].values]

plt.errorbar(x, y, xerr=yerr, capsize=0, fmt="ko", ecolor = "black")
plt.axvline(x=0,color="k",linestyle="--")
ax.set_xlim(-10,16)

# fig.savefig(os.path.join(outDir, "GPi_GLM_Coeff.svg"), dpi=300)

# =============================================================================
#%% Processing methods figure
# =============================================================================

exPt = "30_GPi_R.json"

ptFi = os.path.join("/Users/karajohnson/UFL Dropbox/Kara Johnson/Postdoc/Projects/Intraop/Analyses/RestLFP/PD_GPi_STN/LFP_Spectrograms/",exPt)
with open(ptFi,"r") as file:
    ptDB = json.load(file)
    
#%% Plot raw and filtered LFP

fs = 22000
time = np.arange(len(ptDB["RawLFP"]["Contact_0"]["LFP"]))/fs

fig = GraphingUtility.largeFigure(0, resolution=[600,300])
ax = fig.subplots(1,2,sharex=True,sharey=True)

yshift = 0
for contact in ptDB["RawLFP"]:
    LFP = np.array(ptDB["RawLFP"][contact]["LFP"])
    ax[0].plot(time,LFP+yshift)
    LFP = np.array(ptDB["FiltLFP"][contact]["LFP"])
    ax[1].plot(time,LFP+yshift)

    ax[0].set_xlim(0,30)
    yshift += 400
    
# fig.savefig(os.path.join(outDir,"Examples","Methods_"+exPt.split(".")[0]+"_LFP.svg"), dpi=300)

#%% Plot raw PSDs

freq = np.arange(0,500,0.5)

fig = GraphingUtility.largeFigure(0, resolution=[600,300])
ax = fig.subplots(1)

fig2 = GraphingUtility.largeFigure(0, resolution=[600,300])
ax2 = fig2.subplots(1)

for contact in ptDB["RawLFP"]:
    
    freqEnd = 500
    LFP = np.array(ptDB["RawLFP"][contact]["LFP"])
    freq, PSD = signal.welch(LFP, 
                              fs=fs,
                              nperseg=fs * 1,
                              noverlap=fs * 0.5,
                              nfft=fs * 2, scaling="density") 
    ax.plot(freq[freq<=freqEnd],np.log10(PSD[freq<=freqEnd]))
    ax.set_xlim(0,freqEnd)
    ax.set_ylim(-2.75,2.75)
    
    freqEnd = 100
    LFP = np.array(ptDB["RawLFP"][contact]["LFP"])
    freq, PSD = signal.welch(LFP, 
                              fs=fs,
                              nperseg=fs * 1,
                              noverlap=fs * 0.5,
                              nfft=fs * 2, scaling="density") 
    ax2.plot(freq[freq<=freqEnd],np.log10(PSD[freq<=freqEnd]))
    ax2.set_xlim(0,freqEnd)
    ax2.set_ylim(-2.75,2.75)
    
# fig.savefig(os.path.join(outDir,"Examples","Methods_"+exPt.split(".")[0]+"_RawPSD500.svg"), dpi=300)
# fig2.savefig(os.path.join(outDir,"Examples","Methods_"+exPt.split(".")[0]+"_RawPSD100.svg"), dpi=300)

#%% Normalize / plot PSD with norm curves / plot normPSD

plt.close("all")

lowfreqNorm = (freq<=100)
HiFitFreq = (freq>100) & (freq<=500)

fig = GraphingUtility.largeFigure(0, resolution=[400,600])
ax = fig.subplots(4,1,sharex=True,sharey=True)

fig0 = GraphingUtility.largeFigure(0, resolution=[400,600])
ax0 = fig0.subplots(4,1,sharex=True,sharey=True)

fig1 = GraphingUtility.largeFigure(0, resolution=[600,300])
ax1 = fig1.subplots(1,sharex=True,sharey=True)

fig2 = GraphingUtility.largeFigure(0, resolution=[600,300])
ax2 = fig2.subplots(1,sharex=True,sharey=True)

PSDs = []
for i,contact in enumerate(ptDB["FiltLFP"]):

    LFP = np.array(ptDB["FiltLFP"][contact]["LFP"])
    freq, PSD = signal.welch(LFP, 
                              fs=fs,
                              nperseg=fs * 1,
                              noverlap=fs * 0.5,
                              nfft=fs * 2, scaling="density") 
    
                                        
    model = FOOOF(min_peak_height=0.05, verbose=False)
    model.fit(freq[lowfreqNorm],PSD[lowfreqNorm])
    [offset,exp] = model.aperiodic_params_
    lowFitNorm = offset - np.log10(np.power(freq[lowfreqNorm],exp))
    
    hinormFit = SignalProcessingUtility.fittedNormalization(np.log10(PSD),freq,HiFitFreq,order=2)
    
    normFit = np.array(list(lowFitNorm) + list(hinormFit[freq>100]))
    
    lownormPSD = np.log10(PSD[lowfreqNorm])-lowFitNorm
    hinormPSD = np.log10(PSD[freq>100])-hinormFit[freq>100]
    normPSD = np.array(list(lownormPSD) + list(hinormPSD))
    
    peaks = ptDB["Params"]["NotchFreqPeaks"]
    widths = ptDB["Params"]["NotchFreqWidths"]
    for j,p in enumerate(peaks):
        w = math.ceil(widths[j]/2)+1
        PSD[(freq>=p-w)&(freq<=p+w)] = np.nan
        normPSD[(freq>=p-w)&(freq<=p+w)] = np.nan
        
    PSDs.append(normPSD)
    freqEnd = 500
    ax[i].plot(freq[freq<=freqEnd],np.log10(PSD)[freq<=freqEnd])
    ax[i].plot(freq[freq<=freqEnd],normFit[freq<=freqEnd])
    ax[i].set_xlim(0,freqEnd)
    ax[i].set_ylim(-2.75,2.75)

    ax1.plot(freq[freq<=freqEnd],normPSD[freq<=freqEnd])
    ax1.set_xlim(0,freqEnd)
    ax1.set_ylim(-0.5,2.5)
    
    freqEnd = 100
    
    ax0[i].plot(freq[freq<=freqEnd],np.log10(PSD)[freq<=freqEnd])
    ax0[i].plot(freq[freq<=freqEnd],normFit[freq<=freqEnd])
    ax0[i].set_xlim(0,freqEnd)
    ax0[i].set_ylim(-2.75,2.75)
    
    ax2.plot(freq[freq<=freqEnd],normPSD[freq<=freqEnd])
    ax2.set_xlim(0,freqEnd)
    ax2.set_ylim(-0.5,2.5)
    
freqEnd = 500
avgnormPSD = np.nanmean(PSDs,axis=0)
ax1.plot(freq[freq<=freqEnd],avgnormPSD[freq<=freqEnd],color="k",linewidth=2)

freqEnd = 100
ax2.plot(freq[freq<=freqEnd],avgnormPSD[freq<=freqEnd],color="k",linewidth=2)

# fig.savefig(os.path.join(outDir,"Examples","Methods_"+exPt.split(".")[0]+"_NormPSDFit500.svg"), dpi=300)
# fig0.savefig(os.path.join(outDir,"Examples","Methods_"+exPt.split(".")[0]+"_NormPSDFit100.svg"), dpi=300)
# fig1.savefig(os.path.join(outDir,"Examples","Methods_"+exPt.split(".")[0]+"_AvgNormPSD500.svg"), dpi=300)
# fig2.savefig(os.path.join(outDir,"Examples","Methods_"+exPt.split(".")[0]+"_AvgNormPSD100.svg"), dpi=300)

# =============================================================================
# %% Full PSD x depression figure
# =============================================================================

restDBFi = os.path.join(workDir,"PD_RestLFPNormDatabase_30s_20231004_FOOOF_HiNormExp2_AutoNoiseDetect.json")
with open(restDBFi, "r") as file:
    DB = json.load(file)

#%%

oc = "BDI"
cutoff = 14

freqEnd = 500
    
ptList = [x for x in DB.keys() if not "PSDFreq" in x and len(DB[x]) > 0]
INFORM_IDS = [DB[x]["INFORM_ID"] for x in ptList]

GPi = []
GPi_D = []
ID_GPi_D = []
GPi_ND = []
ID_GPi_ND = []

GPiOCs = []
GPiPDs = []
IDs = []
outArray = []

for exPt in ptList:
        
    inFisplt = list(DB[exPt].keys())[1]
    ptDB = DB[exPt][inFisplt]["RestLFP"]#[contact]["NormPSD"]
    freq = np.array(DB["PSDFreq"])
    # freq = freq[freq<=freqEnd]
    ID = DB[exPt]["INFORM_ID"]
    hemi = DB[exPt][inFisplt]["Hemi"]
    
    if ID in list(Data["idPatient"].astype(int).values) and Scores[Scores["idPatient"]==ID]["Hemi"].values == hemi and ID in list(Scores["idPatient"].values) and not pd.isna(Scores[(Scores["idPatient"]==ID)&(Scores["Hemi"]==hemi)][oc].values[0]):
        ocscore = Scores[Scores["idPatient"]==ID][oc].values[0]
        UPDRS = Scores[Scores["idPatient"]==ID]["UPDRS_OFF"].values[0]
        
        conPSDs = []
        for contact in ptDB.keys():
            if "Contact" in contact:
                ptDB[contact]["NormPSD"][0] = np.nan
                PSD = np.array(ptDB[contact]["NormPSD"])
                conPSDs.append(PSD)
        
        PSD = np.nanmean(conPSDs,axis=0)
                
        if "GPi" in exPt:
            GPi.append(PSD)
            IDs.append(ID)
            outArray.append([ID,ocscore,PSD[freq<=freqEnd].astype(np.float64)])
            GPiOCs.append(ocscore)
            GPiPDs.append(UPDRS)

            if ocscore >= cutoff:
                GPi_D.append(PSD[freq<=freqEnd])
                if not ID in ID_GPi_D:
                    ID_GPi_D.append(ID)
                    
            if ocscore < cutoff:
                GPi_ND.append(PSD[freq<=freqEnd])
                if not ID in ID_GPi_ND:
                    ID_GPi_ND.append(ID)
             
GPi_Dm = np.nanmean(GPi_D,axis=0)
GPi_Dsem = sem(GPi_D,nan_policy="omit",axis=0)
GPi_Dsd = np.nanstd(GPi_D,axis=0)
print("NumPt GPi D "+oc+">="+str(cutoff),len(ID_GPi_D))
print("NumCons GPi D "+oc+">="+str(cutoff),len(GPi_D))

GPi_NDm = np.nanmean(GPi_ND,axis=0)
GPi_NDsem = sem(GPi_ND,nan_policy="omit",axis=0)
GPi_NDsd = np.nanstd(GPi_ND,axis=0)

print("NumPt GPi ND "+oc+"<"+str(cutoff),len(ID_GPi_ND))
print("NumCons GPi "+oc+"<"+str(cutoff),len(GPi_ND))

# Transpose for stats
GPit = np.transpose(GPi)
GPi_Dt = np.transpose(GPi_D)
GPi_NDt = np.transpose(GPi_ND)

# =============================================================================
# %% Write source data - full normPSDs
# =============================================================================

DF = pd.DataFrame(outArray,columns=["idPatient","BDI","PSD"]).sort_values(by=["idPatient"])
DF["PtNum"] = np.arange(1,len(outDF)+1)
outcols = ["idPatient","PtNum","BDI","PSD"]

# Define freq for column names
suffix = [str(f) for f in np.arange(0.5,500.5,0.5)]

# Expand the PSD array column into separate columns
DF_expanded = pd.DataFrame(DF["PSD"].to_list(),index=DF.index)
DF_expanded.columns = ["NormPSD_"+suffix[i]+"Hz" for i,c in enumerate(DF_expanded.columns)]
outDF = pd.concat([DF[outcols],DF_expanded], axis=1)

# Remove the original array column
outDF = outDF.drop('PSD', axis=1)

# Display the result
# print(outDF)

print(outDF.head(),"\n")
print("N=",len(outDF))

# outDF.to_excel(os.path.join(outDir,"PDDepression_SourceData_PSD_INFORM.xlsx"),index=False,na_rep='NaN')

#%% Compare PSD x depression

freqEnd = 50

fig = GraphingUtility.largeFigure(0, resolution=[600,300])
ax = fig.subplots(1, 1)
DColor = "red" # Depressed
NDColor = "blue" # Not depressed

# sigf = []
# ps = []
# for f in range(len(freq[freq<=freqEnd])):
    
    # Gt,Gp = ttest_ind(GPi_Dt[f],GPi_NDt[f])
    # Gt,Gp = f_oneway(GPi_Dt[f],GPi_NDt[f],GPi_NDnt[f])
    # ps.append(Gp)

# rej,pcorr = fdrcorrection(ps[1:],alpha=0.05,method='n',is_sorted=False)
# for f,val in enumerate(pcorr):
    
    # if Gp < 0.05/len(freq[freq<=freqEnd]):
    #     print("GPi:",freq[freq<=freqEnd][f],Gt,Gp)
    # sigf.append(freq[freq<=freqEnd][f])
    # if val<0.05:
    #     plt.axvline(x=freq[(freq<=freqEnd)][f])
        
        # R,Rp = pearsonr(GPit[f],GPiOCs)
        # if Rp < 0.05:
        #     print("GPi R:",freq[freq<=freqEnd][f],R,Rp)
        #     GPiPDs = np.array(GPiPDs)
        #     Rp,Rpp = pearsonr(GPit[f][~np.isnan(GPiPDs)],GPiPDs[~np.isnan(GPiPDs)])
        #     # if Rpp < 0.05:
        #     print("GPi R UPDRS:",freq[freq<=freqEnd][f],Rp,Rpp)
    
# print(freq[freq<=freqEnd][rej])
# print(ps[rej])
            
plt.plot(freq[freq<=freqEnd],GPi_Dm,color=DColor,label="Depressed")    
plt.fill_between(freq[freq<=freqEnd],GPi_Dm-GPi_Dsem,GPi_Dm+GPi_Dsem,alpha=0.3,color=DColor)
plt.plot(freq[freq<=freqEnd],GPi_NDm,color=NDColor,label="Not Depressed")    
plt.fill_between(freq[freq<=freqEnd],GPi_NDm-GPi_NDsem,GPi_NDm+GPi_NDsem,alpha=0.3,color=NDColor) 

ax.set_xlim(0,freqEnd)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylim(-0.5,1.0)
ax.set_yticks([-0.5,0.0,0.5,1.0])
ax.set_ylabel("Normalized PSD")
ax.legend(frameon=False)
sns.despine()

# fig.savefig(os.path.join(outDir, "GPi_Group"+oc+str(cutoff)+"_AvgContacts"+str(freqEnd)+".svg"), dpi=300)
       