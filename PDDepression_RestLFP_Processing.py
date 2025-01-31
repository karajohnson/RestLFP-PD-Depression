#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karajohnson
"""

import os
import numpy as np
import scipy.io as sio
from scipy import signal
from fooof import FOOOF
import pandas as pd
import matplotlib.pyplot as plt
import math

import seaborn as sns
import json

import Neuromega
import GraphingUtility
import SignalProcessingUtility
import decodeMPX

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# =============================================================================
# %% Define patient parameters / construct initial dict
# =============================================================================

workDir = "/Users/karajohnson/UFL Dropbox/Kara Johnson/Postdoc/Projects/Intraop"
outDir = os.path.join(workDir,"Analyses","RestLFP","PD_GPi_STN")
figDir = os.path.join(outDir,"PSD")

overWrite = True
newCohort = True

fiDF = pd.read_excel(os.path.join(workDir, "Data_Processing", "Intraop_File_Notes.xlsx"),
                 keep_default_na=False)

allptDF = pd.read_excel(os.path.join(workDir,"Intraop_Patients","Intraop_Patients.xlsx"),keep_default_na=False)

prevDBFi = os.path.join(outDir,"PD_RestLFPNormDatabase_30s_20240325_FOOOF_HiNormExp2_AutoNoiseDetect_AddedMax.json")

if os.path.exists(prevDBFi) and newCohort == True:
    with open(prevDBFi, "r") as file:
        prevDB = json.load(file)
        
jsonOutFi = os.path.join(outDir,"PD_RestLFPNormDatabase_30s_20240612_FOOOF_HiNormExp2_AutoNoiseDetect_MeanMax.json")
csvOutFi = jsonOutFi.split(".")[0]+".csv"

rawOutDir = os.path.join(outDir,"LFP_Spectrograms")
if not os.path.exists(rawOutDir):
    os.makedirs(rawOutDir)

noisePts = [] # Omit list for deidentification

if os.path.exists(jsonOutFi) and os.path.exists(csvOutFi) and overWrite == False:
    with open(jsonOutFi, "r") as file:
        DB = json.load(file) 
    csvDB = pd.read_csv(csvOutFi)

if not os.path.exists(jsonOutFi) or overWrite == True:
    DB = {}
    csvDB = {}

DB["PSDFreq"] = []
sumVars = []
spectDB = {}

# =============================================================================
# %% Process each patient recording and save band power + PSD
# =============================================================================

for i,ptDir in enumerate(fiDF["PatientFolder"]):
    
    if not "Dystonia" in ptDir and not "_ET" in ptDir and not "VIM" in ptDir and not "ALIC" in ptDir and not "_HD" in ptDir and not "Alz" in ptDir and not "_TS" in ptDir:
        inFi = fiDF["File"][i]
        procDir = os.path.join(workDir, "Recordings", ptDir)
    
        if fiDF["RestLFP_Plots"][i] != "" and os.path.exists(os.path.join(procDir, inFi)) and ptDir not in DB and "_".join(ptDir.split("_")[0:3]) not in DB and "_".join(ptDir.split("_")[0:3]) not in prevDB:# and "_".join(ptDir.split("_")[0:3]) not in csvDB["PtID"]: #and fiDF["INFORM_ID"][i] != ""
        
            print(ptDir)
        
            target = ptDir.split("_")[2]
            
            if ptDir.split("_")[-1] == "L" or ptDir.split("_")[-2] == "L":
                hemi = "Left"
            if ptDir.split("_")[-1] == "R" or ptDir.split("_")[-2] == "R":
                hemi = "Right"
            
            dx = "PD"
            ptID = "_".join(ptDir.split("_")[0:3])
                
            inFisplt = inFi.split(".")[:-1][0]
                            
            if not ptID in DB:
                DB[ptID] = {}
            
            if inFi.endswith(".mat"):
                inMat = sio.loadmat(os.path.join(workDir, "Recordings", ptDir, inFi))
                Data = Neuromega.extractNeuromegaMat(inMat, target, hemi)
            if inFi.endswith(".mpx"):
                MPX = decodeMPX.decodeMPX(os.path.join(workDir, "Recordings", ptDir, inFi),StreamToParse=[decodeMPX.parseCommandModuleStimulus])
                Data = Neuromega.extractNeuromegaMPX(MPX,target,hemi,analogStim=False)
            
            informID = fiDF["INFORM_ID"][i]
            DB[ptID]["INFORM_ID"] = informID
            DB[ptID][inFisplt] = {}
            DB[ptID][inFisplt]["Target"] = target
            DB[ptID][inFisplt]["Hemi"] = hemi
            DB[ptID][inFisplt]["Dx"] = dx
            DB[ptID][inFisplt]["RestLFP"] = {}
            DB[ptID][inFisplt]["RestLFP"]["filtLFP"] = {}
            
            print(ptID, informID, inFisplt, target, hemi, dx)
            
            ## Make a DB for raw/filt LFP for each hemi
            ptDB = {}
            ptDB["RawLFP"] = {}
            ptDB["FiltLFP"] = {}
            ptDB["Params"] = {}
            ptDB["Params"]["SamplingRate_Hz"] = Data["LFP"]["SamplingRate_Hz"]
            
            # Verify number of contacts + Butterworth filter
            numContacts = 0
            for contact in Data["LFP"].keys():
                if "Contact_" in contact:
                    numContacts += 1
                    
                    LFP = Data["LFP"][contact]["rawLFP"]
                    ptDB["RawLFP"][contact] = {}
                    ptDB["FiltLFP"][contact] = {}
                    
                    ## Butterworth filter 5th order, 1-500 Hz
                    sos = signal.butter(5, [1,500], 'bp', fs=Data["LFP"]["SamplingRate_Hz"], output='sos')
                    Data["LFP"][contact]["filtLFP"] = signal.sosfiltfilt(sos, LFP)

            ptDB["Params"]["BandPass"] = ["Butterworth","Order=5","Band=1-500Hz"]
    
            # =============================================================================
            # Extract rest period
            # =============================================================================
            
            restDetect = fiDF["RestDetection"][i]
            restMask = []
            
            if restDetect == "Auto":
                if "Rest" in inFi or "Baseline" in inFi:
                    print("Auto - rest file")
                    restEnd = Data["LFP"]["Time"][-1]
                    if restEnd >= 30:
                        restEnd = 30
                        
                if not "Rest" in inFi and not "Baseline" in inFi and len(Data["Stim"]["StimMarkers"])>0:
                    print("Auto - StimMarkers")
                    restEnd = Data["Stim"]["StimMarkers"]["Time"][0]-5
                    if restEnd >= 30:
                        restEnd = 30
                        
                if not "Rest" in inFi and not "Baseline" in inFi and len(Data["Stim"]["StimMarkers"])<1:
                    print("Auto - Threshold")
                    threshStim = Data["LFP"]["Time"][Data["LFP"]["Contact_0"]["rawLFP"]>1000]
                    if len(threshStim) < 1:
                        restEnd = Data["LFP"]["Time"][-1]
                    else:
                        restEnd = Data["LFP"]["Time"][Data["LFP"]["Contact_0"]["rawLFP"]>1000][0]-0.5
                    if restEnd >= 30:
                        restEnd = 30
    
                restMask = np.array((Data["LFP"]["Time"] <= restEnd)).flatten()
                    
            if restDetect == "Manual":
                restTime = fiDF["RestTime"][i]
                if type(restTime) != str:
                    print("Manual - single val")
                    restEnd = restTime
                    if restEnd >= 30:
                        restEnd = 30
                    restMask = np.array((Data["LFP"]["Time"] < restEnd)).flatten()
                    
                if type(restTime) == str and ">" in restTime:
                    print("Manual - >")
                    t0 = float(restTime.split(">")[1])
                    restMask = np.array((Data["LFP"]["Time"] >= t0)).flatten()
                    restEnd = Data["LFP"]["Time"][-1]-restEnd
                    if restEnd >= 30:
                        restMask = np.array((Data["LFP"]["Time"] >= t0)&(Data["LFP"]["Time"] <= t0+30)).flatten() 
                        restEnd = 30
                    
                if type(restTime) == str and "-" in restTime:
                    print("Manual - range")
                    t0 = float(restTime.split("-")[0])
                    t1 = float(restTime.split("-")[1])
                    restEnd = t1-t0
                    restMask = np.array((Data["LFP"]["Time"] >= t0)&(Data["LFP"]["Time"] <= t1)).flatten()
                    if restEnd >= 30:
                        restMask = np.array((Data["LFP"]["Time"] >= t0)&(Data["LFP"]["Time"] <= t0+30)).flatten()
                        restEnd = 30
                    
            DB[ptID][inFisplt]["RestLFP"]["RestTime"] = restEnd
            print("Rest period: "+str(restEnd))    
       
            # =============================================================================
            # Detect peaks >40 Hz and apply criteria to identify noise
            # =============================================================================
            
            allconPeaks = []
            allconWidths = {}
            numContacts = 0
            for contact in Data["LFP"].keys():
                if "Contact_" in contact:
                    numContacts += 1

                    LFP = Data["LFP"][contact]["filtLFP"][restMask]
                    
                    freq, rawPSD = signal.welch(LFP, 
                                              fs=Data["LFP"]["SamplingRate_Hz"],
                                              nperseg=Data["LFP"]["SamplingRate_Hz"] * 1,
                                              noverlap=Data["LFP"]["SamplingRate_Hz"] * 0.5,
                                              nfft=Data["LFP"]["SamplingRate_Hz"] * 2, scaling="density") 
                                                            
                    allpeaks,props = signal.find_peaks(np.log10(rawPSD),prominence=0.5,width=0,rel_height=0.75)
                    conpeaks = list(allpeaks[(freq[allpeaks]>40)&(freq[allpeaks]<=500)])
                    conwidths = props["widths"][(freq[allpeaks]>40)&(freq[allpeaks]<=500)]
                    allconPeaks.append(conpeaks)
                    for i,p in enumerate(conpeaks):
                        allconWidths[p] = conwidths[i] * 0.5
                        if ptID in noisePts and p==120:
                            print("Fixing 60 Hz width")
                            allconWidths[p] = conwidths[i] * 0.5 + 2
                            if ptID == "AB":
                                print(ptID,"Extra wide")
                                allconWidths[p] = conwidths[i] * 0.5 + 8
                            if ptID == "CD" or ptID == "EF" or ptID == "GH":
                                print(ptID,"Wide")
                                allconWidths[p] = conwidths[i] * 0.5 + 4
                    # print(freq[conpeaks])
                    # print(conwidths)
           
            setPeaks = np.array(sorted(list(set(sum(allconPeaks,[])))))
            origPeaks = list(setPeaks[np.diff(setPeaks,prepend=0)>1])

            fixPeaks = []
            for con in allconPeaks:
                peaksinrange = []
                for p in origPeaks:
                    for k in con:
                         if k >= p-2 and k <= p+2:
                            peaksinrange.append(p)
                fixPeaks.append([x for x in peaksinrange])# if len(x) > 0])            
                     
            peaks = sorted(list(set(list(set.intersection(*[set(x) for x in fixPeaks]))))) # peak indices
            
            harms = [n*60*2 for n in range(1,9)] # in samples
            for h in harms:
                if not h in peaks:
                    peaks = peaks + [h]
                    allconWidths[h] = 4

            peakFreqs = freq[peaks] # in freq
            widths = [allconWidths[x] for x in peaks] # in freq

            print(peakFreqs)
            # print(widths)
            
            # =============================================================================
            # Compute PSD, exclude noise peaks, normalize            
            # =============================================================================
            
            freqNorm = 500
            
            figLF = GraphingUtility.largeFigure(0, resolution=[1600,800])
            axLF = figLF.subplots(1)

            fig500 = GraphingUtility.largeFigure(0, resolution=[1600,800])
            ax500 = fig500.subplots(1)
            
            figraw = GraphingUtility.largeFigure(0, resolution=[1600,800])
            axraw = figraw.subplots(1)
            
            figrawLF = GraphingUtility.largeFigure(0, resolution=[1600,800])
            axrawLF = figrawLF.subplots(1)
            
            lines = []
            
            for contact in Data["LFP"].keys():
                if "Contact_" in contact and int(contact.split("_")[1]) < numContacts:
                    
                    # DB[ptID][inFisplt]["RestLFP"]["filtLFP"][contact] = Data["LFP"][contact]["filtLFP"][restMask]
                    
                    ### Raw PSD (no filtering)
                    
                    freq, rawPSD = signal.welch(Data["LFP"][contact]["rawLFP"][restMask], 
                                              fs=Data["LFP"]["SamplingRate_Hz"],
                                              nperseg=Data["LFP"]["SamplingRate_Hz"] * 1,
                                              noverlap=Data["LFP"]["SamplingRate_Hz"] * 0.5,
                                              nfft=Data["LFP"]["SamplingRate_Hz"] * 2, scaling="density") 
                    
                    ## Write out raw LFP in patient DB
                    ptDB["RawLFP"][contact]["LFP"] = Data["LFP"][contact]["rawLFP"][restMask]
                    
                    ## Write out raw spectrogram in patient DB
                    rawSpect = SignalProcessingUtility.defaultSpectrogram(Data["LFP"][contact]["rawLFP"][restMask], 
                                                                  window=1.0, overlap=0.5, 
                                                                  frequency_resolution=0.5, 
                                                                  fs=Data["LFP"]["SamplingRate_Hz"])
                    spectDB["Time"] = rawSpect["Time"]
                    spectDB["Frequency"] = rawSpect["Frequency"][rawSpect["Frequency"]<=freqNorm]
                    ptDB["Params"]["Spectrogram"] = rawSpect["Config"]
                    ptDB["RawLFP"][contact]["Spect_LogPower"] = rawSpect["logPower"][rawSpect["Frequency"]<=freqNorm]

                    ### Filtered PSD (Butterworth)
                    
                    freq, PSD = signal.welch(Data["LFP"][contact]["filtLFP"][restMask], 
                                              fs=Data["LFP"]["SamplingRate_Hz"],
                                              nperseg=Data["LFP"]["SamplingRate_Hz"] * 1,
                                              noverlap=Data["LFP"]["SamplingRate_Hz"] * 0.5,
                                              nfft=Data["LFP"]["SamplingRate_Hz"] * 2, scaling="density") 
                   
                                                            
                    freqThresh = 100
                    LowFitFreq = freq <= freqThresh
                    
                    HiFitFreq = freq > freqThresh
                    HiFitFreq[freq > freqNorm] = False

                    print(contact,"Notch filtering signals...")
                    
                    notchLFP = Data["LFP"][contact]["filtLFP"][restMask]
                    
                    wf = []
                    for i,p in enumerate(freq[peaks]):
                        w = math.ceil(widths[i]/2)+1
                        LowFitFreq[(freq>=p-w)&(freq<=p+w)] = False
                        HiFitFreq[(freq>=p-w)&(freq<=p+w)] = False
                        
                        # Notch filter butterworth filtered signal to write out in patient DB
                        
                        bNotch, aNotch = signal.iirnotch(p, p/w, Data["LFP"]["SamplingRate_Hz"])
                        notchFreq, h = signal.freqz(bNotch, aNotch, fs=Data["LFP"]["SamplingRate_Hz"])
                        notchLFP = signal.filtfilt(bNotch, aNotch, notchLFP)
                        
                        print(p,w,p/w)
                        wf.append(w)
                    
                    ## Write out filtered LFP in patient DB
                    ptDB["FiltLFP"][contact]["LFP"] = notchLFP
                    ptDB["Params"]["NotchFreqPeaks"] = peakFreqs
                    ptDB["Params"]["NotchFreqWidths"] = wf
                    
                    ## Write out filtered spectrogram in patient DB
                    filtSpect = SignalProcessingUtility.defaultSpectrogram(notchLFP, 
                                                                  window=1.0, overlap=0.5, 
                                                                  frequency_resolution=0.5, 
                                                                  fs=Data["LFP"]["SamplingRate_Hz"])
                    
                    ptDB["FiltLFP"][contact]["Spect_LogPower"] = filtSpect["logPower"][filtSpect["Frequency"]<=freqNorm]
                 
                    # =============================================================================
                    # FOOOF normalization
                    # =============================================================================
                    
                    lowfreqNorm = freq<=freqThresh
                                                            
                    model = FOOOF(min_peak_height=0.05, verbose=False)
                    model.fit(freq[lowfreqNorm],PSD[lowfreqNorm])
                    [offset,exp] = model.aperiodic_params_
                    lowFitNorm = offset - np.log10(np.power(freq[lowfreqNorm],exp))
                    # print(model.r_squared_, model.error_)
                    
                    lownormPSD = np.log10(PSD[lowfreqNorm])-lowFitNorm
                    lowrawPSD = np.log10(PSD[lowfreqNorm])
                    
                    # =============================================================================
                    # Normalize high freq using fitted exponential
                    # =============================================================================
                    
                    hinormFit = SignalProcessingUtility.fittedNormalization(np.log10(PSD),freq,HiFitFreq,order=2)
                    hinormPSD = np.log10(PSD[freq>freqThresh])-hinormFit[freq>freqThresh]
                    
                    ## Combine low and high freq normalized into one PSD  
                    normPSD = np.array(list(lownormPSD) + list(hinormPSD))
                    nonormPSD = np.log10(PSD)
                    
                    ## Write out normed spectrogram to pt DB
                    FitNorm = np.array(list(lowFitNorm) + list(hinormFit[freq>freqThresh]))
                    ptDB["FiltLFP"][contact]["NormSpect_LogPower"] = np.array(filtSpect["logPower"][filtSpect["Frequency"]<=freqNorm] - FitNorm[freq<=freqNorm][:,None])

                    # Exclude noise peaks from normalized PSD
                    for i,p in enumerate(freq[peaks]):
                        w = math.ceil(widths[i]/2)+1
                        nonormPSD[(freq>=p-w)&(freq<=p+w)] = np.nan
                        normPSD[(freq>=p-w)&(freq<=p+w)] = np.nan
                    
                    # =============================================================================
                    # Save + plot data
                    # =============================================================================
                    
                    DB[ptID][inFisplt]["RestLFP"][contact] = {}
                    DB[ptID][inFisplt]["RestLFP"][contact]["RawPSD"] = rawPSD
                    DB[ptID][inFisplt]["RestLFP"][contact]["NoNormPSD"] = nonormPSD
                    DB[ptID][inFisplt]["RestLFP"][contact]["NormPSD"] = normPSD
                    DB[ptID][inFisplt]["RestLFP"][contact]["FOOOF"] = {}
                    DB[ptID][inFisplt]["RestLFP"][contact]["FOOOF"]["Aperiodic"] = model.aperiodic_params_
                    DB[ptID][inFisplt]["RestLFP"][contact]["FOOOF"]["Fit"] = [model.r_squared_, model.error_]
                    if len(DB["PSDFreq"]) == 0:
                        DB["PSDFreq"] = freq
                        
                    axraw.plot(freq[freq<=freqNorm],nonormPSD[freq<=freqNorm])
                    axraw.set_xlim(0,freqNorm)
                    axraw.set_xlabel("Frequency (Hz)")
                    axraw.set_ylabel("PSD")
                    axraw.set_title(ptID+" "+str(informID))
                    
                    axrawLF.plot(freq[freq<=freqThresh],nonormPSD[freq<=freqThresh])
                    axrawLF.set_xlim(0,freqThresh)
                    axrawLF.set_xlabel("Frequency (Hz)")
                    axrawLF.set_ylabel("PSD")
                    axrawLF.set_title(ptID+" "+str(informID))

                    axLF.plot(freq[freq<=freqThresh],normPSD[freq<=freqThresh])
                    axLF.set_xlim(0,freqThresh)
                    axLF.set_xlabel("Frequency (Hz)")
                    axLF.set_ylabel("PSD")
                    axLF.set_title(ptID+" "+str(informID)+" Norm")
                        
                    ax500.plot(freq[freq<=freqNorm],normPSD[freq<=freqNorm])
                    ax500.set_xlim(0,freqNorm)
                    ax500.set_xlabel("Frequency (Hz)")
                    ax500.set_ylabel("PSD")
                    ax500.set_title(ptID+" "+str(informID)+" Norm")
                    sns.despine()
                    
                    if numContacts == 8:
                        renameCons = {"Contact_0":"Contact 0",
                                      "Contact_1":"Contact 1a",
                                      "Contact_2":"Contact 1b",
                                      "Contact_3":"Contact 1c",
                                      "Contact_4":"Contact 2a",
                                      "Contact_5":"Contact 2b",
                                      "Contact_6":"Contact 2c",
                                      "Contact_7":"Contact 3"}
                        lines.append(renameCons[contact])
                    else:
                        lines.append(" ".join(contact.split("_")[:]))
                                    
                    ### Calculate mean / max power 
                                        
                    deltaAvgnorm = np.nanmean(normPSD[(freq>=1)&(freq<=4)])
                    thetaAvgnorm = np.nanmean(normPSD[(freq>4)&(freq<=8)])
                    alphaAvgnorm = np.nanmean(normPSD[(freq>8)&(freq<=12)])
                    betaAvgnorm = np.nanmean(normPSD[(freq>=13)&(freq<=30)])
                    lowbetaAvgnorm = np.nanmean(normPSD[(freq>=13)&(freq<=20)])
                    highbetaAvgnorm = np.nanmean(normPSD[(freq>20)&(freq<=30)])
                    gammaAvgnorm = np.nanmean(normPSD[(freq>30)&(freq<=100)])
                    lowgammaAvgnorm = np.nanmean(normPSD[(freq>30)&(freq<=50)])
                    highgammaAvgnorm = np.nanmean(normPSD[(freq>50)&(freq<=100)])
                    hfoAvgnorm = np.nanmean(normPSD[(freq>=200)&(freq<=400)])
                    
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"] = {}
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"]["Delta"] = deltaAvgnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"]["Theta"] = thetaAvgnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"]["Alpha"] = alphaAvgnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"]["Beta"] = betaAvgnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"]["LowBeta"] = lowbetaAvgnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"]["HighBeta"] = highbetaAvgnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"]["Gamma"] = gammaAvgnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"]["LowGamma"] = lowgammaAvgnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"]["HighGamma"] = highgammaAvgnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower_Norm"]["HFO"] = hfoAvgnorm
                    
                    deltaMaxnorm = np.nanmax(normPSD[(freq>=1)&(freq<=4)])
                    thetaMaxnorm = np.nanmax(normPSD[(freq>4)&(freq<=8)])
                    alphaMaxnorm = np.nanmax(normPSD[(freq>8)&(freq<=12)])
                    betaMaxnorm = np.nanmax(normPSD[(freq>=13)&(freq<=30)])
                    lowbetaMaxnorm = np.nanmax(normPSD[(freq>=13)&(freq<=20)])
                    highbetaMaxnorm = np.nanmax(normPSD[(freq>20)&(freq<=30)])
                    gammaMaxnorm = np.nanmax(normPSD[(freq>30)&(freq<=100)])
                    lowgammaMaxnorm = np.nanmax(normPSD[(freq>30)&(freq<=50)])
                    highgammaMaxnorm = np.nanmax(normPSD[(freq>50)&(freq<=100)])
                    hfoMaxnorm = np.nanmax(normPSD[(freq>=200)&(freq<=400)])
                    
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"] = {}
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"]["Delta"] = deltaMaxnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"]["Theta"] = thetaMaxnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"]["Alpha"] = alphaMaxnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"]["Beta"] = betaMaxnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"]["LowBeta"] = lowbetaMaxnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"]["HighBeta"] = highbetaMaxnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"]["Gamma"] = gammaMaxnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"]["LowGamma"] = lowgammaMaxnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"]["HighGamma"] = highgammaMaxnorm
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower_Norm"]["HFO"] = hfoMaxnorm
                    
                    deltaAvg = np.nanmean(rawPSD[(freq>=1)&(freq<=4)])
                    thetaAvg = np.nanmean(rawPSD[(freq>4)&(freq<=8)])
                    alphaAvg = np.nanmean(rawPSD[(freq>8)&(freq<=12)])
                    betaAvg = np.nanmean(rawPSD[(freq>=13)&(freq<=30)])
                    lowbetaAvg = np.nanmean(rawPSD[(freq>=13)&(freq<=20)])
                    highbetaAvg = np.nanmean(rawPSD[(freq>20)&(freq<=30)])
                    gammaAvg = np.nanmean(rawPSD[(freq>30)&(freq<=100)])
                    lowgammaAvg = np.nanmean(rawPSD[(freq>30)&(freq<=50)])
                    highgammaAvg = np.nanmean(rawPSD[(freq>50)&(freq<=100)])
                    hfoAvg = np.nanmean(rawPSD[(freq>=200)&(freq<=400)])
                    
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"] = {}
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"]["Delta"] = deltaAvg
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"]["Theta"] = thetaAvg
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"]["Alpha"] = alphaAvg
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"]["Beta"] = betaAvg
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"]["LowBeta"] = lowbetaAvg
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"]["HighBeta"] = highbetaAvg
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"]["Gamma"] = gammaAvg
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"]["LowGamma"] = lowgammaAvg
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"]["HighGamma"] = highgammaAvg
                    DB[ptID][inFisplt]["RestLFP"][contact]["MeanPower"]["HFO"] = hfoAvg
                    
                    deltaMax = np.nanmax(rawPSD[(freq>=1)&(freq<=4)])
                    thetaMax = np.nanmax(rawPSD[(freq>4)&(freq<=8)])
                    alphaMax = np.nanmax(rawPSD[(freq>8)&(freq<=12)])
                    betaMax = np.nanmax(rawPSD[(freq>=13)&(freq<=30)])
                    lowbetaMax = np.nanmax(rawPSD[(freq>=13)&(freq<=20)])
                    highbetaMax = np.nanmax(rawPSD[(freq>20)&(freq<=30)])
                    gammaMax = np.nanmax(rawPSD[(freq>30)&(freq<=100)])
                    lowgammaMax = np.nanmax(rawPSD[(freq>30)&(freq<=50)])
                    highgammaMax = np.nanmax(rawPSD[(freq>50)&(freq<=100)])
                    hfoMax = np.nanmax(rawPSD[(freq>=200)&(freq<=400)])
                    
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"] = {}
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"]["Delta"] = deltaMax
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"]["Theta"] = thetaMax
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"]["Alpha"] = alphaMax
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"]["Beta"] = betaMax
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"]["LowBeta"] = lowbetaMax
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"]["HighBeta"] = highbetaMax
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"]["Gamma"] = gammaMax
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"]["LowGamma"] = lowgammaMax
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"]["HighGamma"] = highgammaMax
                    DB[ptID][inFisplt]["RestLFP"][contact]["MaxPower"]["HFO"] = hfoMax
                    
                    csvRow = [ptID,informID,target,hemi,dx,restEnd,contact,offset,exp,model.r_squared_,model.error_,
                                    deltaAvg,thetaAvg,alphaAvg,betaAvg,lowbetaAvg,highbetaAvg,
                                    gammaAvg,lowgammaAvg,highgammaAvg,hfoAvg,
                                    deltaMax,thetaMax,alphaMax,betaMax,lowbetaMax,highbetaMax,
                                    gammaMax,lowgammaMax,highgammaMax,hfoMax,
                                    deltaAvgnorm,thetaAvgnorm,alphaAvgnorm,betaAvgnorm,lowbetaAvgnorm,highbetaAvgnorm,
                                    gammaAvgnorm,lowgammaAvgnorm,highgammaAvgnorm,hfoAvgnorm,
                                    deltaMaxnorm,thetaMaxnorm,alphaMaxnorm,betaMaxnorm,lowbetaMaxnorm,highbetaMaxnorm,
                                    gammaMaxnorm,lowgammaMaxnorm,highgammaMaxnorm,hfoMaxnorm]
                    
                    sumVars.append(csvRow)
    
                    # if overWrite == False:
                    #     with open(csvOutFi,'a') as csvfi:
                    #         writer = csv.writer(csvfi)
                    #         writer.writerow(csvRow)
                
            outDF = pd.DataFrame(sumVars,columns=["PtID","INFORM_ID","Target","Hemi","Dx","RestTime","Contact","FOOOFOffset","FOOOFExp","FOOOFR2","FOOOFError",
                                                  "Delta","Theta","Alpha","Beta","LowBeta",
                                                  "HighBeta","Gamma","LowGamma","HighGamma","HFO",
                                                  "DeltaMax","ThetaMax","AlphaMax","BetaMax","LowBetaMax",
                                                  "HighBetaMax","GammaMax","LowGammaMax","HighGammaMax","HFOMax",
                                                  "DeltaNorm","ThetaNorm","AlphaNorm","BetaNorm","LowBetaNorm",
                                                  "HighBetaNorm","GammaNorm","LowGammaNorm","HighGammaNorm","HFONorm",
                                                  "DeltaNormMax","ThetaNormMax","AlphaNormMax","BetaNormMax","LowBetaNormMax",
                                                  "HighBetaNormMax","GammaNormMax","LowGammaNormMax","HighGammaNormMax","HFONormMax"])

            # Write out LFP DB for each patient
            # print("Writing out patient's LFP DB...")
            # ptDBfi = os.path.join(rawOutDir,str(informID)+"_"+ptID+"_"+hemi+"_"+dx+".json")
            # ptDBjson = json.dumps(ptDB,cls=NumpyEncoder)
            # with open(ptDBfi, "w+") as ptfile:
            #     ptfile.write(ptDBjson)
            # ptfile.close()
            
            # spectDBfi = os.path.join(rawOutDir,"SpectrogramPlot.json")
            # if not os.path.exists(spectDBfi):
            #     spectDBjson = json.dumps(spectDB,cls=NumpyEncoder)
            #     with open(spectDBfi,"w+") as spfile:
            #         spfile.write(spectDBjson)
            #     spfile.close()
            
            print("Writing...")
            if overWrite == True:
                outDF.to_csv(csvOutFi,index=False)
        
            jsonOut = json.dumps(DB, cls=NumpyEncoder)
            
            with open(jsonOutFi, "w+") as file:
                file.write(jsonOut)
            file.close()
            
            axraw.legend(axraw.get_lines(), lines, frameon=False, loc="upper right")
            axrawLF.legend(axrawLF.get_lines(), lines, frameon=False, loc="upper right")
            axLF.legend(axLF.get_lines(), lines, frameon=False, loc="upper right")
            ax500.legend(ax500.get_lines(), lines, frameon=False, loc="upper right")
            
            # fiRoot = ptID+"_"+str(informID)+"_"+dx
            # figrawLF.savefig(os.path.join(outDir,"PSD_AutoNoiseDetect_New20240612",fiRoot+"_LF_NoNorm.png"),dpi=300)
            # figraw.savefig(os.path.join(outDir,"PSD_AutoNoiseDetect_New20240612",fiRoot+"_500Hz_NoNorm.png"),dpi=300)
            # figLF.savefig(os.path.join(outDir,"PSD_AutoNoiseDetect_New20240612",fiRoot+"_LF.png"),dpi=300)
            # fig500.savefig(os.path.join(outDir,"PSD_AutoNoiseDetect_New20240612",fiRoot+"_500Hz.png"),dpi=300)

            plt.close("all")
