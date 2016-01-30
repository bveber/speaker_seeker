###Standard Python packages###
import time, random, pickle, math, os, subprocess, multiprocessing, re
import matplotlib.pyplot as plt
from collections import Counter
from joblib import Parallel, delayed
###Scientific + Math packages
from scipy.io import wavfile
from scipy.signal import butter, lfilter, wiener
from scipy.fftpack import dct
import numpy as np
import pandas as pd
###ML packages###
from sklearn import grid_search, svm, ensemble, neighbors, metrics, preprocessing
###Other###
import pysrt
from speakerFeatures_0 import getFeaturesParallel

#Global Initializers
num_cores = multiprocessing.cpu_count()
fps=10
ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"

def main(episode,subtitles=False):
    numbers=re.findall(r'\d+',episode)
    print(numbers,episode)
    episodePickle = '/'.join(episode.split('/')[:-1])+"/Simpsons_"+numbers[1]+"x"+numbers[2]+".p"
    preds = runEpisode(episodePickle,subtitles)
    for character in ['Burns.p','Principal.p','Moe.p','Flanders.p']:
        cutEpisode(preds,episode,character)

def calcAndSaveFeatures(episode):
    episodeFeatures = getFeaturesParallel(episode)
    pickle.dump({'features':episodeFeatures},open(episode.split('.wav')[0]+'.p','wb'))
    
def runEpisode(episode,subtitles=True):
    t0=time.time()
    folderLoc = 'C:/Users/Brandon/Documents/Simpsons_Project/Character_Pickles/'
    models = {}; imputers = {}; scalers = {}
    for character in os.listdir(folderLoc):
        if len(character.split('.p'))<=1: continue
        print(character)
        models[character] = pickle.load(open(folderLoc+character,'rb'))['model']
        imputers[character] = pickle.load(open(folderLoc+character,'rb'))['imputer']
        scalers[character] = pickle.load(open(folderLoc+character,'rb'))['scaler']
    if episode.split('.')[-1] == 'p':
        savedInfo = pickle.load(open(episode,'rb'))
        episodeFeatures = savedInfo['features']
    else:
        episodeFeatures = getFeaturesParallel(episode,True)    if subtitles: subtitles = getSubtitles(episode.split('.')[0]+'.srt')
    preds = predict(subtitles,episodeFeatures,models,imputers,scalers)
    print('Episode Runtime is: ',time.time()-t0,'(s)')
    return(preds)

def predict(subtitles,episodeFeatures,models,imputers,scalers):
    preds= pd.DataFrame(columns=['start','end','text']+list(models.keys()))
    if 'episode' in list(models.keys()): preds = preds.drop('episode',1)
    if not subtitles: preds.drop('text',1)    
    if subtitles: iterLen = len(subtitles)
    else:
        dur = 1
        starts = range(int(len(episodeFeatures)/(dur*2*fps)))
        iterLen = len(starts)
    print(iterLen)
    for i in range(iterLen):
        if subtitles:
            tempPreds = {'start':subtitles.ix[i]['start'],'end':subtitles.ix[i]['end'],'text':subtitles.ix[i]['text']}
            startIndex = subtitles.ix[i]['start']*fps*2
            endIndex = subtitles.ix[i]['end']*fps*2
        else:
            tempPreds = {'start':starts[i],'end':starts[i]+dur}
            startIndex = starts[i]*2*fps
            endIndex = (starts[i]+dur)*2*fps
        if endIndex > 0 and len(episodeFeatures[startIndex:endIndex])>1:
            for character in models:
                if character != 'episode':
                    tempPreds[character] = np.mean(models[character].predict_proba(episodeFeatures[startIndex:endIndex])[:,1])
            preds.loc[i] = pd.Series(tempPreds)
    return(preds)

def cutEpisode(predictions,episode,character='Flanders.p'):
    episode = episode.replace('/','\\')
    shift = 0
    lim = {'Burns.p': .75, 'Flanders.p': .4, 'Principal.p': .6, 'Moe.p': .6, 'Homer.p':.5}
    indices = predictions[predictions[character]>lim[character]].index.tolist()
    if len(indices)==0: return
    splits = []
    for i,index in enumerate(indices[1:]):
        if int(predictions.ix[index]['start']) - int(predictions.ix[indices[i]]['end']) > 20: splits.append(index)
    if len(indices)>0 and len(splits)==0:
        start = indices[0]; end = indices[-1]
        minutes = str(math.floor(predictions.ix[start]['start']/60))
        if len(minutes) < 2: minutes = "0"+minutes 
        seconds = str(int(predictions.ix[start]['start'] - int(minutes)*60))
        if len(seconds) < 2: seconds = "0"+seconds
        durationSeconds = math.floor(predictions.ix[end]['end']-predictions.ix[start]['start'])
        videoCall(episode,character,str(start),minutes,seconds,durationSeconds)
    else:
        start = indices[np.where(indices < splits[0])[0][0]]
        end = indices[np.where(indices == splits[0])[0][0]-1]
    for i,split in enumerate(splits):
##        print('end_0: ',predictions.ix[end]['end'],', start_0: ',predictions.ix[start]['start'])
        if end-start <= 3:
            if i < len(splits)-1:
                start = indices[np.where(((indices < splits[i+1]) | (indices == splits[i+1])) & (indices > split))[0][0]]
                if start == splits[i+1]: end = splits[i+1]
                else: end = indices[np.where(indices == splits[i+1])[0][0]-1]
##            print('split: ',split,' start_1: ',start,'end_1',end)
            continue
        minutes = str(math.floor(predictions.ix[start]['start']/60))
        if len(minutes) < 2: minutes = "0"+minutes 
        seconds = str(int(predictions.ix[start]['start'] - int(minutes)*60)-1)
##        print('seconds: ',seconds)
        if len(seconds) < 2: seconds = "0"+seconds 
        durationSeconds = math.floor(predictions.ix[end]['end']-predictions.ix[start]['start']+4)
##        print('duration(s): ',durationSeconds)
        videoCall(episode,character,str(start),minutes,seconds,durationSeconds)
        if i < len(splits)-1:
            start = indices[np.where(((indices < splits[i+1]) | (indices == splits[i+1])) & (indices > split))[0][0]]
            if start == splits[i+1]: end = splits[i+1]
            else: end = indices[np.where(indices == splits[i+1])[0][0]-1]
##        print('split: ',split,' start_1: ',start,'end_1',end)
    try:  
        if splits[-1] == indices[-1]:
            minutes = str(math.floor(predictions.ix[splits[-1]]['start']/60))
            if len(minutes) < 2: minutes = "0"+minutes 
            seconds = str(int(predictions.ix[splits[-1]]['start'] - int(minutes)*60))
            if len(seconds) < 2: seconds = "0"+seconds
            durationSeconds = math.floor(predictions.ix[splits[-1]]['end']-predictions.ix[splits[-1]]['start'])
            videoCall(episode,character,str(splits[-1]),minutes,seconds,durationSeconds)
    except:
        pass

def videoCall(episode,character,ind,minutes,seconds,durationSeconds):
    epInd = episode.find('x')+1
    epNum = episode[epInd:epInd+2]
    epFolder = '\\'.join(np.append((episode.split('\\')[:-1]),"ep"+epNum))
    try: os.mkdir(epFolder)
    except: pass
    if durationSeconds > 5: 
        durationMinutes = str(int(math.floor(int(durationSeconds)/60)))
        durationSeconds = str(int(durationSeconds-int(durationMinutes)*60))
        print(durationSeconds)
        if len(durationSeconds) < 2: durationSeconds = "0"+durationSeconds
        if len(durationMinutes) < 2: durationMinutes = "0"+durationMinutes
        command = ffmpeg+" -ss "+"00:"+minutes+":"+seconds+" -i "+episode+" -t 00:"+durationMinutes+":"+\
                  durationSeconds+" -c:v copy -c:a copy "+epFolder+'/'+character.split('.')[0]+ind+".avi"
        command = command.replace('/','\\')
        print(command)
        subprocess.call(command)#,shell=True)

def indexToTime(predInds):
    return([str(int(divmod(elem/(fps*2),60)[0]))+':'+str(divmod(elem/(fps*2),60)[1]) for elem in predInds])

def getSubtitles(srtFile):

    episodeSubtitles = pd.DataFrame(columns=['start','end','text'])
    subs = pysrt.open(srtFile)
    subs.shift(seconds=5)
    for i,sub in enumerate(subs):
        episodeSubtitles.loc[i] = [int(sub.start.minutes)*60+int(sub.start.seconds),
                                   int(sub.end.minutes)  *60+int(sub.end.seconds),
                                  sub.text]
    return(episodeSubtitles)
