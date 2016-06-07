'''
Created on Feb 26, 2016

@author: Preston Wilson

'''
from flask import Flask, render_template, request
import sys
import ftplib
import os
import pyaudio
import wave 
import subprocess
import mpld3
from PIL import Image
import cv2
import numpy as np
import os.path
from matplotlib import pyplot as plt
import webbrowser
import matplotlib.cm as cm
import Dates

app = Flask(__name__)


htmlname = None
files = None
ftp = None
mid = None
curr = 0

'''
Credit to user Rinold from 
http://stackoverflow.com/questions/6951046/pyaudio-help-play-a-file
for the following class 
'''
class AudioFile: #Wrapper class for playing audio files with PyAudio
    chunk = 1024

    def __init__(self, file): 
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def play(self):
        data = self.wf.readframes(self.chunk)
        while data != '':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        self.stream.close()
        self.p.terminate()



@app.route("/")
def main(): #Login page
    return render_template("index.html")


@app.route("/validateLogin", methods = ['POST'])
def validateLogin(): #Authentication page, will download files and render play page if user is found on cs.appstate.edu
    global ftp
    global files
    global mid
    global curr
    global htmlname
    if request.form['media'] == 'audio':
        try:
            _username = request.form['inputEmail']
            _password = request.form['inputPassword']
            _directory = request.form['directory']
            try:
                _date = request.form['inputDate']
                _time = request.form['inputTime']
            except Exception as e:
                print e
            ftp = ftplib.FTP("cs.appstate.edu", _username, _password)
            ftp.cwd('/usr/local/bee/beemon/%s' % _directory)
            if len(_date) == 10:
                try:
                    audio_dir = '/usr/local/bee/beemon/%s/%s/audio' % (_directory, _date)
                    print 'new directory chosen'
                except Exception as e:
                    print e
            else:
                items = ftp.nlst()
                _date = str(items[0])
                print 'date defaulting to first in directory:', _date
                audio_dir = '/usr/local/bee/beemon/%s/%s/audio' % (_directory, _date)
            ftp.cwd(audio_dir)
            print 'current directory:', audio_dir
            files = ftp.nlst() #all audio files in the directory
            #os.rmdir('Audio_Files')
            try:
                os.mkdir('Audio_Files_' + _directory + '_' + _date)
            except Exception as e:
                "Directory exists"
            finally:
                os.chdir('Audio_Files_' + _directory + '_' + _date)
                mid = int(len(files) / 2) #default is the middle file in the directory 
                if (len(_time) > 0): #if user specified, try to find specific time file
                    try:
                        for x in range(0, len(files)): 
                            if files[x] == str(_time + '.wav'):
                                mid = x
                                print 'mid file changed to: ', files[mid]
                    except Exception as e:
                        print 'error finding time ', e
                        
                #first = files[mid]
                if files[mid] is not None:
                    #print 'where did it go wrong? Here?'
                    fhandle = open(files[mid], 'wb')
                    ftp.retrbinary('RETR ' + files[mid], fhandle.write)
                    fhandle.close()
                    '''
                    temp = tempfile.mkstemp('', f, 'Audio_Files', False)
                    print f
                    print temp[1]
                    print 'here?'
                    op = open(temp[1], "wb")
                    print f
                    ftp.retrbinary("RETR " + temp[1], op.write)
                    op.close()
                    '''
                return render_template('audio.html')
            
            
        except Exception as e:
            print e
            return render_template('index.html', name = 'Audio and Video', error = str(e))
    elif request.form['media'] == 'video':
        try:
            _username = request.form['inputEmail']
            _password = request.form['inputPassword']
            _directory = request.form['directory']
            try:
                _date = request.form['inputDate']
                _time = request.form['inputTime']
            except Exception as e:
                print e
            ftp = ftplib.FTP("cs.appstate.edu", _username, _password)
            ftp.cwd('/usr/local/bee/beemon/%s' % _directory)
            if len(_date) == 10:
                try:
                    video_dir = '/usr/local/bee/beemon/%s/%s/video' % (_directory, _date)
                    print 'new directory chosen'
                except Exception as e:
                    print e
            else:
                items = ftp.nlst()
                _date = str(items[0])
                print 'date defaulting to first in directory:', _date
                video_dir = '/usr/local/bee/beemon/%s/%s/video' % (_directory, _date)
            ftp.cwd(video_dir)
            print 'current directory:', video_dir
            files = ftp.nlst() #all video files in the directory
            try:
                os.mkdir('Video_Files_' + _directory + '_' + _date)
            except Exception as e:
                "Directory exists"
            finally:
                os.chdir('Video_Files_' + _directory + '_' + _date)
                mid = int(len(files) / 2) #default is the middle file in the directory 
                if (len(_time) > 0): #if user specified, try to find specific time file
                    try:
                        print to_hex(_date, _time)
                        for x in range(0, len(files)): 
                            if files[x] == str(_time + '.h264'):
                                mid = x
                                print 'mid file: ', files[mid]
                    except Exception as e:
                        print 'error finding time ', e

                if files[mid] is not None:
                    # print 'where did it go wrong? Here?'
                    fhandle = open(files[mid], 'wb')
                    ftp.retrbinary('RETR ' + files[mid], fhandle.write)
                    fhandle.close()
                return render_template('video.html')
            
            
        except Exception as e:
            print e
            
            return render_template('index.html', name = 'Audio and Video', error = str(e))
        return render_template('index.html')
    
    elif request.form['media'] == 'specto': # they clicked the 'Entropy' button
        try:
            _username = request.form['inputEmail']
            _password = request.form['inputPassword']
            _directory = request.form['directory']
            try:
                _date = request.form['inputDate']
                _time = request.form['inputTime']
            except Exception as e:
                print e
            ftp = ftplib.FTP("cs.appstate.edu", _username, _password)
            ftp.cwd('/usr/local/bee/beemon/%s' % _directory)
            if len(_date) == 10:
                try:
                    photo_dir = '/usr/local/bee/beemon/%s/%s/video' % (_directory, _date)
                    print 'new directory chosen'
                except Exception as e:
                    print e
            else:
                items = ftp.nlst()
                _date = str(items[0])
                print 'date defaulting to first in directory:', _date
                photo_dir = '/usr/local/bee/beemon/%s/%s/video' % (_directory, _date)
            ftp.cwd(photo_dir)
            print 'current directory:', photo_dir
            files = ftp.nlst()
            curr = 0
            try:
                os.mkdir('Photo_Files_' + _directory + '_' + _date)
            except Exception as e:
                "Directory exists"
            finally:
                os.chdir('Photo_Files_' + _directory + '_' + _date)
                print 'current dir ', os.getcwd()
                #currfile = 'GoAnimate for Calculus.mp4'
                tempfile = 'GoAnimate for Calculus.mp4'

                return render_template('entropy.html') #load new page which carries out the code in the rest of specto

                time_length = 142.0
                fps = 30.0
                frame = 900
                frame_no = frame / (time_length * fps)
                while curr < int(len(files)):
                    try:
                        fhandle = open(files[curr], 'ab')  # choose first file of directory for download
                        if os.path.getsize(fhandle.name) == 0: #change conditional to now always be true later
                            print 'not a duplicate'
                            ftp.retrbinary('RETR ' + files[curr], fhandle.write)
                            currfile = files[curr]
                            cap = cv2.VideoCapture(currfile)
                            if cap.isOpened():
                                cap.set(1, frame)
                                ret, newframe = cap.read()
                                # cv2.imshow('window_name', newframe)
                                # gray = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
                                if ret:
                                    print 'yea its true'
                                    newname = currfile.split(".")[0]
                                    finalname = newname + '_frame_' + str(frame) + '.jpg'
                                    cv2.imwrite(finalname, newframe)
                                    '''
                                    Credit to  Johannes Maucher
                                    at https://www.hdm-stuttgart.de/~maucher
                                    for the following code
                                    '''
                                    colorIm = Image.open(finalname)
                                    greyIm = colorIm.convert('L')
                                    colorIm = np.array(colorIm)
                                    greyIm = np.array(greyIm)
                                    N = 5
                                    S = greyIm.shape
                                    E = np.array(greyIm)
                                    for row in range(S[0]):
                                        for col in range(S[1]):
                                            Lx = np.max([0, col - N])
                                            Ux = np.min([S[1], col + N])
                                            Ly = np.max([0, row - N])
                                            Uy = np.min([S[0], row + N])
                                            region = greyIm[Ly:Uy, Lx:Ux].flatten()
                                            E[row, col] = entropy(region)
                                    grayImage = cv2.applyColorMap(greyIm, cv2.COLOR_BGR2GRAY)
                                    entropyImage = cv2.applyColorMap(greyIm, cv2.COLORMAP_JET)
                                    cv2.imwrite('gray_' + finalname, greyIm)
                                    #cv2.imwrite('color_' + finalname, entropyImage)

                                    '''
                                    try:
                                        center, start_dir = to_hex(_date, _time)
                                        title = 'something' + "  -  " + 'something else'
                                        fig, ax = plt.subplots()

                                        fig.canvas.draw()
                                        ax.clear()
                                        ax.set_xticks(np.arange(0, 1.0))
                                        ax.set_title(title)

                                        start_hex = center
                                        combined_spec = np.empty((2049, 2 ** 1))
                                        combined_spec[:] = np.NaN
                                        end_hex = '{:08x}'.format(int(start_hex, 16) + 2 ** 1)
                                        for i in range(int(start_hex[:5], 16), int(end_hex[:5], 16) + 1):
                                            i_hex = '{:05x}'.format(i)
                                            i_specgram = colorIm
                                            if start_hex[:5] == end_hex[:5] and start_hex[:5] == i_hex:
                                                start_col = int(start_hex, 16) - int(make_hex8(start_hex[:5]), 16)
                                                end_col = int(end_hex, 16) - int(make_hex8(start_hex[:5]), 16)
                                                combined_spec[:, :(end_col - start_col)] = i_specgram[:,
                                                                                           start_col:end_col]
                                            elif start_hex[:5] == i_hex:
                                                start_col = int(start_hex, 16) - int(make_hex8(start_hex[:5]), 16)
                                                end_col = i_specgram[:, start_col:].shape[1]
                                                combined_spec[:, 0:end_col] = i_specgram[:, start_col:]
                                            elif end_hex[:5] == i_hex:
                                                start_col = int(make_hex8(i_hex), 16) - int(start_hex, 16)
                                                end_col = int(end_hex, 16) - int(make_hex8(i_hex), 16)
                                                combined_spec[:, start_col:] = i_specgram[:, :end_col]
                                            else:
                                                start_col = int(make_hex8(i_hex), 16) - int(start_hex, 16)
                                                end_col = i_specgram.shape[1]
                                                combined_spec[:, start_col:start_col + end_col] = i_specgram[:, :]
                                        print combined_spec.shape

                                        cax = ax.imshow(20 * np.log10(combined_spec), origin='lower',
                                                        aspect='auto', interpolation='nearest')
                                        fig.colorbar(cax)
                                        mpld3.show()
                                    except Exception as e:
                                        print e

                                    '''
                                    #plt.subplot(1, 3, 1)
                                    #plt.imshow(colorIm, origin="lower")

                                    #plt.subplot(1, 3, 2)
                                    #plt.imshow(greyIm, cmap=plt.get_cmap("gray"), origin="lower")

                                    #plt.subplot(1, 1, 1)
                                    #np.flipud()

                                    a = np.empty_like(E)
                                    a[:,:] = E
                                    a = np.flipud(a)
                                    plt.figure()
                                    plt.imshow(a, cmap=plt.get_cmap("jet"), origin="lower")
                                    plt.xlabel('Entropy in 10x10 neighborhood')
                                    plt.colorbar()
                                    plt.savefig('color_' + finalname, bbox_inches='tight')
                                    plt.imshow(E, cmap=plt.get_cmap("jet"), origin="lower")
                                    plt.plot()
                                    #mpld3.show()


                            cap.release()
                            print 'finished ', curr
                        else:
                            print 'duplicate'
                        #return render_template('index.html') #temporary for testing delete later
                        curr += 1

                    except Exception as e:
                        print e

                '''
                try:
                    hexnumrep = Dates.to_hex(_date, _time)
                    fig, ax = plt.subplots()
                    fig.canvas.draw()
                    ax.clear()
                    ax.set_xticks(np.arange(0, 1.0))

                    cax = ax.imshow(20 * np.log10(self.current_spec), origin='lower',
                                         aspect='auto', interpolation='nearest')
                    mpld3.show()
                except Exception as e:
                    print e
                '''
                return render_template('entropy.html')

        except Exception as e:
            print e




@app.route("/playAudio", methods=['POST'])
def playAudio():
    global ftp
    global files
    global mid
    if request.form['play'] == 'play':  
        try:  
            print "playing ", files[mid]
            a = AudioFile(files[mid])
            a.play()
        except Exception as e:
            print e
    elif request.form['play'] == 'next':
        print 'nexting'
        try:
            mid = mid + 1
            fhandle = open(files[mid], 'wb')
            ftp.retrbinary('RETR ' + files[mid], fhandle.write)
            fhandle.close() 
            print mid 
        except Exception as e:
            print e
            
    elif request.form['play'] == 'back':
        print 'backing'
        try:
            mid = mid - 1
            fhandle = open(files[mid], 'wb')
            ftp.retrbinary('RETR ' + files[mid], fhandle.write)
            fhandle.close() 
            print mid 
        except Exception as e:
            print e
    elif request.form['play'] == 'dall':
        print 'downloading all audio files for a day locally'
        try:
            for i in files:
                fhandle = open(i, 'wb')
                ftp.retrbinary('RETR ' + i, fhandle.write)
                fhandle.close() 
        except Exception as e:
            print e
    elif request.form['play'] == 'rall':
        print 'removing local audio files'
        try:
            localfiles = [f for f in os.listdir(".") if f.endswith('.wav')]
            for i in localfiles:     
                os.remove(i)
        except Exception as e:
            print e
    
    return render_template('audio.html')

@app.route("/playVideo", methods=['POST'])
def playVideo():
    global ftp
    global files
    global mid
    if request.form['play'] == 'play':
        print "playing"
        try:
            print 'directory: ' , os.getcwd()
            #subprocess.call([path_to_vlc, path_to_files])
            print files[mid]
            subprocess.call('open %s' % files[mid], shell=True)
        except Exception as e:
            print e
    elif request.form['play'] == 'next':
        print 'nexting'
        try:
            mid = mid + 1
            fhandle = open(files[mid], 'wb')
            ftp.retrbinary('RETR ' + files[mid], fhandle.write)
            fhandle.close() 
            print mid 
        except Exception as e:
            print e
            
    elif request.form['play'] == 'back':
        print 'backing'
        try:
            mid = mid - 1
            fhandle = open(files[mid], 'wb')
            ftp.retrbinary('RETR ' + files[mid], fhandle.write)
            fhandle.close() 
            print mid 
        except Exception as e:
            print e
    elif request.form['play'] == 'dall':
        print 'downloading all  video files for a day locally'
        try:
            for i in files:
                fhandle = open(i, 'wb')
                ftp.retrbinary('RETR ' + i, fhandle.write)
                fhandle.close() 
        except Exception as e:
            print e
    elif request.form['play'] == 'rall':
        print 'removing local video files'
        try:
            localfiles = [f for f in os.listdir(".") if f.endswith('.h264')]
            for i in localfiles:     
                os.remove(i)
        except Exception as e:
            print e
    
    return render_template('video.html')


@app.route("/entropy", methods=['POST'])
def entropy():
    global htmlname
    global files
    global ftp
    global curr
    time_length = 142.0
    fps = 30.0
    frame = 900
    frame_no = frame / (time_length * fps)
    if request.form['play'] == 'play':
        print 'browser view'
        try:
            new = 2
            print htmlname
            url = "file://" + os.path.realpath(htmlname)
            webbrowser.open(url, new=new)
        except Exception as e:
            print 'html file not found'
    elif request.form['play'] == 'next':
        print 'nexting'
    elif request.form['play'] == 'back':
        print 'backing'
    elif request.form['play'] == 'dall':
        curr = 0
        print 'downloading all files for a day with their entropy diagrams'
        while curr < int(len(files)):
            try:
                fhandle = open(files[curr], 'ab')  # choose first file of directory for download
                if os.path.getsize(fhandle.name) == 0:  # change conditional to now always be true later
                    print 'not a duplicate'
                    ftp.retrbinary('RETR ' + files[curr], fhandle.write)
                    currfile = files[curr]
                    cap = cv2.VideoCapture(currfile)
                    if cap.isOpened():
                        cap.set(1, frame)
                        ret, newframe = cap.read()
                        # cv2.imshow('window_name', newframe)
                        # gray = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
                        if ret:
                            print 'yea its true'
                            newname = currfile.split(".")[0]
                            finalname = newname + '_frame_' + str(frame) + '.png'
                            cv2.imwrite(finalname, newframe)
                            '''
                            Credit to  Johannes Maucher
                            at https://www.hdm-stuttgart.de/~maucher
                            for the following code
                            '''
                            colorIm = Image.open(finalname)
                            greyIm = colorIm.convert('L')
                            colorIm = np.array(colorIm)
                            greyIm = np.array(greyIm)
                            N = 5
                            S = greyIm.shape
                            E = np.array(greyIm)
                            for row in range(S[0]):
                                for col in range(S[1]):
                                    Lx = np.max([0, col - N])
                                    Ux = np.min([S[1], col + N])
                                    Ly = np.max([0, row - N])
                                    Uy = np.min([S[0], row + N])
                                    region = greyIm[Ly:Uy, Lx:Ux].flatten()
                                    E[row, col] = entropy(region)
                            grayImage = cv2.applyColorMap(greyIm, cv2.COLOR_BGR2GRAY)
                            entropyImage = cv2.applyColorMap(greyIm, cv2.COLORMAP_JET)
                            cv2.imwrite('gray_' + finalname, greyIm)
                            # cv2.imwrite('color_' + finalname, entropyImage)
                            a = np.empty_like(E)
                            a[:, :] = E
                            a = np.flipud(a)
                            fig = plt.figure()
                            plt.imshow(a, cmap=plt.get_cmap("jet"), origin="lower")
                            plt.xlabel('Entropy in 10x10 neighborhood')
                            plt.colorbar()
                            plt.savefig('color_' + finalname, bbox_inches='tight')
                            plt.imshow(E, cmap=plt.get_cmap("jet"), origin="lower")
                            plt.plot()
                            htmlname = newname + '_frame_' + str(frame) + '.html'
                            mpld3.save_html(fig, htmlname)
                            #mpld3.fig_to_html(fig, template_type="simple")
                            print 'finished!'
                    cap.release()
                    print 'finished ', curr
                else:
                    print 'duplicate'
                curr += 1

            except Exception as e:
                print e
        #return render_template('entropy.html')
    elif request.form['play'] == 'rall':
        try:
            localfiles = [f for f in os.listdir(".") if f.endswith('.h264') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.html')]
            for i in localfiles:
                os.remove(i)
        except Exception as e:
            print e
    return render_template('entropy.html')

'''
Credit goes to Johannes Maucher from 
https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html
'''    
def entropy(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
        symset=list(set(signal))
        numsym=len(symset)
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent

def to_hex(date, time):
    time = time.replace('-', ':')
    d = Dates.to_hex(date, time)
    return d

def make_hex8(hex_num):
    """
    Pads end of hex with 0s to make it length 8.
    :param hex_num: Number to be padded
    :return: padded hex number
    """
    for i in range(0, 8 - len(hex_num)):
        hex_num += "0"
    return hex_num


@app.route('/stop', methods=['POST'])     
def close():   
    global a
    a.close()
    

@app.route("/showSignUp")
def signUp():
    return render_template('signup.html')


if __name__ == "__main__":
    app.run()
    #threaded="True"