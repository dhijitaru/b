import numpy as np
import cv2
from matplotlib import pyplot as plt
%matplotlib notebook 

image = cv2.imread('IMG_3776.JPG',0)
clicklog = np.zeros(image.shape)

fimage = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(fimage)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

dft_ishift = np.fft.ifftshift(dft_shift)
img = cv2.idft(dft_ishift)
img = cv2.magnitude(img[:,:,0],img[:,:,1])

def addClick(x,y,click):
    A = click
    A[y,x] = 1
    return A

def mkWave(x, y):
    a = np.zeros(image.shape)
    a[y, x] = 1
    wave = cv2.idft(a)
    return wave

def reset(x, y, A):
    b = np.zeros(dft_shift.shape)
    b[:,:,0] = A 
    b[:,:,1] = A 
    c = dft_shift*b
    c = np.fft.ifftshift(c)
    c = cv2.idft(c)
    re = cv2.magnitude(c[:,:,0],c[:,:,1])
    return re

def onclick(event):
    global clicklog, X, Y
    if event.button==1:
        X=int(round(event.xdata))
        Y=int(round(event.ydata))
        
        ax7.plot(X, Y, marker='.', markersize='1')    
        
        clicklog = addClick(X, Y, clicklog)

def onkey(event):
    if event.key == 'q':
        sys.exit()

def onrelease(event):
    global clicklog, X, Y
    wave = mkWave(X, Y)
    ax6.imshow(wave, cmap='gray')
    
    reim = reset(X, Y, clicklog)
    reim = np.float32(reim)
    ax4.imshow(reim, cmap='gray')
    
    plt.draw()

fig = plt.figure(figsize=(9,4))

ax1 = fig.add_subplot(2,3,1)
ax1.imshow(image, cmap='gray')
ax1.set_title('Input Image')
ax1.set_xticks([]), ax1.set_yticks([])

ax2 = fig.add_subplot(2,3,2)
ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.set_title('Magnitude Spectrum')
ax2.set_xticks([]), ax2.set_yticks([])

ax3 = fig.add_subplot(2,3,3)
ax3.imshow(img, cmap='gray')
ax3.set_title('IFFT')
ax3.set_xticks([]), ax3.set_yticks([])

ax4 = fig.add_subplot(2,3,4)

ax6 = fig.add_subplot(2,3,6)

fig2=plt.figure(figsize=(8,4))

ax7 = fig2.add_subplot(1,1,1)
ax7.imshow(magnitude_spectrum, cmap='gray')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0)

fig2.canvas.mpl_connect('button_press_event', onclick)
fig2.canvas.mpl_connect('motion_notify_event', onclick)
fig2.canvas.mpl_connect('button_release_event', onrelease)
fig2.canvas.mpl_connect('key_press_event', onkey)

説明
画像の読み込み
clicklogの初期化
フーリエ変換
逆フーリエ変換
過去のクリックした座標のデータを1、それ以外を0で記録
sin波の切り抜き
sin波の合成
クリック時の処理
左クリック&ドラッグ時の挙動
qを押したら終了
指を離したら計算を実行する(高負荷を回避, リアルタイムではなくなる)
画像を表示

使い方　コードをジュピターに載せる
実行の仕方　figure2の画像を左クリックorドラッグ
ライブラリとバージョン
numpy                     1.16.4           py37h19fb1c0_0
matplotlib                3.1.0            py37hc8f65d3_0

参考
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
https://www.hello-python.com/2018/02/16/numpy%E3%81%A8opencv%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E7%94%BB%E5%83%8F%E3%81%AE%E3%83%95%E3%83%BC%E3%83%AA%E3%82%A8%E5%A4%89%E6%8F%9B%E3%81%A8%E9%80%86%E5%A4%89%E6%8F%9B/

実行の様子
