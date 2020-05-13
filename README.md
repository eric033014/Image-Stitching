# High-Dynamic-Range-Imaging
https://eric033014.github.io/High-Dynamic-Range-Imaging/

## Enter Work Space & Create Cmake Work Space
git clone https://github.com/eric033014/High-Dynamic-Range-Imaging.git <br>
or <br>
git clone git@github.com:eric033014/High-Dynamic-Range-Imaging.git

cd High-Dynamic-Range-Imaging

mkdir [cmake work space name]

ex:
mkdir build

cd [cmake work space name]

ex:
cd build

## Build & Compile

cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

ninja

## Execute
請於<cmake work space name>底下執行

./hdr [test_image資料內的資料集名稱]

## Result
結果會在 High-Dynamic-Range-Imaging/result內，產生一張hdr以及一張jpg


# 0.開發環境
Mac OS CML <br>
vim <br>
c++ 11 <br>
opencv4 <br>

# 1.拍攝照片集
我使用的是Canon 700D進行拍攝16張照片為一組。

# 2.紀錄快門速度
將快門速度紀錄在每組測試資料的High-Dynamic-Range-Imaging/test_image/[測試資料集名稱]/setting.txt上。<br>
格式為<br>
圖片名稱 曝光時間<br>
圖片名稱 曝光時間<br>
圖片名稱 曝光時間<br>

# 3.校準照片集
由於自行拍攝的照片，時常會有抖動(按壓快門時的移動或是快門本身的震動導致)，若直接使用該組照片進行HDR影像建置，產生的影像會有重疊的影像，類似鬼影的感覺，<br>
會造成結果不理想，因此需要一些簡單的Alignment進行校正，這次使用的是上課有教導的內容 - Median Threshold Bitmap進行校正。<br>
此演算法利用遞迴的方式進行，並利用影像金字塔的方式(長寬各縮小至1/2倍，每層皆把圖片縮小至1/4倍)，由金字塔最底層的圖片開始，<br>
往{(-1, -1),(-1, 0), (-1, 1), (0,-1), (0, 0), (0, 1), (1,-1), (1, 0), (1, 1)}，計算這幾個方向所造成的誤差，並往誤差最小的方向進行校正。<br>

# 4.建立HDR影像
使用上課所介紹之Debevec的方法，並以C++進行編寫，先利用論文中的Gslove搭配opencv提供之Slover解出方程式的解，並得到Recovering response curve，<br>
再將每個影像輸入得到值後，乘上權重，最後得到Radiance Map，然後輸出成hdr檔案格式。<br>
我有對於取樣的縮小的大小做額外的測試，測試的為10X10，20X20，30X30，40X40，<br>
尺寸越大，處理時間理所當然的會更久，產生的圖片對於顏色有些微差距，應該是縮小圖片，每塊大小不同的區塊，平均出來的亮度也不太一樣所導致的。<br>

# 5.Tone Mapping
由於一般的圖片r,g,b的值都介於0 ~ 255，而hdr image一開始產生的圖片有些點會不在此範圍，因此需要進行Tone Mapping，<br>
這次選擇使用Read List中提供的論文 - Fast Bilateral Filtering for the Display of High Dynamic Range Images進行實作，<br>
利用Radiance Map可以計算出其intensity，並且我在作者的個人網頁中有找到詳細的實作內容，大致如下：<br>
input intensity = 0.212671 * R + 0.715160 * G + 0.072169 * B;<br>
r=R/(input intensity), g=G/input intensity, B=B/input intensity<br>
log(base) = Bilateral(log(input intensity))<br>
log(detail) = log(input intensity) - log(base)<br>
log(output intensity) = log(base) * compressionfactor + log(detail) - log(absolute_scale)<br>
R output = r*10^(log(output intensity)), G output = g*10^(log(output intensity)), B output = b*10^(log(output intensity))<br>
進行Tone Mapping完後，輸出成jpg檔案格式。<br>

# 6.結果
![image]()
第一組圖的效果不是很好，因為相機設定了大光圈，導致有景深效果，背景的內容全部都模糊掉了，加上我在調整快門時，似乎有移動到相機，導致最後出的圖片，似乎有點模糊。
但我已經使用了MTB，讓模糊減少了許多。
![image]()
第二組圖的部分，我選擇了亮暗對比明顯一些的場景進行拍攝，結果有比第一組圖更好的HDR圖片。

# 7.Reference
Paul E. Debevec, Jitendra Malik, Recovering High Dynamic Range Radiance Maps from Photographs, SIGGRAPH 1997 <br>
Fredo Durand, Julie Dorsey, Fast Bilateral Filtering for the Display of High Dynamic Range Images, SIGGRAPH 2002 <br>
Reinhard, E., Stark, M., Shirley, P., and Ferwerda, J.A. Photographic tone reproduction for digital images, SIGGRAPH 2002 <br>
課堂使用之PPT(MTB, HDR, Tone mapping)
