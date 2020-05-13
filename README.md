# Image-Stitching
https://eric033014.github.io/Image-Stitching/

## Enter Work Space & Create Cmake Work Space
git clone https://github.com/eric033014/Image-Stitching.git <br>
or <br>
git clone git@github.com:eric033014/Image-Stitching.git

cd Image-Stitching

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

./stitch [test_image資料內的資料集名稱]

## Result
結果會在 Image-Stitching/result內，產生一張合成jpg


# 0.開發環境
Mac OS CML <br>
vim <br>
c++ 11 <br>
opencv4 <br>

# 1.拍攝照片集
我使用的是Canon 700D進行拍攝6張照片為一組。

# 2.紀錄照片名稱
將快門速度紀錄在每組測試資料的Image-Stitching/test_image/[測試資料集名稱]/setting.txt上。<br>
格式為<br>
圖片名稱<br>
圖片名稱<br>
圖片名稱<br>

# 3.實作SIFT特徵點演算法
此次專題是利用上課提到的SIFT演算法，實作的過程參考了論文的內容，<br>
這部分比較沒有遇到問題，因為內容比較完整，可以參考的東西也比較多。<br>

# 4.計算圖片焦距
這部分其實算是多做的，由於我自己在整理照片時，有將照片進行裁切以及縮放的變化，<br>
才會導致原先的焦距計算有誤，需要再利用opencv內建的estimate focal的功能進行預測，<br>
進而推測出該有的焦距。<br>

# 5.匹配特徵點
先將descriptor的格位轉為openCV的格式後，再把outlier移除，<br>
以免影響到匹配的結果，移除的方法大致上利用最小距離的兩倍為throshold，<br>
使我們可以只留下優良的點。<br>

# 6.Image-Warping
參考了講義的Cylindrical warping的公式，Warping的方式是使用Focal length去做計算，<br>
得知焦距後，便可以知道該投影在圓柱體上的相對位置，以得到較良好的縫合結果。<br>

# 7.Reference
David G. Lowe, Distinctive Image Features from Scale-Invariant Keypoints. Computer Science Department of University of British Columbia. 2004 <br>
課堂使用之PPT
