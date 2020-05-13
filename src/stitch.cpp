#include "stitch.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2//stitching.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
using namespace cv;
using namespace std;

bool cmp(featurePoints a, featurePoints b) {
    return (a.scale_subl < b.scale_subl);
}
//
// STITCH::STITCH() {
//
// }

vector<Mat> STITCH::getInputImage(string testing_set) {
    vector<Mat> inputArrays;
    cout << "getInputImage" << endl;
    for(int i = 0; i < file_name_list.size();i++){
        Mat temp = imread("../test_image/" + testing_set + "/" + file_name_list[i], IMREAD_COLOR);
        inputArrays.push_back(temp);
    }
    cout << "圖片張數：" << inputArrays.size() << endl;
    return inputArrays;
}

void STITCH::getFileName(string testing_set) {
    cout << "getExposureTime" << endl;

    fstream file;
    string name;
    double time;
    file.open("../test_image/" + testing_set + "/setting.txt",ios::in);
    if(!file) {    //檢查檔案是否成功開啟
        cerr << "Can't open file!\n";
        exit(1);     //在不正常情形下，中斷程式的執行
    }

    while(file >> name) {
        // cout << setw(4) << setiosflags(ios::right) << name << setw(8) << setiosflags(ios::right) << time << endl;
        file_name_list.push_back(name);
    }

}


void STITCH::estimate(vector<vector<featurePoints>> &f, vector<Size> &pic_size, vector<vector<DMatch>> &good_matches, vector<double> &focal)
{
    //Convert our features to OpenCV's image features.
    vector<detail::ImageFeatures> ifs;
    toCVImageFeatures(f, pic_size, ifs);
    // std::cout <<  << '\n';
    // Find focal length match infos
    vector<detail::MatchesInfo> infos;
    Ptr<detail::FeaturesMatcher> matcher = new detail::BestOf2NearestMatcher(false);
    (*matcher)(ifs, infos);

    //Estimate focals
    detail::estimateFocal(ifs, infos, focal);

    // Find matching points
    good_matches.clear();
    good_matches.resize(infos.size());
    cout << "infos.size() " << infos.size() << endl;
    FlannBasedMatcher fmatcher;
    for(int j = 0; j < ifs.size() - 1; j++)
    {
        cout << j << endl;
        detail::ImageFeatures &m1 = ifs[j];
        detail::ImageFeatures &m2 = ifs[j + 1];

        vector<DMatch> matches;
        std::cout << m1.descriptors << '\n';
        std::cout << m2.descriptors << '\n';
        fmatcher.match(m1.descriptors, m2.descriptors, matches);

        double min_dist = 100.0;
        for( int i = 0; i < matches.size(); i++ )
        {
            cout << matches[i].distance << endl;
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
        }

        //Eliminate outliers
        for( int i = 0; i < matches.size(); i++ )
        {
            if( matches[i].distance <= max(2 * min_dist, 0.2) )
            {
                good_matches[j].push_back(matches[i]);
            }
        }
    }
}

void STITCH::toCVImageFeatures(vector<vector<featurePoints>> &f, vector<Size> &pic_size, vector<detail::ImageFeatures> &ifs)
{
    ifs.clear();
    ifs.resize(f.size());

    for(int i = 0; i < f.size(); i++)
    {
        cv::detail::ImageFeatures &imgf = ifs[i];
        std::cout << imgf.descriptors.cols << imgf.descriptors.rows << imgf.descriptors.channels() << '\n';
        Mat mat = imgf.descriptors.getMat( ACCESS_READ );
        toCVDescriptor(f[i], mat);
        imgf.descriptors = mat.getUMat(ACCESS_READ );
        imgf.img_idx = i;
        imgf.img_size = pic_size[i];
        imgf.keypoints.resize(f[i].size());

        for(int j = 0; j < f[i].size(); j++)
        {
            cv::KeyPoint &kpt = imgf.keypoints[j];
            featurePoints &ff = f[i][j];
            kpt.pt = cv::Point2f(ff.xHat[0], ff.xHat[1]);
            kpt.octave = ff.l + (ff.subl << 8) + (cvRound((ff.xHat[2] + 0.5) * 255) << 16);
            kpt.size = ff.scale_subl;
            kpt.response = ff.response;
            kpt.angle = (ff.orien * 180) / M_PI;
        }
        std::cout << imgf.descriptors.cols << imgf.descriptors.rows << imgf.descriptors.channels() << '\n';

    }
}


void STITCH::toCVDescriptor(vector<featurePoints> &f, Mat &d)
{
    int hz = f[0].h.size();
    int hy = f[0].h[0].size();
    int hx = f[0].h[0][0].size();
    std::cout << hz << " " << hy << " " << hz << '\n';
    Mat descriptor((int)f.size(), (int)(hz * hy * hx), CV_32F);
    for(int j = 0; j < f.size(); j++)
    {
        for(int z = 0; z < hz; z++)
            for(int y = 0; y < hy; y++)
                for(int x = 0; x < hx; x++)
                {
                    descriptor.at<float>(j, z * hy * hx + y * hx + x) = f[j].h[z][y][x];
                }
    }

    d.release();
    d = descriptor.clone();
}

void STITCH::drawMatches(Mat &img1, vector<featurePoints> &f1, Mat &img2, vector<featurePoints> &f2, vector<DMatch> good_matches)
{

    int w = img1.cols + img2.cols;
    int h = img1.rows > img2.rows ? img1.rows : img2.rows;
    Mat plate(h, w, CV_8UC3);

    for(int j = 0; j < plate.rows; j++)
    {
        for(int i = 0; i < img1.cols; i++)
        {
            for(int c = 0; c < 3; c++)
            {
                plate.at<Vec3b>(j, i)[c] = img1.at<Vec3b>(j, i)[c];
                plate.at<Vec3b>(j, i + img1.cols)[c] = img2.at<Vec3b>(j, i)[c];
            }
        }
    }

    for(int i = 0; i < good_matches.size(); i++)
    {
        DMatch& m = good_matches[i];

        Point one = Point(f1[m.queryIdx].xHat[0], f1[m.queryIdx].xHat[1]);
        Point two = Point(f2[m.trainIdx].xHat[0] + img1.cols, f2[m.trainIdx].xHat[1]);
        circle(plate, one, 2, Scalar(0, 0, 255), -1, 8);
        circle(plate, two, 2, Scalar(0, 0, 255), -1, 8);
        line(plate, one, two, Scalar(rand() % 256, rand() % 256, rand() % 256), 2, 8);
    }
    imwrite("sift.png", plate);
}

void STITCH::alignMatches(Mat &img1, vector<featurePoints> &f1, Mat &img2, vector<featurePoints> &f2, vector<DMatch> good_matches,
                 vector<int> &x,vector<int> &y,double FL1,double FL2)
{
    int cal_dx = 0;
    int cal_dy = 0;
    double avg_dx = 0 ;
    double avg_dy = 0;
    cout << "good matches" << good_matches.size() << endl;
    for(int i = 0; i < good_matches.size(); i++)
    {
        DMatch& m = good_matches[i];
        cout << good_matches.size() << "/" << i  << endl;
        if(i == good_matches.size()-1)
        {

            avg_dx = cal_dx/(good_matches.size()-1);
            x.push_back(int(avg_dx));
            avg_dy = cal_dy/(good_matches.size()-1);
            y.push_back(int(avg_dy));
            //qDebug()<<avg_dx<<avg_dy;
        }
        else
        {
            int mid_x1 = img1.cols/2;
            int mid_x2 = img2.cols/2;
            int mid_y1 = img1.rows/2;
            int mid_y2 = img2.rows/2;

            double fL1 = FL1;
            double theta1 = atan((f1[m.queryIdx].xHat[0]-mid_x1)/fL1);
            double h1 = (f1[m.queryIdx].xHat[1]-mid_y1)/pow(pow((f1[m.queryIdx].xHat[0]-mid_x1),2)+pow(fL1,2),0.5);
            int x1 = fL1*theta1+mid_x1;
            int y1 = fL1*h1+mid_y1;


            double fL2 = FL2;
            double theta2 = atan((f2[m.trainIdx].xHat[0]-mid_x2)/fL2);
            double h2 = (f2[m.trainIdx].xHat[1]-mid_y2)/pow(pow((f2[m.trainIdx].xHat[0]-mid_x2),2)+pow(fL2,2),0.5);
            int x2 = fL2*theta2+mid_x2+img1.cols;
            int y2 = fL2*h2+mid_y2;
            int distance = x2 - x1;
            int distancey1 = img1.rows-y1+y2;

            cal_dx = cal_dx+distance;

                cal_dy = cal_dy+distancey1;

        }
    }
}

void STITCH::DOG(Mat inputArray, vector<Mat> &_dogs, vector<Mat> &_gpyr, const double _k, const double _sig)
{
    cout << "DOG" << endl;
    if(inputArray.type() == CV_8UC3)
    {
        cvtColor(inputArray, inputArray, COLOR_BGR2GRAY);
    }

    if(inputArray.type() == CV_8UC1)
    {
        inputArray.convertTo(inputArray, CV_32FC1);
    }

    vector<double> si;
    si.push_back(_sig);

    int i = 0;
    for(i = 1; i < (int)(s + 3); i++)
    {
        double prev = pow(_k, (double)(i - 1)) * _sig;
        double news = prev * _k;
        si.push_back(sqrt(news * news - prev * prev));
    }
    si[0] = sqrt(_sig * _sig - 0.25);

    for(int b = 0; b < (int)ocv; b++) // 3
    {
        for(int a = 0; a < (int)(s + 3); a++) // 6
        {
            if(a == 0 && b == 0)
            {
                GaussianBlur(inputArray, _gpyr[0], Size(0, 0), si[a], si[a]);
            }
            else if(a == 0)
            {
               resize(_gpyr[(b - 1) * (s + 3) + s], _gpyr[b * (s + 3) + a], Size(_gpyr[(b - 1) * (s + 3) + s].cols / 2.0, _gpyr[(b - 1) * (s + 3) + s].rows / 2.0), 0, 0, INTER_NEAREST);
            }
            else
            {
                GaussianBlur(_gpyr[b * (s + 3) + a - 1], _gpyr[b * (s + 3) + a], Size(0, 0), si[a], si[a]);
                Mat tmp(_gpyr[b * (s + 3) + (a - 1)].rows, _gpyr[b * (s + 3) + (a - 1)].cols, CV_32FC1);

                #pragma omp parallel for private(i)
                for (int j = 0; j < tmp.rows; j++)
                {
                    for (i = 0; i < tmp.cols; i++)
                    {
                        tmp.at<float>(j, i) = _gpyr[b * (s + 3) + a].at<float>(j, i) - _gpyr[b * (s + 3) + a - 1].at<float>(j, i);
                    }
                }

                //Build the pyramid
                _dogs.push_back(tmp.clone());
            }
        }
    }
}

//SIFT main function
void STITCH::SIFT(Mat &inputArray, vector<featurePoints> &_features)
{
    vector<Mat> gpyr((int)(ocv * (s + 3)));
    vector<Mat> dogs;

    DOG(inputArray, dogs, gpyr, k, sig);

    featurePoints f;

    vector<featurePoints> siftFeatures;
    for(int b = 0; b < ocv; b++)
        for(int a = 1; a <= s; a++)
        {
            for(int j = 1; j < dogs[(a) + b * (s + 2)].rows - 1; j++)
                for(int i = 1; i < dogs[(a) + b * (s + 2)].cols - 1; i++)
                {
                    //Threshold of half of 0.03
                    if(abs(dogs[(a) + b * (s + 2)].at<float>(j, i)) > cvFloor(0.5 * 0.03 / s * 255))
                        if(isExtreme(dogs, b, a, j, i))
                        {
                            if(interp(dogs, b, a, j, i, f))
                            {
                                siftFeatures.push_back(f);
                            }
                        }

                }
        }


    orien(siftFeatures, gpyr);

    descriptor(siftFeatures, gpyr);

    //Copy features to output vector
    _features.clear();
    _features.resize(siftFeatures.size());
    for(int i = 0; i < siftFeatures.size(); i++)
    {
        _features[i] = siftFeatures[i];
    }
}

bool STITCH::isExtreme(vector<Mat> dogs, int _l, int _sl, int j, int i)
{
    int status = dogs[(_sl) + _l * (s + 2)].at<float>(j, i) > dogs[(_sl + 1) + _l * (s + 2)].at<float>(j, i) ? 2 : 1;

    if(status == 1)
    for(int y = -1; y <= 1; y++)
    {
        for(int x = -1; x <= 1; x++)
        {
            int&& tx = i + x;
            int&& ty = j + y;
            if((tx >= 0 && tx < dogs[(_sl) + _l * (s + 2)].cols && ty >= 0 && ty < dogs[(_sl) + _l * (s + 2)].rows))
            {
                for(int z = -1; z <= 1; z++)
                {
                    if(dogs[(_sl) + _l * (s + 2)].at<float>(j, i) > dogs[(_sl + z) + _l * (s + 2)].at<float>(ty, tx))
                    {
                        return false;
                    }
                }
            }
        }
    }
    else if(status == 2)
    for(int y = -1; y <= 1; y++)
    {
        for(int x = -1; x <= 1; x++)
        {
            int&& tx = i + x;
            int&& ty = j + y;
            if((tx >= 0 && tx < dogs[(_sl) + _l * (s + 2)].cols && ty >= 0 && ty < dogs[(_sl) + _l * (s + 2)].rows))
            {
                for(int z = -1; z <= 1; z++)
                {
                    if(dogs[(_sl) + _l * (s + 2)].at<float>(j, i) < dogs[(_sl + z) + _l * (s + 2)].at<float>(ty, tx))
                    {
                        return false;
                    }
                }
            }
        }
    }
    else
    {
        return false;
    }

    return true;
}


bool STITCH::interp(vector<Mat> _dog, int _layer, int _sublayer, int j, int i, featurePoints &f)
{
    // cout << "interp" << endl;
    int a = 0;
    Mat x;

    //Scale extreme point location
    int interpnum = 5;
    for(a = 0; a < interpnum; a++)
    {
        x = xHat(_dog, _layer, _sublayer, j, i);
        // 0.5 from Lowe's paper
        if(abs(x.at<double>(0, 0)) < 0.5 && abs(x.at<double>(1, 0)) < 0.5 && abs(x.at<double>(2, 0)) < 0.5)
            break;

        i += cvRound(x.at<double>(0, 0));
        j += cvRound(x.at<double>(1, 0));
        _sublayer += cvRound(x.at<double>(2, 0));


        if(_sublayer < 1 || _sublayer > (int)(s)|| j >= _dog[_layer * (s + 2) + _sublayer].rows - 1
                                                || i >= _dog[_layer * (s + 2) + _sublayer].cols - 1
                                                || j < 1 || i < 1)
        {
            return false;
        }
    }
    //If fail to find the precise location
    if(a >= interpnum)
        return false;

    //Count the contrast threshold
    Mat D2(1, 1, CV_64FC1);
    Mat &&dD = diff(_dog, _layer, _sublayer, j, i);
    gemm(dD, x, 1, NULL, 0, D2, GEMM_1_T);
    double thresh = (double)_dog[_layer * (s + 2) + _sublayer].at<float>(j, i) * norming + D2.at<double>(0, 0) * 0.5;

    if(abs(thresh) * s < 0.03)
    {
        return false;
    }

    //Remove if is not edge
    if(!removeEdge(_dog[_layer * (s + 2) + _sublayer], j, i))
        return false;

    //Write data into feature point structure
    f.x = i;
    f.y = j;
    f.l = _layer;
    f.subl = _sublayer;
    f.xHat[0] = (i + x.at<double>(0, 0)) * pow(2.0, _layer);
    f.xHat[1] = (j + x.at<double>(1, 0)) * pow(2.0, _layer);
    f.xHat[2] = x.at<double>(2, 0);

    double _sub = f.subl + f.xHat[2];
    f.scale_subl = sig * pow(2.0, f.l + _sub / s);
    f.scale = sig * pow(2.0, _sub / s);
    f.response = abs(thresh);

    return true;
}

Mat STITCH::xHat(vector<Mat> _dog, int _layer, int _sublayer, int j, int i)
{
    Mat &&dD = diff(_dog, _layer, _sublayer, j, i);
    Mat &&dH = hessian(_dog, _layer, _sublayer, j, i);

    Mat invdH(3, 3, CV_64FC1);
    invert(dH, invdH, DECOMP_SVD);

    Mat x(3, 1, CV_64FC1);
    gemm(invdH, dD, -1, NULL, 0, x);

    return x;
}

Mat STITCH::diff(vector<Mat> _dog, int _layer, int _sublayer, int j, int i)
{
    double dx, dy, ds;
    dx = ((double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j, i + 1)) - (double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j, i - 1))) / 2.0 * norming;
    dy = ((double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j + 1, i)) - (double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j - 1, i))) / 2.0 * norming;
    ds = ((double)(_dog[_layer * (s + 2) + _sublayer + 1].at<float>(j, i)) - (double)(_dog[_layer * (s + 2) + _sublayer - 1].at<float>(j, i))) / 2.0 * norming;

    Mat result(3, 1, CV_64FC1);
    result.at<double>(0, 0) = dx;
    result.at<double>(1, 0) = dy;
    result.at<double>(2, 0) = ds;

    return result;
}

Mat STITCH::hessian(vector<Mat> _dog, int _layer, int _sublayer, int j, int i)
{
    double c, dxx, dxy, dyy, dys, dss, dxs;
    c = (double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j, i));
    dxx = ((double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j, i + 1)) + (double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j, i - 1)) - 2 * c) * norming;
    dxy = ((double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j + 1, i + 1)) + (double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j - 1, i - 1))
          -(double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j - 1, i + 1)) - (double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j + 1, i - 1))) / 4.0 * norming;
    dyy = ((double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j + 1, i)) + (double)(_dog[_layer * (s + 2) + _sublayer].at<float>(j - 1, i)) - 2 * c) * norming;
    dys = ((double)(_dog[_layer * (s + 2) + _sublayer + 1].at<float>(j + 1, i)) + (double)(_dog[_layer * (s + 2) + _sublayer - 1].at<float>(j - 1, i))
          -(double)(_dog[_layer * (s + 2) + _sublayer - 1].at<float>(j + 1, i)) - (double)(_dog[_layer * (s + 2) + _sublayer + 1].at<float>(j - 1, i))) / 4.0 * norming;
    dss = ((double)(_dog[_layer * (s + 2) + _sublayer + 1].at<float>(j, i)) + (double)(_dog[_layer * (s + 2) + _sublayer - 1].at<float>(j, i)) - 2 * c) * norming;
    dxs = ((double)(_dog[_layer * (s + 2) + _sublayer + 1].at<float>(j, i + 1)) + (double)(_dog[_layer * (s + 2) + _sublayer - 1].at<float>(j, i - 1))
          -(double)(_dog[_layer * (s + 2) + _sublayer + 1].at<float>(j, i - 1)) - (double)(_dog[_layer * (s + 2) + _sublayer - 1].at<float>(j, i + 1))) / 4.0 * norming;

    Mat result(3, 3, CV_64FC1);
    result.at<double>(0, 0) = dxx;
    result.at<double>(0, 1) = dxy;
    result.at<double>(0, 2) = dxs;
    result.at<double>(1, 0) = dxy;
    result.at<double>(1, 1) = dyy;
    result.at<double>(1, 2) = dys;
    result.at<double>(2, 0) = dxs;
    result.at<double>(2, 1) = dys;
    result.at<double>(2, 2) = dss;

    return result;
}

bool STITCH::removeEdge(Mat _dogImg, int j, int i)
{
    double c, dxx, dxy, dyy;
    c = (double)(_dogImg.at<float>(j, i));
    dxx = ((double)(_dogImg.at<float>(j, i + 1)) + (double)(_dogImg.at<float>(j, i - 1)) - 2 * c) * norming;
    dxy = ((double)(_dogImg.at<float>(j + 1, i + 1)) + (double)(_dogImg.at<float>(j - 1, i - 1))
          -(double)(_dogImg.at<float>(j - 1, i + 1)) - (double)(_dogImg.at<float>(j + 1, i - 1))) / 4.0 * norming;
    dyy = ((double)(_dogImg.at<float>(j + 1, i)) + (double)(_dogImg.at<float>(j - 1, i)) - 2 * c) * norming;

    double tr = dxx + dyy;
    double dH = dxx * dyy - (dxy * dxy);

    if(dH <= 0 || tr * tr / dH >= pow(11.0, 2) / 10.0)
        return false;

    return true;
}

void STITCH::orien(vector<featurePoints> &f, vector<Mat> &_gpyr)
{

  int fs = f.size();

  int dir;
  double weight;
  vector<featurePoints> tmp;


  for(int a = 0; a < fs; a++)
  {
    // 36 directions
    vector<double> h(36);

    int radius = cvRound(1.5 * 3.0 * f[a].scale);
    double _sig = 2.0 * pow(1.5 * f[a].scale, 2);

    for(int j = - radius; j <= radius; j++)
      for(int i = - radius; i <= radius; i++)
      {
          int tx = f[a].x + i;
          int ty = f[a].y + j;
          if(ty >= 1 && ty < _gpyr[f[a].l * (s + 3) + f[a].subl].rows - 1 && tx >= 1 && tx < _gpyr[f[a].l * (s + 3) + f[a].subl].cols - 1)
          {
            double dx = _gpyr[f[a].l * (s + 3) + f[a].subl].at<float>(ty, tx + 1) - _gpyr[f[a].l * (s + 3) + f[a].subl].at<float>(ty, tx - 1);
            double dy = _gpyr[f[a].l * (s + 3) + f[a].subl].at<float>(ty - 1, tx) - _gpyr[f[a].l * (s + 3) + f[a].subl].at<float>(ty + 1, tx);
            double &&mag = sqrt(dx * dx + dy * dy);
            double &&the = atan2(dy , dx);

            if(the >= M_PI * 2){the -= (M_PI * 2);}
            if(the < 0){the += (M_PI * 2);}

            weight = exp( - ( j * j + i * i ) / _sig );
            dir = cvRound((h.size() / 360.0f) * ((the * 180.0) / M_PI));

            if(dir >= h.size()){dir -= h.size();}
            if(dir < 0){ dir += h.size();}

            h[dir] += weight * mag;
          }
      }

    //Apply Gaussian of 3x3 mask
    double prev = h[h.size() - 1], next;
    vector<double> t(h.size());
    for(int i = 0; i < h.size(); i++)
    {
        i + 1 >= h.size() ? next = h[0] : next = h[i + 1];
        t[i] = 0.25 * prev + 0.5 * h[i] + 0.25 * next;
        prev = h[i];
    }
    int i = 0;

#pragma omp parallel for
    for(i = 0; i < h.size(); i++)
    {
        h[i] = t[i];
    }
    t.clear();

    double val = h[0];
    for(int i = 1; i < h.size(); i++)
    {
        if(h[i] > val)
        {
            val = h[i];
        }
    }

    int x, y;

    //Find 0.8 * peak of the orientation. 15% of points will have more than one orientation features
    for(int i = 0; i < h.size(); i++ )
    {
        x = (i == 0            ? h.size() - 1 : i - 1);
        y = (i == h.size() - 1 ? 0            : i + 1);

        if(h[i] > h[x]  &&  h[i] > h[y]  &&  h[i] >= val * 0.8)
        {
            double ddir = i + (0.5 * (h[x] - h[y]) / (h[x] - 2 * h[i] + h[y]));
            if(ddir < 0){ddir = h.size() + ddir;}
            if(ddir >= h.size()){ddir = ddir - h.size();}
            featurePoints t = f[a];
            t.orien = (360.0 - (ddir * (360.0 / h.size()))) * M_PI / 180.0;
            tmp.push_back(t);
        }
    }
  }

  f.clear();
  for(int i = 0; i < tmp.size(); i++)
  {
      f.push_back(tmp[i]);
  }
}

void STITCH::descriptor(vector<featurePoints> &f, vector<Mat> &_gpyr)
{
    //8 orientation, 4 * 4 descriptore (Lowe Sec 6.2)
    const int bins = 8;
    int fs = f.size();
    for(int a = 0; a < fs; a++)
    {
        Mat& img = _gpyr[f[a].l * (s + 3) + f[a].subl];
        double h_w = 3.0 * f[a].scale_subl;
        double w = img.cols;
        double h = img.rows;
        int radius = min(cvRound(h_w * sqrt(2) * (4 + 1.0) * 0.5), (int)sqrt(w * w + h * h));
        double _sig = 4 * 4 * 0.5;

        double sin_t = sin(M_PI * 2 - f[a].orien);
        double cos_t = cos(M_PI * 2 - f[a].orien);

        for(int j = -radius; j <= radius; j++)
            for(int i = -radius; i <= radius; i++)
            {
                double j_rot = (i * sin_t + j * cos_t) / h_w; //i_rot, j_rot converts the 36 bin direction to 8 bin
                double i_rot = (i * cos_t - j * sin_t) / h_w;
                double i_bin = i_rot + 4 / 2 - 0.5;  //i_bin, j_bin is the 4 * 4 matrix position
                double j_bin = j_rot + 4 / 2 - 0.5;

                if(j_bin > -1.0 && j_bin < 4 && i_bin > -1.0 && i_bin < 4)
                {
                    int tx = f[a].x + i;
                    int ty = f[a].y + j;
                    if(ty > 0 && ty < img.rows - 1 && tx > 0 && tx < img.cols - 1)
                    {
                        //Calculate rientation
                        double dx = img.at<float>(ty, tx + 1) - img.at<float>(ty, tx - 1);
                        double dy = img.at<float>(ty - 1, tx) - img.at<float>(ty + 1, tx);
                        double &&mag = sqrt(dx * dx + dy * dy);
                        double &&the = atan2(dy , dx);

                        if(the >= M_PI * 2){the -= (M_PI * 2);}
                        if(the < 0){the += (M_PI * 2);}

                        double weight = exp(-(i_rot * i_rot + j_rot * j_rot)/ _sig);
                        mag = weight * mag;

                        double orien_bin = ((the - (M_PI * 2 - f[a].orien)) * 180.0 / M_PI) * (bins / 360.0);

                        double ob = cvFloor(orien_bin);
                        double ib = cvFloor(i_bin);
                        double jb = cvFloor(j_bin);

                        double do_bin = orien_bin - ob;
                        double di_bin = i_bin - ib;
                        double dj_bin = j_bin - jb;

                        //Make sure ob is in 0 - 8 dimension
                        if(ob < 0){ob += bins;}
                        if(ob >= bins){ob -= bins;}

                        //put the orientation descriptor into the right dimension
                        for(int y = 0; y <= 1; y++)
                        {
                            int yy = jb + y;
                            if(yy >= 0 && yy < 4)
                            {
                                double mag_yy;
                                if(j == 0)
                                    mag_yy = mag * (1.0 - (dj_bin));
                                else
                                    mag_yy = mag * dj_bin;

                                for(int x = 0; x <= 1; x++)
                                {
                                    int xx = ib + x;
                                    if(xx >= 0 && xx < 4)
                                    {
                                        double mag_xx;
                                        if(x == 0)
                                            mag_xx = mag_yy * (1.0 - (di_bin));
                                        else
                                            mag_xx = mag_yy * di_bin;
                                        for(int o = 0; o <= 1; o++)
                                        {
                                            double mag_oo;
                                            if(o == 0)
                                                mag_oo = mag_xx * (1.0 - (do_bin));
                                            else
                                                mag_oo = mag_xx * do_bin;

                                            int oo = (int)(ob + o) % 8;
                                            f[a].h[yy][xx][oo] += mag_oo;
                                        }
                                    }
                                }
                            }
                        }
                    }

                }
            }
        //norm
        double tmp = 0;
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int o = 0; o < 8; o++)
                {
                    tmp += f[a].h[j][i][o] * f[a].h[j][i][o];

                }

        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int o = 0; o < 8; o++)
                {
                    double h_val = f[a].h[j][i][o] * 1.0 / sqrt(tmp * tmp);
                    if(h_val > 0.2){h_val = 0.2;}
                    f[a].h[j][i][o] = h_val;
                }

        tmp = 0;
        //renorm
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int o = 0; o < 8; o++)
                {
                    tmp += f[a].h[j][i][o] * f[a].h[j][i][o];

                }

        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int o = 0; o < 8; o++)
                {
                    f[a].h[j][i][o] *= 1.0 / sqrt(tmp);
                }


        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int o = 0; o < 8; o++)
                {
                    f[a].h[j][i][o] = min(255.0, 512.0 * f[a].h[j][i][o]);
                }
    }
}

void STITCH::drawSIFTFeatures(vector<featurePoints> &f, Mat &img)
{
    Mat tmp = img.clone();
    for(int i = 0; i < f.size(); i++)
    {
        circle(tmp, Point(f[i].xHat[0], f[i].xHat[1]), 2, Scalar(0, 0, 255), -1, 8);
        int line_l = cvRound(f[i].scale_subl * 10.0);
        Point t(line_l * cos(f[i].orien) + f[i].xHat[0], line_l * (-1) * sin(f[i].orien) + f[i].xHat[1]);
        line(tmp, Point(f[i].xHat[0], f[i].xHat[1]), t, Scalar(255, 0, 0), 2, 8);
    }

    img = tmp.clone();
}

void STITCH::warping(vector<Mat> &inputArrays,vector<double> FL2,vector<Mat> &Output,vector<Point> &upedge,vector<Point> &downedge)
{

    for(int j = 0; j < inputArrays.size(); j++)
    {
        Mat image = inputArrays[j].clone();

        int mid_x = image.cols/2;
        int mid_y = image.rows/2;

        double FL = FL2[j];


        Mat temp=Mat(image.rows,image.cols,CV_8UC3);
        temp = Scalar::all(0);
        for(int b = 0; b < image.rows; b++)
        {
            for(int a = 0; a < image.cols; a++)
            {
                double theta = atan((a-mid_x)/FL);
                double h = (b-mid_y)/pow(pow((a-mid_x),2)+pow(FL,2),0.5);
                int x = FL*theta+mid_x;
                int y = FL*h+mid_y;
                temp.at<Vec3b>(y, x)[0] = image.at<Vec3b>(b,a)[0];
                temp.at<Vec3b>(y, x)[1] = image.at<Vec3b>(b,a)[1];
                temp.at<Vec3b>(y, x)[2] = image.at<Vec3b>(b,a)[2];
                if(b == 0)
                {
                    Point temp = Point(x,y);
                    upedge.push_back(temp);
                }
                else if(b==image.rows -1 )
                {
                    Point temp = Point(x,y);
                    downedge.push_back(temp);
                }
            }
        }
        Output.push_back(temp);
    }
}

void STITCH::multiBandBlend(Mat &limg, Mat &rimg, int dx, int dy)
{
    cout << "in multiBandBlend" << endl;
    if(dx % 2 == 0)
    {
        if(dx + 1 <= limg.cols && dx + 1 <=rimg.cols)
        {
            dx += 1;
        }
        else
        {
            dx -= 1;
        }
    }
    if(dy % 2 == 0)
    {
        if(dy + 1 <= limg.rows && dy + 1 <=rimg.rows)
        {
            dy += 1;
        }
        else
        {
            dy -= 1;
        }
    }


    vector<Mat> llpyr, rlpyr;
    cout << "buildLaplacianMap" << endl;
    buildLaplacianMap(limg, llpyr, dx, dy, LEFT);
    buildLaplacianMap(rimg, rlpyr, dx, dy, RIGHT);
    cout << "getGaussianKernel" << endl;
    int center = 0;
    int i, c;
    vector<Mat> LS(level);
    for(int a = 0; a < llpyr.size(); a++) {
        Mat k = getGaussianKernel(llpyr[a].cols, llpyr[a].rows, llpyr[a].cols);
        LS[a] = Mat(llpyr[a].rows, llpyr[a].cols, CV_32FC3).clone();
        center = (int)(llpyr[a].cols / 2.0);
        for(int j = 0; j < LS[a].rows; j++) {
            for(i = 0; i < LS[a].cols; i++) {
                for(c = 0; c < 3; c++) {
                    if(a == llpyr.size() - 1)
                        LS[a].at<Vec3f>(j, i)[c] = llpyr[a].at<Vec3f>(j, i)[c] * k.at<float>(j, i) + rlpyr[a].at<Vec3f>(j, i)[c] * (1.0 - k.at<float>(j, i));
                    else
                        if(i == center) {
                            LS[a].at<Vec3f>(j, i)[c] = (llpyr[a].at<Vec3f>(j, i)[c] + rlpyr[a].at<Vec3f>(j, i)[c]) / 2.0;
                        } else if (i > center) {
                            LS[a].at<Vec3f>(j, i)[c] = rlpyr[a].at<Vec3f>(j, i)[c];
                        } else {
                            LS[a].at<Vec3f>(j, i)[c] = llpyr[a].at<Vec3f>(j, i)[c];
                        }
                }
            }
        }
    }
    cout << "pyrUp" << endl;
    Mat result;
    for(int a = level - 1; a > 0; a--)
    {
        pyrUp(LS[a], result, LS[a - 1].size());
        for(int j = 0; j < LS[a - 1].rows; j++)
        {
            for(i =0; i < LS[a - 1].cols; i++)
            {
                for(c = 0; c < 3; c++)
                {
                    LS[a - 1].at<Vec3f>(j, i)[c] = saturate_cast<uchar>(LS[a - 1].at<Vec3f>(j, i)[c] + result.at<Vec3f>(j, i)[c]);
                }
            }
        }
    }
    cout << "blendImg" << endl;
    result = LS[0].clone();

    blendImg(limg, result, dx, dy, LEFT);
    blendImg(rimg, result, dx, dy, RIGHT);
}

Mat STITCH::getGaussianKernel(int x, int y, int dx, int dy)
{
    Mat kernel = Mat::ones(Size(x, y), CV_32FC1);
    //double sigma = 0.3 * ((dx - 1) * 0.5 -1) + 0.8;
    double half = (dx - 1) / 2.0;

    double sigma =  sqrt( (-1) * pow((double)kernel.cols - 1 - half, 2.0) / (2 * log(0.5)));
    for(int i = (kernel.cols - dx); i < kernel.cols ; i++)
    {
        double g;
        if(i <= (kernel.cols - half))
        {
            g = exp((-1) * i * i / (2 * sigma * sigma));
        }
        else
        {
            g = 1 - exp((-1) * pow(kernel.cols - i - 1 ,2.0) / (2 * sigma * sigma));
        }

        for(int j = 0; j < kernel.rows; j++)
        {
            kernel.at<float>(j, i) = g;
        }
    }


//    for(int i = 0; i < kernel.cols; i++)
//    {
//        cout << kernel.at<float>(0, i) << endl;
//    }
    return kernel;
}

void STITCH::buildLaplacianMap(Mat &inputArray, vector<Mat> &outputArrays, int dx, int dy, int lr)
{

    Mat tmp(Size(dx, abs(dy)), CV_8UC3);


    int disx = (lr == RIGHT) ? 0 : (inputArray.cols - dx);
    int disy = dy >= 0 ? ((lr == RIGHT) ? 0 : (inputArray.rows - dy)) : ((lr == RIGHT) ? (inputArray.rows + dy) : 0);


    if(disx < 0){disx = 0;}

    for(int j = 0; j < tmp.rows; j++)
    {
        for(int i = 0; i < tmp.cols; i++)
        {
            for(int c = 0; c < 3; c++)
            {
                if(j + disy < inputArray.rows && i + disx < inputArray.cols)
                    tmp.at<Vec3b>(j, i)[c] = inputArray.at<Vec3b>(j + disy, i + disx)[c];
            }
        }
    }
    waitKey();
    tmp.convertTo(tmp, CV_32FC3);

    outputArrays.clear();
    outputArrays.resize(level);

    outputArrays[0] = tmp.clone();
    for(int i = 0; i < level - 1; i++)
    {
        pyrDown(outputArrays[i], outputArrays[i + 1]);
    }

    int i = 0, c = 0;
    for(int a = 0; a < level - 1; a++)
    {
        pyrUp(outputArrays[a + 1], tmp, outputArrays[a].size());

#pragma omp parallel for private(i, c)
        for(int j = 0; j < outputArrays[a].rows; j++)
        {
            for(i =0; i < outputArrays[a].cols; i++)
            {
                for(c = 0; c < 3; c++)
                {
                    outputArrays[a].at<Vec3f>(j, i)[c] = outputArrays[a].at<Vec3f>(j, i)[c] - tmp.at<Vec3f>(j, i)[c];
                }
            }
        }
    }
}

void STITCH::blendImg(Mat &inputArray, Mat &overlap_area, int dx, int dy, int lr)
{

    int disx = (lr == RIGHT) ? 0 : (inputArray.cols - dx);
    int disy = dy >= 0 ? ((lr == RIGHT) ? 0 : (inputArray.rows - dy)) : ((lr == RIGHT) ? (inputArray.rows + dy) : 0);

    if(disy < 0){disy = 0;}
    if(disx < 0){disx = 0;}

    int  i , c;
#pragma omp parallel for private(i, c)
    for(int j = 0; j < overlap_area.rows; j++)
    {
        for(i =0; i < overlap_area.cols; i++)
        {
            for(c = 0; c < 3; c++)
            {
                if(j + disy < inputArray.rows && i + disx < inputArray.cols)
                    inputArray.at<Vec3b>(j + disy, i + disx)[c] = saturate_cast<uchar>(overlap_area.at<Vec3f>(j, i)[c]);
            }
        }
    }
}
