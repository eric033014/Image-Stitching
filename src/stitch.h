#ifndef STITCH_H
#define STITCH_H
#define RIGHT 1
#define LEFT 0
#define _USE_MATH_DEFINES

#include "opencv2/opencv.hpp"
#include "featureproperties.h"
#include <math.h>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace detail;
using namespace std;

class STITCH {
    public:
        // STITCH();
        vector<Mat> getInputImage(string testing_set);
        void getFileName(string testing_set);
        void SIFT(Mat &inputArray, vector<featurePoints> &_features);
        void drawSIFTFeatures(vector<featurePoints> &f, Mat &img);
        void process(vector<vector<featurePoints>> &f, vector<Size> &pic_size, vector<vector<DMatch>> &good_matches, vector<double> &focal);
        void estimate(vector<vector<featurePoints>> &f, vector<Size> &pic_size, vector<vector<DMatch>> &good_matches, vector<double> &focal);
        void drawMatches(Mat &img1, vector<featurePoints> &f1, Mat &img2, vector<featurePoints> &f2, vector<DMatch> good_matches);
        void alignMatches(Mat &img1, vector<featurePoints> &f1, Mat &img2, vector<featurePoints> &f2, vector<DMatch> good_matches,
                     vector<int> &x,vector<int> &y,double FL1,double FL2);
        void toCVImageFeatures(vector<vector<featurePoints> > &f, vector<Size> &pic_size, vector<detail::ImageFeatures> &ifs);
        void toCVDescriptor(vector<featurePoints> &f, Mat &d);
        void warping(vector<Mat> &inputArrays,vector<double> FL2,vector<Mat> &Output,vector<Point> &upedge,vector<Point> &downedge);
        void DOG(Mat inputArray, vector<Mat> &_dogs, vector<Mat> &_gpyr, const double _k, const double _sig);
        bool isExtreme(vector<Mat> dogs, int _l, int _sl, int j, int i);
        bool interp(vector<Mat> _dog, int _layer, int _sublayer, int j, int i, featurePoints &f);
        //Differential function
        Mat xHat(vector<Mat> _dog, int _layer, int _sublayer, int j, int i);
        Mat diff(vector<Mat> _dog, int _layer, int _sublayer, int j, int i);
        Mat hessian(vector<Mat> _dog, int _layer, int _sublayer, int j, int i);
        bool removeEdge(Mat _dogImg, int j, int i);
        void orien(vector<featurePoints> &f, vector<Mat> &_gpyr);
        void descriptor(vector<featurePoints> &f, vector<Mat> &_gpyr);
        void multiBandBlend(cv::Mat &limg, cv::Mat &rimg, int dx, int dy);
        cv::Mat getGaussianKernel(int x, int y, int dx, int dy = 0);
        void buildLaplacianMap(cv::Mat &inputArray, std::vector<cv::Mat> &outputArrays, int dx, int dy, int lr);
        void blendImg(cv::Mat &img, cv::Mat &overlap_area, int dx, int dy, int lr);
    private:

        // Mat outputArray;
        vector<string> file_name_list;
        vector<double> FocalL;
        const double ocv = 3.0;
        const double s = 3.0;
        const double sig = 1.6;
        const double k = pow(2.0, 1.0 / s);
        const double norming = 1.0 / 255.0;
        const int level = 5;
};

#endif
