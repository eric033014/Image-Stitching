#include "src/stitch.h"
#include "src/featureproperties.h"
#include "iostream"
using namespace std;
using namespace cv;
// using namespace vec;


int main(int argc, char **argv) {
    cout << "Starting" << endl;

    STITCH main_process;
    // main_process.getExposureTime(argv[1]);
    vector<Mat> inputArrays;
    Mat outputArray;
    cout << "getFileName" << endl;
    main_process.getFileName(argv[1]);
    inputArrays = main_process.getInputImage(argv[1]);
    vector<vector<featurePoints>> p(inputArrays.size());
    vector<Size> pic_size;
    cout << "SIFT" << endl;
    for(int k = 0;k < inputArrays.size();k++) {
        main_process.SIFT(inputArrays[k], p[k]);
        pic_size.push_back(Size(inputArrays[k].cols, inputArrays[k].rows));
    }
    vector<vector<DMatch>> gm;
    vector<double> FL;
    cout << "estimate" << endl;
    main_process.estimate(p, pic_size, gm, FL);

    vector<int> dx;
    vector<int> dy;
    cout << "alignMatches" << endl;
    cout << inputArrays.size() <<endl;
    for(int k = 1; k < inputArrays.size(); k++) {
        main_process.alignMatches(inputArrays[k-1], p[k - 1], inputArrays[k], p[k], gm[k - 1], dx, dy, FL[k-1], FL[k]);
    }

    vector <Point> upedge;
    vector <Point> downedge;
    vector <Mat> warpingImg;
    cout << "warping" << endl;
    main_process.warping(inputArrays, FL,warpingImg,upedge,downedge);

    cout << "multiBandBlend1 " << warpingImg.size() <<  endl;
    for(int i = 0; i < warpingImg.size() - 1; i++){
        cout << dy.size() << endl;
        int dyy = 0;
        cout << dy[i] << " " << warpingImg[i].rows << endl;
        if(dy[i] > warpingImg[i].rows) {
            dyy = -1*(warpingImg[i].rows-abs(warpingImg[i].rows-dy[i]));
        } else {
            dyy = dy[i];
        }
        cout << "???" << i << endl;
        main_process.multiBandBlend(warpingImg[i], warpingImg[i + 1], dx[i], dyy);
    }

    cout << "multiBandBlend2" << endl;
    Mat left;
    int cols = 0;
    int rows = 0;
    for(int i = 0; i < warpingImg.size(); i++)
    {
        if( i != warpingImg.size() - 1)
        {
            if(dy[i] > warpingImg[i].rows)
            {
                rows +=  warpingImg[i].rows -warpingImg[i].rows;
            }
            else
                rows += warpingImg[i].rows - dy[i];
        }
        else
        {
            rows += warpingImg[i].rows;
        }

        if( i != warpingImg.size() - 1)
            cols += warpingImg[i].cols - dx[i];
        else
            cols += warpingImg[i].cols;
    }

    cout << "create result" << endl;

    Mat result;
    Size size(cols,rows*2);
    result.create(size,CV_MAKETYPE(left.depth(),3));
    result = Scalar::all(0);
    left = result(Rect(0,warpingImg[0].rows/10,warpingImg[0].cols,warpingImg[0].rows+warpingImg[0].rows/10));
    warpingImg[0].copyTo(left);
    int distance = 0;
    int distancey = warpingImg[0].rows/10;
    //Point uppoint;
    double minb = warpingImg[0].rows/2;
    double mina = warpingImg[0].cols/2;
    double maxb = warpingImg[0].rows/2;
    double maxa = warpingImg[0].cols/2;
    double upb = 0.0;
    double upa = 0.0;
    double downb = 0.0;
    double downa = 0.0;

    for(int b = distancey; b < distancey+warpingImg[0].rows; b++)
        for(int a = distance; a < warpingImg[0].cols+distance; a++)
        {
            if(b-distancey>=0 && b-distancey<warpingImg[0].rows && a-distance>=0 && a-distance<warpingImg[0].cols )
            {
                result.at<Vec3b>(b, a)[0] = warpingImg[0].at<Vec3b>(b-distancey,a-distance)[0];
                result.at<Vec3b>(b, a)[1] = warpingImg[0].at<Vec3b>(b-distancey,a-distance)[1];
                result.at<Vec3b>(b, a)[2] = warpingImg[0].at<Vec3b>(b-distancey,a-distance)[2];
            }
        }

    for(int i =1;i<=dx.size();i++)
    {
        distance = distance + warpingImg[i-1].cols - dx[i-1];
        if(dy[i-1]>warpingImg[i-1].rows)
        {
            distancey = distancey -abs(warpingImg[i-1].rows - dy[i-1]);
        }
        else
        {
            distancey = distancey +abs(warpingImg[i-1].rows- dy[i-1]);
        }


        for(int b = distancey; b < distancey+warpingImg[i].rows; b++)
        {

            for(int a = distance; a < warpingImg[i].cols+distance; a++)
            {
                if(b<0)
                {

                }
                else
                {
                    if(result.at<Vec3b>(b, a)[0] == 0 && result.at<Vec3b>(b, a)[1] == 0 && result.at<Vec3b>(b, a)[2] == 0)
                    {
                        if(b-distancey>=0 && b-distancey<warpingImg[i].rows && a-distance>=0 && a-distance<warpingImg[i].cols )
                        {
                            result.at<Vec3b>(b, a)[0] = warpingImg[i].at<Vec3b>(b-distancey,a-distance)[0];
                            result.at<Vec3b>(b, a)[1] = warpingImg[i].at<Vec3b>(b-distancey,a-distance)[1];
                            result.at<Vec3b>(b, a)[2] = warpingImg[i].at<Vec3b>(b-distancey,a-distance)[2];
                        }
                    }
                    if(b < minb)
                    {
                        minb = b;
                        mina = a;
                        upb =b;
                        upa = a;
                    }
                    if(b>maxb)
                    {
                        maxb = b;
                        maxa = a;
                        downb = b-warpingImg[i].rows;
                        downa = a;
                    }
                }
            }
        }
    }
    cout << "rotateImg" << endl;
    Mat rotateImg;
    //rotateImg.create(Size(),CV_MAKETYPE(rotateImg.depth(),3);
    //result.create(size,CV_MAKETYPE(left.depth(),3));
    double angle = atan((upb-downb)/(upa-downa))*180/3.14159;
    if(upa == downa)
    {
        angle = 0;
    }

    Point2f pt(result.cols/2,result.rows/2);
    Mat r = getRotationMatrix2D(pt,angle,1.0);
    cout << "warpAffine" << endl;
    warpAffine(result,rotateImg,r,Size(result.cols,result.rows));

    int upline =0;
    int downline=result.rows;
    for(int b= 0;b<rotateImg.rows;b++)
    {
        int upn = 0;
        int downn = 0;
        for(int a=0;a<rotateImg.cols;a++)
        {
            if(rotateImg.at<Vec3b>(b, a)[0] == 0 && rotateImg.at<Vec3b>(b, a)[1] == 0 && rotateImg.at<Vec3b>(b, a)[2] == 0)
            {
                if(b<rotateImg.rows/2)
                {
                    upn++;
                    if(upn == rotateImg.cols  && upline < b )
                    {
                        upline = b;
                    }
                }
                else
                {
                    downn++;
                    if(downn == rotateImg.cols  && downline >b)
                    {
                        downline = b;
                    }
                }
            }
        }
    }



    Mat cutImage;
    int finalrows = downline - upline+1;
    int finalcols = rotateImg.cols;
    cout << "sizefinal" << endl;
    Size sizefinal(finalcols,finalrows);
    cutImage.create(sizefinal,CV_MAKETYPE(rotateImg.depth(),3));
    cutImage = Scalar::all(0);

    for(int b = 0; b < cutImage.rows; b++)
    {
        for(int a = 0; a < cutImage.cols; a++)
        {
            //qDebug()<<b+minb<<a;
            if(b +upline >= rotateImg.rows){}
            else
            {
                cutImage.at<Vec3b>(b, a)[0] = rotateImg.at<Vec3b>(b+upline,a)[0];
                cutImage.at<Vec3b>(b, a)[1] = rotateImg.at<Vec3b>(b+upline,a)[1];
                cutImage.at<Vec3b>(b, a)[2] = rotateImg.at<Vec3b>(b+upline,a)[2];
            }
        }
    }



    int leftline =0;
    int rightline=cutImage.cols;
    for(int a= 0;a<cutImage.cols;a++)
    {
        int leftn = 0;
        int rightn = 0;
        for(int b=0;b<cutImage.rows;b++)
        {
            if(cutImage.at<Vec3b>(b, a)[0] == 0 && cutImage.at<Vec3b>(b, a)[1] == 0 && cutImage.at<Vec3b>(b, a)[2] == 0)
            {
                if(a<cutImage.cols/2)
                {
                    leftn++;
                    if(leftn == cutImage.rows  && leftline < a )
                    {
                        leftline = a;
                    }
                }
                else
                {
                    rightn++;
                    if(rightn == cutImage.rows  && rightline >a)
                    {
                        rightline = a;
                    }
                }
            }
        }
    }



    Mat cutImage2;
    int finalrows2 = cutImage.rows;
    int finalcols2 = rightline-leftline;
    cout << "sizefinal2" << endl;
    Size sizefinal2(finalcols2,finalrows2);
    cutImage2.create(sizefinal2,CV_MAKETYPE(cutImage.depth(),3));
    cutImage2 = Scalar::all(0);

    for(int b = 0; b < cutImage2.rows; b++)
    {
        for(int a = 0; a < cutImage2.cols; a++)
        {
            //qDebug()<<b+minb<<a;
            if(a+leftline >= cutImage.cols){}
            else
            {
                cutImage2.at<Vec3b>(b, a)[0] = cutImage.at<Vec3b>(b,a+leftline)[0];
                cutImage2.at<Vec3b>(b, a)[1] = cutImage.at<Vec3b>(b,a+leftline)[1];
                cutImage2.at<Vec3b>(b, a)[2] = cutImage.at<Vec3b>(b,a+leftline)[2];
            }
        }
    }





    outputArray.release();
    outputArray = cutImage2.clone();
    cout << "writeImage" << endl;
    imwrite("../result/final.jpg",outputArray);
    cout << "Finished" << endl;
    return 0;
}
