#ifndef FEATUREPROPERTIES_H
#define FEATUREPROPERTIES_H
#include <vector>

//Custom feature point structure
struct featurePoints{
    int x;
    int y;
    int l;
    int subl;
    double response;
    double xHat[3];
    double scale;
    double scale_subl;
    double orien = 0;
    std::vector<std::vector<std::vector<double>>> h;
    featurePoints(){
        h.resize(4);
        for(int j = 0; j < 4; j++)
        {
            h[j].resize(4);
           for(int i = 0; i < 4; i++)
           {
               h[j][i].resize(8, 0);
           }
        }
    }

    void operator = (featurePoints a){
        x = a.x;
        y = a.y;
        l = a.l;
        subl = a.subl;
        response = a.response;
        xHat[0] = a.xHat[0]; xHat[1] = a.xHat[1]; xHat[2] = a.xHat[2];
        scale = a.scale;
        scale_subl = a.scale_subl;
        orien = a.orien;
        for(int j = 0; j < 4; j++){
            for(int i = 0; i < 4; i++){
               for(int o = 0; o < 8; o++)
               {
                   h[j][i][o] = a.h[j][i][o];
               }
            }
        }
    }
};


#endif // FEATUREPROPERTIES_H
