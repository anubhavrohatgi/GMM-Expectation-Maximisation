#include <iostream>
#include <opencv2/opencv.hpp>
#include "gaussianmodelem.h"
#include <conio.h>

using namespace std;


int main()
{
    cv::Mat dst = cv::imread("data/2.jpg");
    cv::Mat dst_mask = cv::imread("data/var_mask.png");

    std::vector<std::string> img_paths;
    std::vector<std::string> img_masks;

    std::string path("data/img/");

    path.append("fire/");


    int num_samples = 9;

    char fbuffer[200];
    char maskbuffer[200];


    for(int i = 0; i < num_samples; ++i)
    {
        std::sprintf(fbuffer, "%strain%03d.jpg\0", path.c_str(), i);
        std::sprintf(maskbuffer, "%strain%03d_mask.png\0", path.c_str(), i);

        img_paths.push_back(std::string(fbuffer));
        img_masks.push_back(std::string(maskbuffer));
    }

    std::cout<<"\nThe following files are used for Training:::\n\n\n";

    for(size_t i = 0; i < img_paths.size(); ++i)
    {
        printf("%s \t  %s\n", img_paths[i].c_str(), img_masks[i].c_str());
    }



    GaussianModelEM instance(8);

    instance.Sample_Source(img_paths,img_masks);
	instance.Sample_Target(dst,dst_mask);
	instance.trainGMM_source();
	instance.trainGMM_target();

	instance.WriteGMMModel("data/");
	instance.ReadGMMModel("data/model.em");

	instance.SetParameters("data/","data/");
	instance.GetParameters("data/std_devs.em","data/means.em");
	instance.ClassifyImageClusters(dst);
	instance.IdentifyPixels(dst);

    dst.release();
    dst_mask.release();

	cv::waitKey(0);
    cout << "\n\nTerminating Sequence!!!!!\n\n\n" << endl;

	_getch();
	

	cv::destroyAllWindows();
    return 0;
}


