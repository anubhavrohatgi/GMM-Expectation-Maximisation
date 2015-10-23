/* Gaussian Mixture Modele and Expectation Maximization
 *  Developed by Anubhav Rohatgi
 *  Date :: 18/09/2013
 *
 *
 */

#include "gaussianmodelem.h"

const double GaussianModelEM::conv_alpha = 1.0/255.0; /* Scaling factor for input pixels */
const double GaussianModelEM::conv_beta = 255.0; /* 1/conv_alpha */

//Contructor
GaussianModelEM::GaussianModelEM(int nClusters)
{
    clusters = nClusters;
	s_means = NULL;
	std_devs = NULL;
}



//Destructor
GaussianModelEM::~GaussianModelEM()
{
	//Clear Samples
    source_samples.release();
    target_samples.release();

	//Free the pointers
	free(s_means);
	free(std_devs);

	//Clear the models
    source_model.clear();
    target_model.clear();

}


//Creates a samples matrix for the Training Set ( All images are 32F)
void GaussianModelEM::Sample_Source(std::vector<std::string> &img_paths, std::vector<std::string> &img_masks)
{
    if(DEBUGCONSOLE)
        std::cout<<"\n[Start] collecting the sample data for the source training set...\n";

    start = clock();


    CV_Assert(img_paths.size() > 0);
    CV_Assert(img_masks.size() > 0);
    CV_Assert(img_paths.size() == img_masks.size());

    std::vector<cv::Vec3f> samples_vec;

    //Fill the vector with row vector of all the samples in BGR planes respectively
    for(size_t i = 0; i < img_paths.size(); ++i)
    {
        cv::Mat img = cv::imread(img_paths[i]);
        cv::Mat img_mask = cv::imread(img_masks[i], 0);

        CV_Assert(img.data != NULL);
        CV_Assert(img_mask.data != NULL);

        cv::Mat img_32f;
        img.convertTo(img_32f, CV_32F);

        //Use the mas to mask the pixels and push only important pixels
        for (int y = 0; y < img_32f.rows; y++)
        {
            cv::Vec3f* row = img_32f.ptr<cv::Vec3f>(y);
            uchar* mask_row = img_mask.ptr<uchar>(y);
            for (int x = 0; x < img_32f.cols; x++)
            {
                if (mask_row[x]>0)
                    samples_vec.push_back(row[x]);

            }
        }

        //free the memory
        img.release();
        img_mask.release();
        img_32f.release();
    }

    cv::Mat new_samples = cv::Mat(samples_vec.size(), 3, CV_32FC1);

    //BGR planes -- Vec3
    for (int r = 0; r < new_samples.rows; ++r)
    {
        new_samples.at<cv::Vec3f>(r, 0) = samples_vec[r];
    }

    if(DEBUGCONSOLE)
    {
        std::cout<<"\n\tTotal Time Spent in making samples === "<<(clock()- start)/(double)(CLOCKS_PER_SEC/1000)<<" ms\n";
        std::cout<<"\n[Ready] with the source samples  for training ....\n";
    }

	//Copy the samples to source_samples mat
	source_samples = new_samples.clone();

    //free the memory
    samples_vec.clear();
	new_samples.release();
}


//Creates a samples matrix for the Test Image( Test image is 32F)
void GaussianModelEM::Sample_Target(cv::Mat &targetImg, cv::Mat &target_mask)
{
    if(DEBUGCONSOLE)
        std::cout<<"\n[Start] collecting the sample data for the Test image...\n";

    start = clock();

    CV_Assert(targetImg.data != NULL);
    CV_Assert(target_mask.data != NULL);

    cv::Mat img_32f;
    targetImg.convertTo(img_32f, CV_32F);

    cv::Mat new_samples = img_32f.reshape(1,targetImg.rows*targetImg.cols);

    if(DEBUGCONSOLE)
    {
        std::cout<<"\n\tTotal Time Spent in making samples === "<<(clock()- start)/(double)(CLOCKS_PER_SEC/1000)<<" ms\n";
        std::cout<<"\n[Ready] with the Target samples  for training ....\n";
    }

	//Copy the target samples to the target samples mat
	target_samples = new_samples.clone();

    //Free the memory
	new_samples.release();
	img_32f.release();
}


//Trains the Source samples and generates the GMM model.
void GaussianModelEM::trainGMM_source()
{
	source_model.clear();
	
	CV_Assert(!source_samples.empty());
	
	start = clock();
	
	if(DEBUGCONSOLE)
		std::cout<<"\n\n\n[Start] training the Source GMM model..\n";

	cv::EMParams params;
	//cv::Mat labels;
	/*params.covs		 = NULL;
	params.means     = NULL;
    params.weights   = NULL;
    params.probs     = NULL;*/
	params.nclusters = clusters;
    params.cov_mat_type       = CvEM::COV_MAT_SPHERICAL;
    params.start_step         = CvEM::START_AUTO_STEP;
    params.term_crit.max_iter = 300;
    params.term_crit.epsilon  = 0.1;
    params.term_crit.type     = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;

	source_model.train(source_samples,cv::Mat(),params,NULL);

	if(DEBUGCONSOLE)
	{
		std::cout<<"\n\tTotal Time Spent in Training the Source Model === "<<(clock()- start)/(double)(CLOCKS_PER_SEC/1000)<<" ms\n";
		std::cout<<"\n[END] Source model training done ... \n\n\n";
	}

}


//Trains the Target samples and generates the GMM model.
void GaussianModelEM::trainGMM_target()
{
	target_model.clear();

	CV_Assert(!target_samples.empty());
	
	start = clock();
	
	if(DEBUGCONSOLE)
		std::cout<<"\n\n\n[Start] training the Target GMM model..\n";

	cv::EMParams params;
	//cv::Mat labels;
	/*params.covs		 = NULL;
	params.means     = NULL;
    params.weights   = NULL;
    params.probs     = NULL;*/
	params.nclusters = clusters;
    params.cov_mat_type       = CvEM::COV_MAT_SPHERICAL;
    params.start_step         = CvEM::START_AUTO_STEP;
    params.term_crit.max_iter = 300;
    params.term_crit.epsilon  = 0.1;
    params.term_crit.type     = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;

	target_model.train(source_samples,cv::Mat(),params,NULL);

	if(DEBUGCONSOLE)
	{
		std::cout<<"\n\tTotal Time Spent in Training the Target Model === "<<(clock()- start)/(double)(CLOCKS_PER_SEC/1000)<<" ms\n";
		std::cout<<"\n[END] Target model training done ... \n\n\n";
	}
}


//Write the SOURCE GMM Model to a file
void GaussianModelEM::WriteGMMModel(const char *filePath)
{
	std::string filename(filePath);
	filename.append("model.em");
	cv::FileStorage file = cv::FileStorage(filename.c_str(), cv::FileStorage::WRITE);
	source_model.write(file.fs,"EM");
}


//Read the Source GMM model from the given filename
void GaussianModelEM::ReadGMMModel(const char *filename)
{
	source_model.clear();

	cv::FileStorage f = cv::FileStorage(filename,cv::FileStorage::READ);
	cv::FileNode fileNode = f["EM"];
	source_model.read(const_cast<CvFileStorage*>(fileNode.fs), const_cast<CvFileNode*>(fileNode.node));
	clusters = source_model.getNClusters();
}


//Matches the Source and Traget Model based on Kullback Leibler Distance and returns the index of the matches 
std::vector<int> GaussianModelEM::MatchModels(cv::ExpectationMaximization source_model, cv::ExpectationMaximization target_model)
{
	if(DEBUGCONSOLE)
		std::cout<<"\n\nMatching Started\n";

	int num_g = source_model.getNClusters();
	
	//source means and target means 
	cv::Mat sMu(source_model.get_means());
	cv::Mat tMu(target_model.get_means());
	
	//covariances of the two models
	const CvMat** target_covs = target_model.get_covs();
	const CvMat** source_covs = source_model.get_covs();

	double best_dist = std::numeric_limits<double>::max();

	std::vector<int> best_res(num_g);
	std::vector<int> prmt(num_g); 

	for(int itr = 0; itr < 10; itr++) 
	{
		for(int i=0;i<num_g;i++) 
			prmt[i] = i;	//make a permutation
			
		cv::randShuffle(cv::Mat(prmt));

		//Greedy selection
		std::vector<int> res(num_g);
		std::vector<bool> taken(num_g);

		for(int sg = 0; sg < num_g; sg++) 
		{
			double min_dist = std::numeric_limits<double>::max(); 
			int minv = -1;

			for(int tg = 0; tg < num_g; tg++) 
			{
				if(taken[tg]) continue;

				//TODO: can save on re-calculation of pairs - calculate affinity matrix ahead
				//double d = norm(sMu(Range(prmt[sg],prmt[sg]+1),Range(0,3)),	tMu(Range(tg,tg+1),Range(0,3)));
					
				//symmetric kullback-leibler
				cv::Mat diff = cv::Mat(sMu(cv::Range(prmt[sg],prmt[sg]+1),cv::Range(0,3)) - tMu(cv::Range(tg,tg+1),cv::Range(0,3)));
				cv::Mat d = diff * cv::Mat(cv::Mat(source_covs[prmt[sg]]).inv() + cv::Mat(target_covs[tg]).inv()) * diff.t();
				cv::Scalar tr = cv::trace(cv::Mat(cv::Mat(cv::Mat(source_covs[prmt[sg]])*cv::Mat(target_covs[tg])) + cv::Mat(cv::Mat(target_covs[tg])*cv::Mat(source_covs[prmt[sg]]).inv()) + cv::Mat(cv::Mat::eye(3,3,CV_64FC1)*2)	));
				double kl_dist = ((double*)d.data)[0] + tr[0];
				if(kl_dist<min_dist) 
				{
					min_dist = kl_dist;
					minv = tg;
				}
			}
			
			res[prmt[sg]] = minv;
			taken[minv] = true;
		}

		double dist = 0;
		for(int i=0;i<num_g;i++) 
		{
			dist += cv::norm(sMu(cv::Range(prmt[i],prmt[i]+1),cv::Range(0,3)),tMu(cv::Range(res[prmt[i]],res[prmt[i]]+1),cv::Range(0,3)));
		}

		if(dist < best_dist) 
		{
			best_dist = dist;
			best_res = res;
		}
	}

	if(DEBUGCONSOLE)
		std::cout<<"\n\nMatching Completed\n";

	return best_res;
}



//Sets means, calculates the std_deviation and saves the parameters to respective files.
void GaussianModelEM::SetParameters(const char *stddev_filePath,const char *mean_filePath)
{
	//intialize pointers to NULL
	s_means = NULL;
	std_devs = NULL;

	//Set the means from the existing source training model
	setMeans();

	//Save the Means to file
	SaveParameter_Mean(mean_filePath);

	//Calculate the Standard Deviations
	calcStdDev();

	//Save the calculated Standard Deviations to a file
	SaveParameter_StdDev(stddev_filePath);
}


//Reads and sets means and std_deviation from respective files (fullpath+filename)
void GaussianModelEM::GetParameters(const char *stddev_pathfname,const char *mean_pathfname)
{
	//intialize pointers to NULL
	std_devs = NULL;
	s_means = NULL;

	//Retrieve all the means from the loaded source model
	setMeans();

	//Load all the standard deviations from the provided path and filename
	ReadParameter_StdDev(stddev_pathfname);

	if(DEBUGCONSOLE)
	{
		std::cout<<"\n\nStandard Deviations\n";
		for(int idx = 0; idx < clusters; ++idx) 
		{
				std::cout << std_devs[idx] << std::endl;
		}

		std::cout<<"\n\nMeans - BGR format\n";

		for(int i=0;i< clusters;++i)
			std::cout<<s_means[i].x<<"\t"<<s_means[i].y<<"\t"<<s_means[i].z<<std::endl;

		std::cout<<"\n\n";
	}


}


//Function calculates the distance between two 3D points
float GaussianModelEM::calc3DDis(cv::Vec3f &p1, point3d &p2)
{
	float a = p1.val[0] - p2.x;
	float b = p1.val[1] - p2.y;
	float c = p1.val[2] - p2.z;
	return sqrt(a*a + b*b + c*c);
}


//Sets the means of the Source Training Samples GMM model to means Matrix
void GaussianModelEM::setMeans()
{
	cv::Mat means_g = source_model.getMeans();
	
	if(s_means == NULL)
		s_means = (point3d*) malloc(clusters * sizeof(point3d));

	CV_Assert(means_g.rows == clusters);
	for(int i = 0; i < means_g.rows; i++)
	{
		s_means[i].x = (float)means_g.at<double>(i,0);
		s_means[i].y = (float)means_g.at<double>(i,1);
		s_means[i].z = (float)means_g.at<double>(i,2);
	}
	
	//free the memory
	means_g.release();
}


//Gets the pointer to the stored means (Before using the means should be already set by using setMeans()
point3d* GaussianModelEM::get_Means()
{
	return s_means;
}


//Calculates the Standard Deviation of Each Sample in a mixture and passes the reference to std_devs. Uses the Source Samples.
void GaussianModelEM::calcStdDev()
{
	point3d* nmeans =  get_Means();
	CV_Assert(nmeans != NULL); // Check if the s_means was not empty
	
	int mix_size = clusters; //pass the clusters (Mixtures)
	std::vector<int> cardinality(mix_size); //fix the size of this vector to the number of clusters (cardinality-> no of elements in this set)

	float *sds = (float *)malloc(mix_size * sizeof(float)); //stddev mem alloc
	memset(sds, 0, mix_size * sizeof(float)); //setting the pointer block to Zero

	cv::Mat sample(1, 3, CV_32FC1);

	//since the samples are stored in vector of planes in matrix
	for(int r = 0; r < source_samples.rows; ++r)
	{
	        sample.at<cv::Vec3f>(0, 0) = source_samples.at<cv::Vec3f>(r, 0);
			int idx = (int)source_model.predict(sample);
	        cardinality[idx]++;
	        float dis = calc3DDis(sample.at<cv::Vec3f>(0, 0), nmeans[idx]);
	        sds[idx] += dis*dis;

			/*if(DEBUGCONSOLE)
				std::cout <<sds[idx] << std::endl;*/
	}


	for(int idx = 0; idx < mix_size; ++idx) 
	{
	        sds[idx] = sqrt(sds[idx]/(cardinality[idx] - 1));

			if(DEBUGCONSOLE)
 				std::cout << sds[idx] << std::endl;
	}
	
	if(std_devs != NULL) //check if std_devs was empty
	{
	        free(std_devs);
	}

	std_devs = sds;

	//free the memory
	cardinality.clear();
	
}


//Save the mean of the training samples BGR order
void GaussianModelEM::SaveParameter_Mean(const char *filePath)
{
	point3d *nmeans = get_Means();
	CV_Assert(nmeans != NULL);

	std::ofstream means_file;
	std::string mean_filePath(filePath);
	mean_filePath.append("means.em");
	means_file.open(mean_filePath.c_str());
	if(!means_file.is_open())
		std::cerr<<"\n[ERROR] Unable to open the means file"<<std::endl;

	//storing in BGR order
	for(int i=0;i< clusters;++i)
		means_file<<nmeans[i].x<<"\t"<<nmeans[i].y<<"\t"<<nmeans[i].z<<std::endl;

	//free the memory
	means_file.close();
}


//Save the stddev of the training samples
void GaussianModelEM::SaveParameter_StdDev(const char *filePath)
{
	float *stddev = std_devs;
	CV_Assert(stddev != NULL);

	std::ofstream stddev_file;
	std::string stddev_filePath(filePath);
	stddev_filePath.append("std_devs.em");
	stddev_file.open(stddev_filePath.c_str());

	if(!stddev_file.is_open())
		std::cerr<<"\n[ERROR] Unable to open the stdev file"<<std::endl;

	for(int i=0;i < clusters; ++i)
		stddev_file<<stddev[i]<<std::endl;

	//free the memory
	stddev_file.close();
}


//Reads the Standard Deviations of each cluster from given filename
void GaussianModelEM::ReadParameter_StdDev(const char *filename)
{
	std::ifstream stddev_file;
	stddev_file.open(filename);
	
	if(!stddev_file.is_open())
	{
		std::cerr<<"\n[ERROR] Unable to open the stddev file"<<std::endl;
		return;
	}


	int num_mixtures = clusters;
	float *stddevs = (float*) malloc(num_mixtures* sizeof(float));

	int line_number = 0;
	std::string c_line;

	while(stddev_file.good())
	{
		std::getline(stddev_file,c_line);

		if(c_line.empty())
			break;

		std::istringstream in(c_line);
		in>>stddevs[line_number];
		
		if(line_number >= num_mixtures)
		{
			std::cerr<<"\n[ERROR] Too many values in stddev file"<<std::endl;
			return;
		}

		++line_number;
	}

	

	if(std_devs != NULL)
		free(std_devs);

	std_devs = stddevs;

	//free the memory
	stddev_file.close();
}


void GaussianModelEM::ClassifyImageClusters(cv::Mat &dst)
{
	cv::Mat target_32f;
	if(dst.type() != CV_32F)
		dst.convertTo(target_32f,CV_32F,conv_alpha);
	else
		dst = target_32f.clone();


	cv::Mat nmeans = source_model.getMeans();
	//cv::Mat nweights = source_model.getWeights();

	cv::Mat meanImg = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_32FC3);
	cv::Mat fgImg = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_8UC3);
    cv::Mat bgImg = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_8UC3);

	cv::Mat f1 = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_8UC3);
	cv::Mat f2 = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_8UC3);
	cv::Mat f3 = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_8UC3);
	cv::Mat f4 = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_8UC3);
	cv::Mat f5 = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_8UC3);
	cv::Mat f6 = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_8UC3);
	cv::Mat f7 = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_8UC3);

	//now classify each of the source pixels
    int idx = 0;
    for (int y = 0; y < target_32f.rows; y++) {
        for (int x = 0; x < target_32f.cols; x++) {

            //classify
            const int result = cvRound(source_model.predict(target_samples.row(idx++), NULL));

		
            //get the according mean (dominant color)
            const double* ps = nmeans.ptr<double>(result, 0);

            //set the according mean value to the mean image
            float* pd = meanImg.ptr<float>(y, x);
            //float images need to be in [0..1] range normalization
            pd[0] = (float)ps[0] / 255.0f;
            pd[1] = (float)ps[1] / 255.0f;
            pd[2] = (float)ps[2] / 255.0f;

            //set either foreground or background
            if ((result ==0)) {
                fgImg.at<cv::Point3_<uchar> >(y, x, 0) = dst.at<cv::Point3_<uchar> >(y, x, 0);
				// bgImg.at<cv::Point3_<uchar> >(y, x, 0) = emptyImg.at<cv::Point3_<uchar> >(y, x, 0);
				} 
			if(result == 1)
				f1.at<cv::Point3_<uchar> >(y, x, 0) = dst.at<cv::Point3_<uchar> >(y, x, 0);
			if(result == 2)
				f2.at<cv::Point3_<uchar> >(y, x, 0) = dst.at<cv::Point3_<uchar> >(y, x, 0);
			if(result ==3)
				f3.at<cv::Point3_<uchar> >(y, x, 0) = dst.at<cv::Point3_<uchar> >(y, x, 0);
			if(result ==4)
				f4.at<cv::Point3_<uchar> >(y, x, 0) = dst.at<cv::Point3_<uchar> >(y, x, 0);
			if(result ==5)
				f5.at<cv::Point3_<uchar> >(y, x, 0) = dst.at<cv::Point3_<uchar> >(y, x, 0);
			if(result ==6)
				f6.at<cv::Point3_<uchar> >(y, x, 0) = dst.at<cv::Point3_<uchar> >(y, x, 0);
			if(result ==7)
				f7.at<cv::Point3_<uchar> >(y, x, 0) = dst.at<cv::Point3_<uchar> >(y, x, 0);



			//else {
			//	//fgImg.at<cv::Point3_<uchar> >(y, x, 0) = emptyImg.at<cv::Point3_<uchar> >(y, x, 0);
   //             bgImg.at<cv::Point3_<uchar> >(y, x, 0) = dst.at<cv::Point3_<uchar> >(y, x, 0);
           // }
        }
    }

	meanImg.convertTo(meanImg,CV_32F);
	cv::imshow("Means", meanImg);
	cv::imshow("Foreground", fgImg);
    cv::imshow("Background", bgImg);
	cv::imshow("Mix 1", f1);
	cv::imshow("Mix 2", f2);
	cv::imshow("Mix 3", f3);
	cv::imshow("Mix 4", f4);
	cv::imshow("Mix 5", f5);
	cv::imshow("Mix 6", f6);
	cv::imshow("Mix 7", f7);

	f1.release();
	f2.release();
	f3.release();
	f4.release();
	f5.release();
	f6.release();
	f7.release();
	meanImg.release();

}


void GaussianModelEM::IdentifyPixels(cv::Mat &dst)
{
	cv::Mat target_32f;
	if(dst.type() != CV_32F)
		dst.convertTo(target_32f,CV_32F,conv_alpha);
	else
		dst = target_32f.clone();


	point3d *nmeans = get_Means();
	
	cv::Mat fgImg = cv::Mat::zeros(target_32f.rows, target_32f.cols, CV_8UC1);

	cv::normalize(target_32f,target_32f,0.0,255.0,CV_MINMAX);
	std::vector<cv::Mat> planes;

	//cv::split(target_32f,planes);
	// for (int y = 0; y < target_32f.rows; y++) 
	// {
	//	for (int x = 0; x < target_32f.cols; x++) 
	//	{

	//		std::cout<<"\t"<<planes[0].at<float>(y,x);
	//	}

	//	std::cout<<std::endl;
	// }


	 for (int y = 0; y < target_32f.rows; y++) 
	 {
		cv::Vec3f* row = target_32f.ptr<cv::Vec3f>(y);
		for (int x = 0; x < target_32f.cols; x++) 
		{
			uchar *pd = fgImg.ptr<uchar>(y,x);
			//std::cout<<"\n"<<row[x].val[0] <<"\t"<<row[x].val[1]<<"\t"<<row[x].val[2];
			for(int i =0; i< clusters; i++)
			{
				
				if(calc3DDis(row[x],nmeans[i]) < std_devs[i])
				{
					pd[0] = 255;
					
				}
				//std::cout<<"\t "<< calc3DDis(row[x],nmeans[i]);
			}


		}
	 }

	 //std::cout<<fgImg;
	 cv::imshow("Dude Map",fgImg);
	 
	 //free the memory
	 fgImg.release();

}
