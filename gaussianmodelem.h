/* Gaussian Mixture Modele and Expectation Maximization
 *  Developed by Anubhav Rohatgi
 *  Date :: 18/09/2013
 *
 *
 */


#ifndef GAUSSIANMODELEM_H
#define GAUSSIANMODELEM_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/ml/ml.hpp>
#include <fstream>
#include <limits>
#include <math.h>
#include <conio.h>
#include <time.h>

#define DEBUGCONSOLE 1

//! Structure that stores a 3D point Coordinates
/*!
  * Stores the floating point type 3D coordinates 
  */
struct point3d 
	{   /*! x coordinate*/
		float x; 
		/*! y coordinate*/
	    float y; 
		/*! z coordinate*/
		float z; 
    };


//! Gaussian Mixture Models and Expectation Maximisation
/*!
  * The class performs GMM algorithm using EM.
  * Trains the model using training set and then computes the relative EM parameters for cluster segmentation
  * The clustering is performed using K-Means algorithm.
  * References:: <a href="http://seiya-kumada.blogspot.co.uk/2013/03/gaussian-mixturesem.html">Knowledge Link 1</a>
  * <a href="http://www.autonlab.org/tutorials/gmm14.pdf">Knowledge Link 2</a>
  */

class GaussianModelEM
{
public:

    /*!
                 \fn    GaussianModelEM(int nClusters)
                 \brief instantiates the Contructor of the class
                 \param nClusters The number of clusters for GMM
    */
    GaussianModelEM(int nClusters);




    /*!
                 \fn    ~GaussianModelEM()
                 \brief Destroys the class objects
    */
    ~GaussianModelEM();




    /*!
                 \fn    Sample_Source(std::vector<std::string> &img_paths,std::vector<std::string> &img_masks)
                 \brief Creates samples matrix from the training set of images.
                        All the samples are stored in rows while channels are distributed in 3 Channels in BGR format.
                        The function copies the Rows(samples) x Columns(Channels) Matrix to Source_Samples Matrix
				 \param img_paths The list of all the image paths to training images.
				 \param img_masks The list of all the masks related to the training images.
    */
    void Sample_Source(std::vector<std::string> &img_paths,std::vector<std::string> &img_masks);




    /*!
                 \fn    Sample_Target(cv::Mat &targetImg,cv::Mat &target_mask)
                 \brief Creates samples matrix from the Test image.
                        All the samples are stored in rows while channels are distributed in 3 Channels in BGR format.
						The function copies the Rows(samples) x Columns(Channels) Matrix to Target_Samples Matrix
				 \param targetImg The input test image.
				 \param target_mask The input test mask which is applied to the test image (It can be an empty white matrix).
    */
    void Sample_Target(cv::Mat &targetImg,cv::Mat &target_mask);




    /*!
                 \fn    trainGMM_source()
                 \brief Trains the Source samples and generates the GMM model.
    */

    void trainGMM_source();

	

	/*!
                 \fn    trainGMM_target()
                 \brief Trains the Target samples and generates the GMM model.
	*/

	void trainGMM_target();

	
	/*!
                 \fn    WriteGMMModel(const char *filePath)
                 \brief Writes the Source GMM Model to a specified file.
				 \param filePath The path of the file to save the model.
	*/
	void WriteGMMModel(const char *filePath);


	/*!
                 \fn    ReadGMMModel(const char *filename)
                 \brief Reade the Source GMM Model from a specified file and loads the model in source_model.
				 \param filename The name of the file from where the model is read.
	*/
	void ReadGMMModel(const char *filename);

	

	/*!
                 \fn    MatchModels(cv::ExpectationMaximization source_model, cv::ExpectationMaximization target_model)
                 \brief Matches the source and target models and generates the Gaussian Index related to the source.
						It compares the mixtures and generates the corresponding number of target gaussian mixture
						which relates to the source gaussian mixture number. The method uses Symmetric Kullback-Leibler Distance 
						Ref:: <a href="http://www.cs.buap.mx/~dpinto/research/CICLing07_1/Pinto06c/node2.html">Link 1</a>
						and:: <a href="http://www.morethantechnical.com/2010/06/24/image-recoloring-using-gaussian-mixture-model-and-expectation-maximization-opencv-wcode/">Link 2</a>
				 \param source_model The source Gaussian Mixture Model 
				 \param target_model The target Gaussian Mixture Model
				 \return Returns the vector of correxponding index number for the Target Mixture 
	*/
	std::vector<int> MatchModels(cv::ExpectationMaximization source_model, cv::ExpectationMaximization target_model);

	
	/*!
                 \fn    SetParameters(const char *stddev_filePath,const char *mean_filePath)
                 \brief Sets means, calculates the std_deviation and saves the parameters to respective files.
				 \param stddev_filePath The path of the file to save the standard deviation of the mixtures.
				 \param mean_filePath The path of the file to save the means.
	*/
	void SetParameters(const char *stddev_filePath,const char *mean_filePath);


	/*!
                 \fn    GetParameters(const char *stddev_pathfname,const char *mean_pathfname)
                 \brief Reads and sets means and std_deviation from respective files (fullpath+filename).
				 \param stddev_pathfname Full path and filename of standard deviations file.
				 \param mean_pathfname full path and filename of means file.
	*/
	void GetParameters(const char *stddev_pathfname,const char *mean_pathfname);



	/*!
                 \fn    IdentifyPixels(cv::Mat &dst)
                 \brief Classify the Input pixels into Fire Probable Pixels.
				 \param dst Destination Image Source
	*/
	void IdentifyPixels(cv::Mat &dst);


	/*!
                 \fn    ClassifyImageClusters(cv::Mat &dst)
                 \brief Classify the Input pixels into different Clusters.
				 \param dst Destination Image Source
	*/
	void ClassifyImageClusters(cv::Mat &dst);


private:

	clock_t start; //timer
    int clusters;
	point3d *s_means;
	float *std_devs;
	static const double conv_alpha; /* Scaling factor for input pixels */
    static const double conv_beta; /* 1/conv_alpha */
    cv::ExpectationMaximization source_model;
    cv::ExpectationMaximization target_model;
    cv::Mat source_samples;
    cv::Mat target_samples;


	/*!
                 \fn    calc3DDis(cv::Vec3f &p1, point3d &p2)
                 \brief Calculates the distance between two 3D points
				 \param p1 coordinates of point1 in 3D space (vector)
				 \param p2 coordinates of point2 in 3D space (point3d)
				 \return The function returns the floating point distance between the two points
	*/
	float calc3DDis(cv::Vec3f &p1, point3d &p2);



	/*!
                 \fn    setMeans()
                 \brief Sets the Means of from the Source Training GMM model. 
	*/
	void setMeans();


	/*!
                 \fn    get_Means()
                 \brief Gets the pointer to the stored means (Before using the means should be already set by using setMeans().
				 \return The function returns the pointer to the s_means
	*/
	point3d* get_Means();



	/*!
                 \fn    calcStdDev()
                 \brief Calculates the Standard Deviation of Each Sample in a mixture and passes the reference to std_devs.
						Uses the Source Samples.
	*/
	void calcStdDev();


	/*!
                 \fn    SaveParameter_Mean(const char *filePath)
                 \brief Writes the calculated means to a file
				 \param filePath The path of the file to save the means.
	*/
	void SaveParameter_Mean(const char *filePath);



	/*!
                 \fn    SaveParameter_StdDev(const char *filePath)
                 \brief Writes the calculated Standard Deviation to a file
				 \param filePath The path of the file to save the standard deviation of the mixtures.
	*/
	void SaveParameter_StdDev(const char *filePath);



	/*!
                 \fn    ReadParameter_StdDev(const char *filename)
                 \brief Read all the stored standard deviation from the given filename.
				 \param filePath The path and filename from where the stddev is read.
	*/
	void ReadParameter_StdDev(const char *filename);





};

#endif // GAUSSIANMODELEM_H
