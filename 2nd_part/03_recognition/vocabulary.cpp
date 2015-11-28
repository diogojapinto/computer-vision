#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

bool openImage(const std::string &filename, Mat &image);
void drawKeypoints(string windowName, Mat &image, std::vector<KeyPoint> &keypoints, std::vector<int> &words);

class Vocabulary
{
public:
	Mat vocabulary;
	int nWords;

	Vocabulary(int nWords);
	void train(vector<string> &listOfImages);
	int whichWord(Mat descriptor);
};


int main( int argc, char** argv ) 
{
	initModule_nonfree();

	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");

    ///////////////////////////////////////






    ///////////////////////////////////////

    waitKey(0);
}

bool openImage(const std::string &filename, Mat &image)
{
	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE );
	if( !image.data ) {
		std::cout << " --(!) Error reading image " << filename << std::endl;
		return false;
	}
	return true;
}

void drawKeypoints(string windowName, Mat &image, vector<KeyPoint> &keypoints, std::vector<int> &words)
{
	if(keypoints.size() != words.size())
		return;

	Mat newImage;
	cvtColor(image, newImage, CV_GRAY2RGB);
	int max=0;
	for(int i=0; i<words.size(); i++)
		if(words[i]>max)
			max = words[i];

	int steps = (int)(255/(log(max+1)/log(3)));
	vector<Scalar> colors;
	for(int r=1; r<256; r+=steps)
		for(int g=1; g<256; g+=steps)
			for(int b=1; b<256; b+=steps)
				colors.push_back(cvScalar(b,g,r));

	for(int i=0; i<keypoints.size(); i++)
	{
		circle(newImage, keypoints[i].pt, 4, colors[words[i]],2);
	}

	namedWindow( windowName, CV_WINDOW_AUTOSIZE );
    imshow( windowName, newImage );
}

Vocabulary::Vocabulary(int nWords)
{
	this->nWords = nWords;
}

void Vocabulary::train(vector<string> &listOfImages)
{
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    Mat image;
	std::vector<KeyPoint> keypoints;
	Mat descriptors, allDescriptors;

	for(int i=0; i<listOfImages.size(); i++)
	{
		if(!openImage(listOfImages[i], image))
			continue;

		detector->detect( image, keypoints );
		extractor->compute( image, keypoints, descriptors );
		allDescriptors.push_back(descriptors);
	}


	Mat labels;
    kmeans( allDescriptors, nWords, labels, TermCriteria(), 1, KMEANS_PP_CENTERS, vocabulary );
}

int Vocabulary::whichWord(Mat descriptor)
{
	double minDistance, distance;
	int minIndex;

	if(vocabulary.rows <= 1)
		return -1;

	minIndex = 0;
	minDistance = norm(vocabulary.row(0),descriptor,NORM_L2);

	for(int i=1; i<vocabulary.rows; i++)
	{
		distance = norm(vocabulary.row(i),descriptor,NORM_L2);
		if(distance < minDistance)
		{
			minDistance = distance;
			minIndex = i;
		}
	}

	return minIndex;
}

