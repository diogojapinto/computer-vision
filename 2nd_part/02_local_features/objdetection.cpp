#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

bool openImage(const std::string &filename, Mat &image);
void filterMatchesByAbsoluteValue(std::vector<DMatch> &matches, float maxDistance);
Mat filterMatchesRANSAC(std::vector<DMatch> &matches, std::vector<KeyPoint> &keypointsA, std::vector<KeyPoint> &keypointsB);
void showResult(Mat &imgA, std::vector<KeyPoint> &keypointsA, Mat &imgB, std::vector<KeyPoint> &keypointsB, std::vector<DMatch> &matches, Mat &homography);


int main( int argc, char** argv ) 
{
	initModule_nonfree();

	//////////////////////////////////////////////////////////////////////////
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	//////////////////////////////////////////////////////////////////////////



    // TODO: implement the rest of the code...


	return 0; 
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

void filterMatchesByAbsoluteValue(std::vector<DMatch> &matches, float maxDistance)
{
	std::vector<DMatch> filteredMatches;
	for(size_t i=0; i<matches.size(); i++)
	{
		if(matches[i].distance < maxDistance)
			filteredMatches.push_back(matches[i]);
	}
	matches = filteredMatches;
}

Mat filterMatchesRANSAC(std::vector<DMatch> &matches, std::vector<KeyPoint> &keypointsA, std::vector<KeyPoint> &keypointsB)
{
	Mat homography;
	std::vector<DMatch> filteredMatches;
	if(matches.size() >= 4)
	{
		vector<Point2f> srcPoints;
		vector<Point2f> dstPoints;
		for(size_t i=0; i<matches.size(); i++)
		{

			srcPoints.push_back(keypointsA[matches[i].queryIdx].pt);
			dstPoints.push_back(keypointsB[matches[i].trainIdx].pt);
		}

		Mat mask;
		homography = findHomography(srcPoints, dstPoints, CV_RANSAC, 1.0, mask);

		for(int i=0; i<mask.rows; i++)
		{
			if(mask.ptr<uchar>(i)[0] == 1)
				filteredMatches.push_back(matches[i]);
		}
	}
	matches = filteredMatches;
	return homography;
}

void showResult(Mat &imgA, std::vector<KeyPoint> &keypointsA, Mat &imgB, std::vector<KeyPoint> &keypointsB, std::vector<DMatch> &matches, Mat &homography)
{
	// Draw matches
	Mat imgMatch;
	drawMatches(imgA, keypointsA, imgB, keypointsB, matches, imgMatch);

	if(!homography.empty())
	{
		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( imgA.cols, 0 );
		obj_corners[2] = cvPoint( imgA.cols, imgA.rows ); obj_corners[3] = cvPoint( 0, imgA.rows );
		std::vector<Point2f> scene_corners(4);

		perspectiveTransform( obj_corners, scene_corners, homography);
		
		float cols = (float)imgA.cols;
		line( imgMatch, scene_corners[0] + Point2f(cols, 0), scene_corners[1] + Point2f(cols, 0), Scalar(0, 255, 0), 4 );
		line( imgMatch, scene_corners[1] + Point2f(cols, 0), scene_corners[2] + Point2f(cols, 0), Scalar( 0, 255, 0), 4 );
		line( imgMatch, scene_corners[2] + Point2f(cols, 0), scene_corners[3] + Point2f(cols, 0), Scalar( 0, 255, 0), 4 );
		line( imgMatch, scene_corners[3] + Point2f(cols, 0), scene_corners[0] + Point2f(cols, 0), Scalar( 0, 255, 0), 4 );
	}


	namedWindow("matches", CV_WINDOW_KEEPRATIO);
	imshow("matches", imgMatch);
	waitKey(0);
}