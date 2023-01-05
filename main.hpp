#pragma once

#include <iostream>
#include <vector>
#include "opencv2/core.hpp"

// STRUCTS //
struct cmdArgs{
	std::string img, outimg, dir, outdir;
	float scale;
	bool centered, display, denoise;
};

struct preprocParams {
	cv::Size gBlurSize;
	int thr, thrMax;
	cv::Size kernelSize;
};

struct contourResults {
    std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Point> flatPoints;
};

// parse the command line arguments into args struct
cmdArgs parseArgs(int, char**);

// image prep for bounding
cv::Mat preProcessImage(const cv::Mat&, preprocParams, bool);

// full processing procedure
cv::Mat processImage(const std::string&, const cmdArgs&);

// find greatest bounding rectangle
cv::Rect getGlbBounds(const cv::Mat&, int=20);

// denoise the image of artificats
void cleanImage(cv::Mat&);

// get all of the countering results from an image
contourResults getContours(const cv::Mat&);	

// find the angle a min bounding rect is askew
float getSkewAngle(const cv::Mat&);

// rotate image by an angle
cv::Mat rotateImage(const cv::Mat&, float);

// get the minimum rotated rectangle bounds over all features
cv::RotatedRect getMinRotatedRect(const cv::Mat&);
