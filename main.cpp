/*
  	From Father Joseph:
	(1). Read an image from a file and convert the image to grayscale.
	(2). Convert the grayscale image to a binary image.
	(3). Find the external contours of the text block in the binary image.
	(4). Bound the image with a bounding rectangle.

	(A). Crop the area of the bounding rectangle.
	(B). Convert back the binary image to its previous format and convert the converted grayscale image back to RGB.
	(C). Inscribe the converted cropped image into a white background image with specific dimensions.

	ToDo:
		- normalize pages ( static margins, dynamic resizing to printed area )
		- add options for margin buffers and static output size for page prep
		- deal with rotation (deskew image subject to constraints)
		- de-warping --3-space problem (may not be tractable)

*/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"

#include "main.hpp"
#include "utils.hpp"

using std::cout, std::endl, std::vector, std::string;
using cv::Mat, cv::Rect, cv::Point, cv::Size;
namespace fs = std::filesystem;


int main(int argc, char** argv) { 

	cmdArgs args = parseArgs(argc, argv);
	auto startTime = std::chrono::high_resolution_clock::now(); // start timer
	if (args.dir=="") {
		// single image case
		Mat imgOut = processImage(args.img, args);

		// display image if single image
		if (args.display) {
			cv::imshow("output image", imgOut);
			cv::waitKey(); // Wait for a keystroke in the window
		}

		// write file
		cout << "writing file to: " << args.outimg << endl;
		if (!cv::imwrite(args.outimg, imgOut)) {
			cout << "bad file write to: " << args.outimg << endl;
			exit(1);
		};

	} else {
		// directory case
		vector<string> files = getDirFiles(args.dir);
		if (files.empty()) {
			cout << "dir: " << args.dir << " did not find any *.jpg files to load." << endl;
			exit(1);
		}

		// set up output dir
		if (std::filesystem::create_directory(args.outdir)) {
			cout << "creating directory: " << args.outdir << endl;
		}

		// file process loop
		fs::path indir (args.dir);
		fs::path outdir (args.outdir);
		int totalFiles = files.size();
		int currentCount = 0;
		int pollFreq = std::max(totalFiles / 20, 1);
		int cnt = 0;

		for (auto f : files) {
			fs::path file (f);
			fs::path fullPath = indir / file;
			//cout << f << ", "; cout.flush();  //for debugging
			Mat imgOut = processImage(fullPath, args);

			// write file
			fs::path fullOutPath = outdir / file;
			if (!cv::imwrite(fullOutPath, imgOut)) {
				cout << "bad file write to: " << args.outimg << endl;
				exit(1);
			};

			// progress
			cnt++;
			if (cnt % pollFreq == 0) { printProgress(totalFiles, cnt); }
		}
	}

	auto finishTime = std::chrono::high_resolution_clock::now();
	string elapsed = getElapsedTime(std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count());
	cout << "\nfinished, elapsed time: " << elapsed << endl;

	return 0;
}

// parse the command line args
cmdArgs parseArgs(int argc, char** argv) {
	
	const string keys =
	"{ help h usage ? || $ mImgProc [-i<image.jpg>] [-io=out.jpg] [-d=DIR] [-do=out]..}"
    "{ i img || single image path to process }"
	"{ io imgout |out.jpg| path for single output image }"
	"{ d dir || directory to process (abs path) }"
	"{ do dirout |out| output directory to store processed images }"
	"{ s scale |1.0| scale factor of output image (resize) }"
	"{ c centered |false| center image in output (true), default retains position }"
	"{ p show |false| show final image if a single file used, ignored if dir }"
	"{ n denoise |false| denoise the crop before copying to blank }"
	;

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Missal Image Processor v0.1");

	// no argument case
	if (argc < 2) {parser.printMessage(); exit(0);}

	// help case
	if (parser.has("help")){
    	parser.printMessage();
    	exit(0);
	}

	cmdArgs inArgs;

	inArgs.img = parser.get<string>("i");
	inArgs.outimg = parser.get<string>("io");
	inArgs.dir = parser.get<string>("d");
	inArgs.outdir = parser.get<string>("do");
	inArgs.scale = parser.get<float>("s");
	inArgs.centered = parser.get<bool>("c");
	inArgs.display = parser.get<bool>("p");
	inArgs.denoise = parser.get<bool>("n");  

	if (!parser.check()) {
		cout << "command line parsing error. Usage:\n";
		parser.printMessage();
		exit(1);
	}

	string feedbackStr {};
	if (inArgs.img != "") {
		feedbackStr.append("using image: " + inArgs.img + '\n');
		feedbackStr.append("output image: " + inArgs.outimg + '\n');
	} else if (inArgs.dir != "") {
		feedbackStr.append("using dir: " + inArgs.dir + '\n');
		feedbackStr.append("output dir: " + inArgs.outdir + '\n');
	} else {
		cout << "no argument received and no default set, terminating.\n";
		exit(1);
	}

	cout << feedbackStr;
	cout << std::fixed; cout.precision(2);
	cout << "with scale: " << inArgs.scale << endl;
	cout << std::boolalpha;
	cout << "denoise crop: " << inArgs.denoise << endl;
	cout << "centered output: " << inArgs.centered << endl;
	cout << "display output: " << inArgs.display << endl;

	return inArgs;
}

//pre process the input image preparing it for boundary identification
Mat preProcessImage(const Mat& inMat, preprocParams params, bool log) {
	Mat imgProcess = inMat.clone();
	// convert to greyscale and resize for easy viewing
	if(log) {cout << "converting to grayscale" << endl;}
	cv::cvtColor(imgProcess, imgProcess, cv::COLOR_BGR2GRAY);
	if(log) {cout << "bluring image" << endl;}
	cv::GaussianBlur(imgProcess, imgProcess, params.gBlurSize, 0);
	cv::threshold(imgProcess, imgProcess, params.thr, params.thrMax, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
	if(log) {cout << "thresholding image" << endl;}
	Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, params.kernelSize);
	if(log) {cout << "dilating image" << endl;}
	cv::dilate(imgProcess,imgProcess,kernel);

	return imgProcess;
}

// check boundaries for buffer expand
int calcBuffer(Rect rect, int ph, int pw, int startBuff) {
	Point r_tl = rect.tl();
	Point r_br = rect.br();

	// find smallest of the distances
	int lstDist = std::min({r_tl.x, r_tl.y, pw-r_br.x, ph-r_br.y},
		[](int i1, int i2) {
			return i1 < i2;
		}
	);

	// use the smaller of lstDist and startBuffer
	return std::min(lstDist * 2, startBuff);
}

// get the contours of an image
contourResults getContours(const cv::Mat& img) {
	// check channel depth
	if (img.dims > 2) {
		cout << "getBounds received an image with color information, convert to single channel first\n";
		exit(1);
	}

	contourResults res;

	// find contours 
	cv::findContours(img, res.contours, res.hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// flatten contours
	for (auto c : res.contours) {
		for (auto p : c) {
			res.flatPoints.push_back(p);
		}
	}

	return res;
}	

//from the contours grab the global bounding rectangle
Rect getGlbBounds(const Mat &inImg, int bufferPix) {

	contourResults cntrs = getContours(inImg);
	Rect bndRect = cv::boundingRect(cntrs.flatPoints);
	int useBuffer = calcBuffer(bndRect, inImg.rows, inImg.cols, bufferPix);

	// shift rect by x + B/2, y + B/2, then enlarge by Size(B, B) --center expand
	bndRect = bndRect - Point {useBuffer / 2, useBuffer / 2};	// shift to NW
	bndRect += Size(useBuffer, useBuffer);	// expand to the SE

	return bndRect;
}

cv::RotatedRect getMinRotatedRect(const cv::Mat& img) {
	return cv::RotatedRect {};
}

// denoise the image
void cleanImage(Mat& inImg) {
	cv::fastNlMeansDenoising(inImg, inImg, 5.0f);
}

// process image procedure
Mat processImage(const string& imgPath, const cmdArgs& args) {
	bool log = args.img != "";
	
	// load image
	if(log) {cout << "loading image" << endl;}
	Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
	if(img.empty()) {
        cout << "Could not load the image: " << imgPath << endl;
        exit(1);
    }

	if(log) {cout << "pre-processing image" << endl;}
	Mat procImg = preProcessImage(img, preprocParams {Size(7,7), 150, 255, Size(5,15)}, log);

	// get the global exterior bounding rectangle
	if(log) {cout << "finding external boundary" << endl;}
	Rect bndRect = getGlbBounds(procImg);
	if(log) {cout << "bound rect size: " << bndRect.size() << endl;}

	// crop original image to bounding
	Mat imgCrop;
	if(!bndRect.empty()) {
		if(log) {cout << "cropping image" << endl;}
		imgCrop = img(
			cv::Range(bndRect.tl().y , bndRect.tl().y + bndRect.height),
			cv::Range(bndRect.tl().x , bndRect.tl().x + bndRect.width)
		);
	}

	// denoise
	if(log) {cout << "denoising the crop" << endl;}
	if(args.denoise && !imgCrop.empty()) { cleanImage(imgCrop); }
	
	// place cropped image into new blank
	if(log) {cout << "pasting croped image into blank form" << endl;}
	Mat imgOut = Mat(img.rows, img.cols, img.type(), cv::Scalar(255,255,255));
	if(!imgCrop.empty()) {
		if (args.centered) {
			imgCrop.copyTo(
				imgOut
				.rowRange((imgOut.rows - imgCrop.rows)/2, (imgOut.rows - imgCrop.rows)/2 + imgCrop.rows)
				.colRange((imgOut.cols - imgCrop.cols)/2, (imgOut.cols - imgCrop.cols)/2 + imgCrop.cols)
			);
		} else {
			imgCrop.copyTo(
				imgOut
				.rowRange(bndRect.tl().y, bndRect.tl().y + bndRect.height)
				.colRange(bndRect.tl().x, bndRect.tl().x + bndRect.width)
			);
		}
	}

	// TEST MARK ON IMAGES
	// cout << "drawing rectangle on image" << endl;
	// cv::rectangle(imgOut, bndRect, cv::Scalar(0,0,0), 2);
	

	// resize output if necessary
	if (args.scale != 1.0) {
		if(log) {cout << "resizing image" << endl;}
		cv::resize(imgOut, imgOut, Size(), args.scale, args.scale, cv::INTER_AREA);
	}

	return imgOut;
}
