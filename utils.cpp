
#include "utils.hpp"

using std::vector, std::string, std::cout, std::endl;

// get jpeg files in the path
vector<string> getDirFiles(const string& dirPath) {
	vector<string> outFiles;
	for (const auto & entry : std::filesystem::directory_iterator(dirPath)) {
		string fileStr = entry.path().filename();
		if (fileStr.find(".jpg") != std::string::npos) {
			outFiles.push_back(fileStr);
		}
	}
	return outFiles;   
}

// print progress bar to stdout buffer and flush
void printProgress(int tot, int cur) {
    const int SYMBOL_WIDTH = 20;
    const char SYMBOL = '#';

    float progress = (float)cur / (float)tot;
    cout << "\r[";
    int symCnt = std::ceil(SYMBOL_WIDTH * progress);
    cout << string(symCnt, '#') << string(SYMBOL_WIDTH - symCnt, ' ') << ']';
    cout << " " << std::fixed << std::setprecision(1) << progress * 100 << "%";
    cout.flush();
}

// calculate elapsed time from milliseconds
string getElapsedTime(int milis) {

	int rem = milis;
	int hours = (milis / (1000 * 60 * 60)); rem -= hours * 60 * 60 * 1000;
	int min = (rem / (1000 * 60)); rem -= min * 60 * 1000;
	int sec = (rem / 1000); rem -= sec * 1000;

	std::stringstream outStr;
	outStr << std::setw(2) << std::setfill('0') << hours << ":";
	outStr << std::setw(2) << std::setfill('0') << min << ":";
	outStr << std::setw(2) << std::setfill('0') << sec << ":";
	outStr << std::setw(3) << std::setfill('0') << rem;

	return outStr.str();
}