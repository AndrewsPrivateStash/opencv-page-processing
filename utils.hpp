/*
    collection of miscellaneous utilites
    to de-clutter main

*/

#pragma once


#include <iostream>
#include <vector>
#include <filesystem>


// grab all of the .jpg files found in the path
std::vector<std::string> getDirFiles(const std::string&);

// produce a completion percentage string from the passed counts
float checkProgress(int, int);

// print formatted progress bar
void printProgress(int, int);

// take in milliseconds and return a formatted elapsed time hh:mm:ss:mmm
std::string getElapsedTime(int);
