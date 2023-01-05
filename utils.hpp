/*
    collection of miscellaneous utilites
    to de-clutter main

*/

#pragma once


#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>


// grab all of the .jpg files found in the path
std::vector<std::string> getDirFiles(const std::string&);

// print formatted progress bar
void printProgress(int, int);

// take in milliseconds and return a formatted elapsed time hh:mm:ss:mmm
std::string getElapsedTime(int);
