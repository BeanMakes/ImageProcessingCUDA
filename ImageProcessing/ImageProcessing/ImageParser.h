#include <vector>
#ifndef IMAGEPARSER_H
#define IMAGEPARSER_H
#include <iostream>


class ImageParser
{
private:

	char* m_file;

public:

	ImageParser(char* file);

	unsigned char* readBMP();

	std::vector < std::vector <int>> turnGreyScale(std::vector< std::vector< std::vector<int>>> arr);

	std::vector< std::vector< std::vector<int>>> readBMPToArray();
};
#endif
