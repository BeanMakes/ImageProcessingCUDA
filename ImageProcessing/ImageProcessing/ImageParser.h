#ifndef IMAGEPARSER_H
#define IMAGEPARSER_H
#include <iostream>


class ImageParser
{
private:

	char* m_file;

public:

	ImageParser(char* file);

	void getRBGOfImage();

	unsigned char* readBMP();
};
#endif
