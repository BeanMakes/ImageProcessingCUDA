#include "ImageParser.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include<array> 
using namespace std;
ImageParser::ImageParser(char* file)
	: m_file{ file }

{
}

unsigned char* ImageParser::readBMP()
{
    int i;
    FILE* f = fopen(m_file, "rb");

    if (f == NULL)
        throw "Argument Exception";

    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    cout << endl;
    cout << "  Name: " << m_file << endl;
    cout << " Width: " << width << endl;
    cout << "Height: " << height << endl;

    int row_padded = (width * 3 + 3) & (~3);
    unsigned char* data = new unsigned char[row_padded];
    unsigned char tmp;

    for (int i = 0; i < height; i++)
    {
        fread(data, sizeof(unsigned char), row_padded, f);
        for (int j = 0; j < width * 3; j += 3)
        {
            // Convert (B, G, R) to (R, G, B)
            tmp = data[j];
            data[j] = data[j + 2];
            data[j + 2] = tmp;

            //cout << "R: " << (int)data[j] << " G: " << (int)data[j + 1] << " B: " << (int)data[j + 2] << endl;
        }
    }

    fclose(f);

    cout << "Length: " << row_padded << endl;
    return data;
}

std::vector< std::vector< std::vector<int>>> ImageParser::readBMPToArray()
{
    int i;
    FILE* f = fopen(m_file, "rb");

    if (f == NULL)
        throw "Argument Exception";

    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    cout << endl;
    cout << "  Name: " << m_file << endl;
    cout << " Width: " << width << endl;
    cout << "Height: " << height << endl;
    std::vector< std::vector< std::vector<int>>> arr(height,vector<vector<int>>(width,vector<int>(3,1)));

    int row_padded = (width * 3 + 3) & (~3);
    unsigned char* data = new unsigned char[row_padded];
    unsigned char tmp;

    for (int i = 0; i < height; i++)
    {
        fread(data, sizeof(unsigned char), row_padded, f);
        int jVal = 0;
        for (int j = 0; j < width; j += 1)
        {
            // Convert (B, G, R) to (R, G, B)
            tmp = data[jVal];
            data[jVal] = data[jVal + 2];
            data[jVal + 2] = tmp;

            arr[i][j][0] = (int)data[jVal];
            arr[i][j][1] = (int)data[jVal+1];
            arr[i][j][2] = (int)data[jVal + 2];

            jVal += 3;
            //cout << "R: " << (int)data[j] << " G: " << (int)data[j + 1] << " B: " << (int)data[j + 2] << endl;
        }
    }

    fclose(f);

    return arr;


}