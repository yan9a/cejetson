// File: ceimnet.cpp
// Description: An example to use OpenCV, wxWidgets, and CUDA
// Author: Yan Naing Aye
// Date: 2024 October 12
// MIT License - Copyright (c) 2024 Yan Naing Aye
// Source: https://github.com/yan9a/cejetson
// References
// https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md

#include <wx/wx.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <jetson-inference/imageNet.h>
#include <jetson-utils/loadImage.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "ce/ceCvtIm.h"
using namespace ce;
using namespace std;
using namespace cv;

//-----------------------------------------------------------------------------
void TestCuda(Mat &img)
{

	// mat 2 cuda
	
	uchar3* im1 = nullptr;
	int w1 = 0;
	int h1 = 0;

	if(!ceCvtIm::mat2cuda(img,im1,w1,h1)){
		printf("Fail to convert mat 2 cuda\n");
		return;		
	}
	const char* imgCudaOut = "./testcuda.jpg";

        printf("Saving cuda img \n");
        if(!saveImage(imgCudaOut,im1,w1,h1)) {
              printf("failed to save image '%s'\n", imgCudaOut);
        }

	// cuda 2 mat
	const char* imgFilename = "./test.jpg";
        uchar3* imgCuda;
        int imgWidth = 0;
        int imgHeight = 0;

        if(!loadImage(imgFilename, &imgCuda, &imgWidth, &imgHeight)){
                printf("failed to load image '%s'\n", imgFilename);
        }

	Mat im2 = ceCvtIm::cuda2mat(imgCuda,imgWidth,imgHeight);
	imwrite("./cvimgoutput.jpg",im2);
	//imshow("cuda to mat",ceCvtIm::cuda2mat(imgCuda,imgWidth,imgHeight));
}
//-----------------------------------------------------------------------------
class MyFrame : public wxFrame
{
	wxStaticBitmap *thiri;
public:
	MyFrame(const wxString& title);

};
MyFrame::MyFrame(const wxString& title)
	: wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(512, 600))
{
	Centre();
	string fname = "./thiri.png";
	Mat imcv1 = imread(fname,IMREAD_UNCHANGED);	

	if (!imcv1.data) {
		// fail to read img
		imcv1 = Mat::zeros(512, 512, CV_8UC3);
		printf("Failed to read image %s\n",fname.c_str());
	}
	//From opencv to wx
	string str = "Channels:" + to_string(imcv1.channels());
	putText(imcv1, str, Point(100, 100), FONT_HERSHEY_PLAIN, 4.0, CV_RGB(128, 0, 128), 4.0);
	wxBitmap imwx1 = ceCvtIm::mat2wx(imcv1);

	// thiri = new wxStaticBitmap(this, wxID_ANY, wxBitmap(wxT("./thiri.png"), wxBITMAP_TYPE_PNG), wxPoint(256, 0), wxSize(512,512));
	thiri = new wxStaticBitmap(this, wxID_ANY, imwx1, wxPoint(256, 0), wxSize(512,512));
	// thiri->SetBitmap(imwx1);
	TestCuda(imcv1);
}

class MyApp : public wxApp
{
public:
	virtual bool OnInit();
};

IMPLEMENT_APP(MyApp)
bool MyApp::OnInit()
{
	if (!wxApp::OnInit())
		return false;
	wxInitAllImageHandlers();
	MyFrame *frame = new MyFrame(wxT("OpenCV and Cuda"));
	frame->Show(true);

	return true;
}

