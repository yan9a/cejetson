/////////////////////////////////////////////////////////////////////////////
// File: ceCvtIm.h
// Description: Image Format Conversion Between wxWidgets, OpenCV, and CUDA
// 	This process involves converting images between three different frameworks: 
// 		wxWidgets for GUI-based image handling, 
// 		OpenCV for image processing, and 
// 		CUDA for GPU-accelerated operations. 
// 	The workflow typically starts with an image in wxWidgets format, 
// 	which is converted into an OpenCV Mat for image processing tasks. 
// 	From OpenCV, the image can be transferred to a CUDA-accelerated format 
// 	for fast GPU computations. 
// 		After processing on the GPU, the result is converted back to 
// 	OpenCV format and finally returned to wxWidgets for display. 
// 	Proper handling of memory layout, data types, and pixel formats 
// 	(e.g., RGB, BGR) is essential for seamless conversion between these environments.
// WebSite: https://yan9a.github.io/cejetson/
// MIT License (https://opensource.org/licenses/MIT)
// Copyright (c) 2024 Yan Naing Aye
//
// References:
// 	https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md
/////////////////////////////////////////////////////////////////////////////


#ifndef CECVTIM_H
#define CECVTIM_H

#include <string>
#include <opencv2/opencv.hpp>

#define CE_CUDA 1
#define CE_WX 1

#if CE_WX==1
 #include <wx/wx.h>
#endif

#if CE_CUDA==1
 #include <jetson-utils/loadImage.h>
 #include <jetson-utils/cudaMappedMemory.h>
#endif

namespace ce {

class ceCvtIm {
private:
public:
#if CE_WX==1
	static wxImage mat2wx(cv::Mat &img) {
		cv::Mat im2;
		if(img.channels()==1) {cvtColor(img,im2,cv::COLOR_GRAY2RGB);}
		else if (img.channels() == 4) { cvtColor(img, im2, cv::COLOR_BGRA2RGB);}
		else {cvtColor(img,im2,cv::COLOR_BGR2RGB);}
		size_t imsize = im2.rows*im2.cols*im2.channels();
		wxImage wx(im2.cols, im2.rows,(unsigned char*)malloc(imsize), false);
		memcpy((unsigned char*)wx.GetData(), (unsigned char*)im2.data, imsize);
		return wx;
	};

	static cv::Mat wx2mat(wxImage &wx) {
		cv::Mat im2(cv::Size(wx.GetWidth(),wx.GetHeight()),CV_8UC3,wx.GetData());
		cv::cvtColor(im2,im2,cv::COLOR_RGB2BGR);
		return im2;	
	};
#endif

#if CE_CUDA==1
	// cv::Mat to cuda IMAGE_RGB8 
	static bool mat2cuda(const cv::Mat &imgMat, uchar3*& imgCuda, int& width, int& height){
		cv::Mat im2;
		if(imgMat.channels()==1) {cvtColor(imgMat,im2,cv::COLOR_GRAY2RGB);}
                else if (imgMat.channels() == 4) { cvtColor(imgMat, im2, cv::COLOR_BGRA2RGB);}
                else {cvtColor(imgMat,im2,cv::COLOR_BGR2RGB);}
                size_t imsize = im2.rows*im2.cols*im2.channels();
		width = im2.cols;
		height = im2.rows;
		if(!cudaAllocMapped(&imgCuda,width,height,IMAGE_RGB8)){
			printf("Failed to allocate mapped cuda memory\n");
			return false;
		}
		if( CUDA_FAILED(cudaMemcpy(imgCuda, im2.data, imsize, cudaMemcpyHostToDevice)) ){
			printf("Memory cpy error \n");
			return false;
		}
		return true;
	};

	static cv::Mat cuda2mat(const uchar3* imgCuda, int width, int height) {
		size_t imsize = width * height * sizeof(uchar3);
		uint8_t* img = (uint8_t*)malloc(imsize);
		//copy image from GPU
		cudaMemcpy(img, imgCuda, imsize, cudaMemcpyDeviceToHost);
		cv::Mat im2(cv::Size(width,height),CV_8UC3,img);
		cv::Mat im3;
		cvtColor(im2,im3,cv::COLOR_BGR2RGB);
		free(img);
		return im3;
	};

#endif	

};


}

#endif //CECVTIM_H
