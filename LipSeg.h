/**********************************************************************************
*
*					嘴唇分割LipSegmentation
*             Robust Lip Segmentation Based on Complexion Mixture Model
*					   by Hu yangyang 2016/12/23
*
***********************************************************************************/
#ifndef LIPSEG_H
#define LIPSEG_H

#include "Global.h"
#include "GMM.h"

#define DECREASE -1   //单调递减
#define INCREASE 1    //单调递增

class LipSegmentation
{
public:
	LipSegmentation(const IplImage* face);
	~LipSegmentation();
	bool ProcessFlow();
	CvScalar ExtractLipColorFeature();//提取嘴唇颜色值(L,a,b)
	IplImage* GetLipMask();
	IplImage* GetLipImage();
protected:
	void BuildComplexionGMM();
	void ComputeComplexionProbabilityMap();
	bool DetectLipRegion(IplImage*& image_LipBi,CvRect& rect_Lip);
	void OptimizationByGMMs(const IplImage* image_LipBi,const CvRect rect_Lip);
	void LipContourExtract(const int maxStep);
private:
	void calculateDarkAndBrightThreshold(const IplImage* mask,const IplImage* grayImage,int& i_dark,int& i_bright,double rate_dark,double rate_bright,float lowPart);
	//int colorhist(float *data,int i_dark,int i_bright);
	//输入图像尺寸动态调整
	void DynamicScale(const IplImage* inputImage);
	//嘴唇初定位
	CvRect LipLocate(IplImage* faceDown_Gray,bool &falg_Lip);
	//嘴唇精确定位
	CvRect LipAccurateLocate(IplImage* faceDown_Gray,bool &falg_Lip,CvRect lip_region);
	//优化嘴唇
	int LipLocateRefine(IplImage* image_lip);
	//嘴唇检测，用于嘴唇优化时检测嘴唇是否存在
	int LipDetect(IplImage* faceDown_Gray);
	//去除粗嘴唇mask中杂质干扰，得到唯一嘴唇mask
	void InitLipMakRefine(IplImage* mask);
	//五点平均法，去除突起物，平滑嘴唇边缘
	IplImage* FivePointAverage(IplImage* mask,int step);//mask为嘴唇掩膜（唯一轮廓，无内空洞），step为步长
	//填充轮廓
	void FillContour(IplImage* contour);
	//分割出嘴唇区域
	void SegLip();
private:
	IplImage* face;
	IplImage* faceDwn;
	IplImage* faceDwn_Gauss;//Complexion Probability Map
	IplImage* lipExtractMask;//lip extracted mask
	IplImage* lipExtractImage;//lip extracted image

	IplImage* lipRawMask;
	IplImage* lipImage;

	const int nGMM;
	GMM* mComplexionGMM;
};

#endif
