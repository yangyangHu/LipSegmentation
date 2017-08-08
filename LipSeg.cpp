/**********************************************************************************
*
*					嘴唇分割LipSegmentation
*            Robust Lip Segmentation Based on Complexion Mixture Model
*					   by Hu yangyang 2016/12/23
*
***********************************************************************************/
#include "LipSeg.h"
#include "RemoveNoise.h"

LipSegmentation::LipSegmentation(const IplImage* face):nGMM(5)
{
	DynamicScale(face);
	mComplexionGMM = NULL;
	faceDwn = NULL;
	faceDwn_Gauss = NULL;
	lipExtractMask = NULL;
	lipExtractImage = NULL;
	lipRawMask = NULL;
	lipImage = NULL;
}

LipSegmentation::~LipSegmentation()
{
	//release
	if(face != NULL) cvReleaseImage(&face);
	if(mComplexionGMM != NULL) delete mComplexionGMM;
	if(faceDwn != NULL) cvReleaseImage(&faceDwn);
	if(faceDwn_Gauss != NULL) cvReleaseImage(&faceDwn_Gauss);
	if(lipExtractMask != NULL) cvReleaseImage(&lipExtractMask);
	if(lipExtractImage != NULL) cvReleaseImage(&lipExtractImage);
	if(lipRawMask != NULL) cvReleaseImage(&lipRawMask);
	if(lipImage != NULL) cvReleaseImage(&lipImage);
}

//输入图像尺寸动态调整
void LipSegmentation::DynamicScale(const IplImage* inputImage)
{
	int size;
	size = (inputImage->height > inputImage->width) ? inputImage->height : inputImage->width;
	double type = ((double)size)/600.0;

	if (type<=1)
	{
		face = cvCloneImage(inputImage);
	}
	else
	{
		face = cvCreateImage(cvSize(cvRound(inputImage->width/type),cvRound(inputImage->height/type)),inputImage->depth,inputImage->nChannels);
		cvResize(inputImage,face);
	}
	//cvShowImage("face",face);
}

bool LipSegmentation::ProcessFlow()
{
	BuildComplexionGMM();
	ComputeComplexionProbabilityMap();

	CvRect rect_Lip = cvRect(0,0,0,0);
	IplImage* image_LipBi = NULL;
	if(!DetectLipRegion(image_LipBi,rect_Lip))
	{
		return false;//嘴唇检测失败
	}
	//cvShowImage("BIII",image_LipBi);

	OptimizationByGMMs(image_LipBi,rect_Lip);

	LipContourExtract(rect_Lip.width/10);

	//release
	cvReleaseImage(&image_LipBi);

	return true;//嘴唇检测成功
}

CvScalar LipSegmentation::ExtractLipColorFeature()
{
	//获得灰度图像
	IplImage* lipImageGray = cvCreateImage(cvGetSize(lipImage),8,1);
	cvCvtColor(lipImage,lipImageGray, CV_BGR2GRAY);
	//cvShowImage("Lgray",lipImageGray);

	//进一步去除一些明暗的像素
	int i_dark,i_bright;
	double rate_dark = 0.2, rate_bright = 0.02;
	calculateDarkAndBrightThreshold(lipRawMask,lipImageGray,i_dark,i_bright,rate_dark,rate_bright,0);
	for (int y=0;y<lipRawMask->height;++y)
	{
		for (int x=0;x<lipRawMask->width;++x)
		{
			if (cvGetReal2D(lipImageGray,y,x)<i_dark || cvGetReal2D(lipImageGray,y,x)>i_bright)
			{
				cvSetReal2D(lipRawMask,y,x,0);
			}
		}
	}
	cvErode(lipRawMask,lipRawMask);
	RemoveNoise rn;
	rn.LessConnectedRegionRemove(lipRawMask,lipRawMask->height*lipRawMask->width/30);
	//cvShowImage("lipRawMaskFine",lipRawMask);

	//convert to Lab
	IplImage* lipImage_Lab = cvCreateImage(cvGetSize(lipImage),8,3);
	cvCvtColor(lipImage,lipImage_Lab, CV_BGR2Lab);

	CvScalar lipColorFeature = cvAvg(lipImage_Lab,lipRawMask);

	//release
	cvReleaseImage(&lipImageGray);
	cvReleaseImage(&lipImage_Lab);

	return lipColorFeature;
}

//以上半脸和下部分脸中皮肤为训练样本建立肤色混合高斯模型
void LipSegmentation::BuildComplexionGMM()
{
	//缩放人脸，以加快GMM训练速度
	int len;
	len = (face->height > face->width) ? face->height : face->width;
	double type = ((double)len)/60.0;
	IplImage* faceScale = cvCreateImage(cvSize(cvRound(face->width/type),cvRound(face->height/type)),face->depth,face->nChannels);
	cvResize(face,faceScale);
	//cvShowImage("faceScale",faceScale);

	//获得灰度图像
	IplImage* faceImageScaleGray = cvCreateImage(cvGetSize(faceScale),8,1);
	cvCvtColor(faceScale,faceImageScaleGray, CV_BGR2GRAY);
	//cvShowImage("gray",faceImageScaleGray);

	//获取椭圆脸
	IplImage* faceScaleEllipseMask = cvCreateImage(cvGetSize(faceScale),8,1);
	cvZero(faceScaleEllipseMask);
	CvPoint center = cvPoint(cvRound(faceScaleEllipseMask->width*0.5),cvRound(faceScaleEllipseMask->height*0.5));
	CvSize size = cvSize(cvRound(faceScaleEllipseMask->width*0.36),cvRound(faceScaleEllipseMask->height*0.50));
	cvEllipse(faceScaleEllipseMask,center,size,0,0,360,cvScalar(255),CV_FILLED);//用来截取椭圆形的脸型
	//cvShowImage("ellipse",faceScaleEllipseMask);

	//去除下部分脸的嘴唇、胡子等非肤色噪声
	int i_dark,i_bright;
	double rate_dark = 0.2, rate_bright = 0;
	float lowPart = 0.65;
	calculateDarkAndBrightThreshold(faceScaleEllipseMask,faceImageScaleGray,i_dark,i_bright,rate_dark,rate_bright,lowPart);
	for (int y=faceScaleEllipseMask->height*lowPart;y<faceScaleEllipseMask->height;++y)
	{
		for (int x=0;x<faceScaleEllipseMask->width;++x)
		{
			if (cvGetReal2D(faceImageScaleGray,y,x)<i_dark)
			{
				cvSetReal2D(faceScaleEllipseMask,y,x,0);
			}
		}
	}
	cvErode(faceScaleEllipseMask,faceScaleEllipseMask);//erode
	//cvShowImage("ellipseFine",faceScaleEllipseMask);

	//取Lab颜色空间
	IplImage* faceImageScale_Lab = NULL;
	faceImageScale_Lab = cvCreateImage(cvGetSize(faceScale),8,3);
	cvCvtColor(faceScale,faceImageScale_Lab,CV_BGR2Lab); //设置颜色空间

	//以上部分脸和下部分脸中皮肤作为训练数据
	//准备训练数据
	uint cnt = 0,nrows = 0;
	for (int y=0;y<faceScaleEllipseMask->height;y++)
	{
		for (int x=0;x<faceScaleEllipseMask->width;x++)
		{
			if (y<=faceScaleEllipseMask->height*2/3)
			{
				nrows++;
			}
			else
			{
				if (cvGetReal2D(faceScaleEllipseMask,y,x)>200) nrows++;
			}
		}
	}

	double** data ;
	data = (double**)malloc(nrows*sizeof(double*));
	for (int i = 0; i < nrows; i++) data[i] = (double*)malloc(3*sizeof(double));//设置为三维高斯模型

	//	copy the data from the color array to a temp array 
	//	and assin each sample a random cluster id
	for (int y=0;y<faceScaleEllipseMask->height;y++)
	{
		for (int x=0;x<faceScaleEllipseMask->width;x++)
		{
			if (y<=faceScaleEllipseMask->height*2/3)
			{
				data[cnt][0] = cvGet2D(faceImageScale_Lab,y,x).val[0];//设置颜色空间的三分量用于高斯建模
				data[cnt][1] = cvGet2D(faceImageScale_Lab,y,x).val[1];
				data[cnt++][2] = cvGet2D(faceImageScale_Lab,y,x).val[2];
			}
			else
			{
				if (cvGetReal2D(faceScaleEllipseMask,y,x)>200)
				{
					data[cnt][0] = cvGet2D(faceImageScale_Lab,y,x).val[0];//设置颜色空间的三分量用于高斯建模
					data[cnt][1] = cvGet2D(faceImageScale_Lab,y,x).val[1];
					data[cnt++][2] = cvGet2D(faceImageScale_Lab,y,x).val[2];
				}
			}
		}
	}
	mComplexionGMM = new GMM(nGMM);
	mComplexionGMM->Build(data,nrows);

	for (int i = 0; i < nrows; i++) free(data[i]);
	free(data);
	
	//release
	cvReleaseImage(&faceScale);
	cvReleaseImage(&faceImageScaleGray);
	cvReleaseImage(&faceScaleEllipseMask);
	cvReleaseImage(&faceImageScale_Lab);
}

//利用肤色GMM求出下半脸的肤色概率图
void LipSegmentation::ComputeComplexionProbabilityMap()
{
	//获取下半脸
	cvSetImageROI(face,cvRect(0,face->height/2,face->width,face->height/2));
	faceDwn = cvCreateImage(cvGetSize(face),8,3);
	cvCopy(face,faceDwn);
	cvResetImageROI(face);
	//cvShowImage("faceDwn",faceDwn);

	//计算下半脸的肤色概率
	IplImage* faceDwn_Lab = NULL;
	faceDwn_Lab = cvCreateImage(cvGetSize(faceDwn),8,3);
	cvCvtColor(faceDwn,faceDwn_Lab,CV_BGR2Lab);//设置颜色空间

	faceDwn_Gauss = cvCreateImage(cvGetSize(faceDwn),IPL_DEPTH_64F,1);
	cvZero(faceDwn_Gauss);
	for (int y=0;y<faceDwn_Gauss->height;y++)
	{
		for (int x=0;x<faceDwn_Gauss->width;x++)
		{
			CvScalar pixel = cvGet2D(faceDwn_Lab,y,x);
			Color c(pixel.val[0],pixel.val[1],pixel.val[2]);//三维高斯模型的测试数据
			float px =  mComplexionGMM->p(c);
			cvSetReal2D(faceDwn_Gauss,y,x,px);
		}
	}
	cvNormalize(faceDwn_Gauss,faceDwn_Gauss,1.0,0.0,CV_C);

	//cvShowImage("ComplexionProbabilityOriginal",faceDwn_Gauss);

	cvSmooth(faceDwn_Gauss,faceDwn_Gauss,CV_GAUSSIAN,5,5);//改进

	//cvShowImage("ComplexionProbabilityFine",faceDwn_Gauss);

	//release
	cvReleaseImage(&faceDwn_Lab);
}

void LipSegmentation::OptimizationByGMMs(const IplImage* image_LipBi,const CvRect rect_Lip)
{
	cvSetImageROI(faceDwn,rect_Lip);
	lipImage = cvCreateImage(cvGetSize(faceDwn),8,3);
	cvCopy(faceDwn,lipImage);
	cvResetImageROI(faceDwn);

	IplImage* lipBi = cvCloneImage(image_LipBi);

	//cvShowImage("lip",lipImage);
	//cvShowImage("lipBi",lipBi);

	//缩放嘴唇图像，以加快GMM训练速度
	int len;
	len = (lipImage->height > lipImage->width) ? lipImage->height : lipImage->width;
	double type = ((double)len)/50.0;
	IplImage* lipImageScale = cvCreateImage(cvSize(cvRound(lipImage->width/type),cvRound(lipImage->height/type)),lipImage->depth,lipImage->nChannels);
	cvResize(lipImage,lipImageScale);
	//cvShowImage("lipImageScale",lipImageScale);

	IplImage* lipBiScale = cvCreateImage(cvSize(cvRound(lipBi->width/type),cvRound(lipBi->height/type)),lipBi->depth,lipBi->nChannels);
	cvResize(lipBi,lipBiScale);
	cvThreshold(lipBiScale, lipBiScale, 0, 255, CV_THRESH_BINARY| CV_THRESH_OTSU);//可考虑采用大津阈值
	//cvShowImage("lipBiScale",lipBiScale);

	//对嘴唇作为训练样本建立高斯模型
	IplImage* lipScale_Lab = NULL;
	lipScale_Lab = cvCreateImage(cvGetSize(lipImageScale),8,3);
	cvCvtColor(lipImageScale,lipScale_Lab,CV_BGR2Lab);//设置颜色空间

	uint cnt1 = 0,nrows1 = 0;
	double** data1 ;
	//nrows = faceTop_ycbcr->width*faceTop_ycbcr->height;
	for (int y=0;y<lipBiScale->height;y++)
	{
		for (int x=0;x<lipBiScale->width;x++)
		{
			if (cvGetReal2D(lipBiScale,y,x)>200)
			{
				nrows1++;
			}
		}
	}
	data1 = (double**)malloc(nrows1*sizeof(double*));
	for (int i = 0; i < nrows1; i++) data1[i] = (double*)malloc(3*sizeof(double));//设置三维GMM

	//	copy the data from the color array to a temp array 
	//	and assin each sample a random cluster id
	for (int y=0;y<lipBiScale->height;y++)
	{
		for (int x=0;x<lipBiScale->width;x++)
		{
			if (cvGetReal2D(lipBiScale,y,x)>200)
			{
				data1[cnt1][0] = cvGet2D(lipScale_Lab,y,x).val[0];
				data1[cnt1][1] = cvGet2D(lipScale_Lab,y,x).val[1];
				data1[cnt1++][2] = cvGet2D(lipScale_Lab,y,x).val[2];
			}
		}
	}
	GMM* mComplexionGMM1 = new GMM(3);
	mComplexionGMM1->Build(data1,nrows1);

	for (int i = 0; i < nrows1; i++) free(data1[i]);
	free(data1);

	//对嘴唇周围的皮肤作为训练样本建立高斯模型
	uint cnt2 = 0,nrows2 = 0;
	double** data2 ;
	//nrows = faceTop_ycbcr->width*faceTop_ycbcr->height;
	for (int y=0;y<lipBiScale->height;y++)
	{
		for (int x=0;x<lipBiScale->width;x++)
		{
			if (cvGetReal2D(lipBiScale,y,x)<100)
			{
				nrows2++;
			}
		}
	}
	data2 = (double**)malloc(nrows2*sizeof(double*));
	for (int i = 0; i < nrows2; i++) data2[i] = (double*)malloc(3*sizeof(double));//三维GMM

	//	copy the data from the color array to a temp array 
	//	and assin each sample a random cluster id
	for (int y=0;y<lipBiScale->height;y++)
	{
		for (int x=0;x<lipBiScale->width;x++)
		{
			if (cvGetReal2D(lipBiScale,y,x)<100)
			{
				data2[cnt2][0] = cvGet2D(lipScale_Lab,y,x).val[0];
				data2[cnt2][1] = cvGet2D(lipScale_Lab,y,x).val[1];
				data2[cnt2++][2] = cvGet2D(lipScale_Lab,y,x).val[2];
			}
		}
	}
	GMM* mComplexionGMM2 = new GMM(3);
	mComplexionGMM2->Build(data2,nrows2);

	for (int i = 0; i < nrows2; i++) free(data2[i]);
	free(data2);

	//使用高斯模型来处理下部分脸
	IplImage* lip_Lab = NULL;
	lip_Lab = cvCreateImage(cvGetSize(lipImage),8,3);
	cvCvtColor(lipImage,lip_Lab,CV_BGR2Lab);//设置颜色空间

	IplImage* lip_Gauss = cvCreateImage(cvGetSize(lip_Lab),IPL_DEPTH_64F,1);
	cvZero(lip_Gauss);
	for (int y=0;y<lip_Gauss->height;y++)
	{
		for (int x=0;x<lip_Gauss->width;x++)
		{
			CvScalar pixel = cvGet2D(lip_Lab,y,x);
			Color c(pixel.val[0],pixel.val[1],pixel.val[2]);
			float px_f =  mComplexionGMM1->p(c);
			float px_b =  mComplexionGMM2->p(c);
			float px = px_b/(px_f+px_b);
			cvSetReal2D(lip_Gauss,y,x,px);
		}
	}
	cvNormalize(lip_Gauss,lip_Gauss,1.0,0.0,CV_C);

	//cvShowImage("lip_Gauss",lip_Gauss);

	cvSmooth(lip_Gauss,lip_Gauss,CV_GAUSSIAN,5,5);//改进

	//cvShowImage("lip_GaussFine",lip_Gauss);

	IplImage* lip_similar = cvCreateImage(cvGetSize(lip_Gauss),8,1);
	cvScale(lip_Gauss,lip_similar,255);
	double biThreshold = cvThreshold(lip_similar,lip_similar,200,255,CV_THRESH_BINARY_INV|CV_THRESH_OTSU);
	//InitLipMakRefine(lip_similar);
	RemoveNoise rn;
	rn.LessConnectedRegionRemove(lip_similar,lip_similar->height*lip_similar->width/20);
	//cvShowImage("lip_similar",lip_similar);

	lipRawMask = cvCloneImage(lip_similar);

	//重映像到下半脸，获取下部分脸中嘴唇的mask
	lipExtractMask = cvCreateImage(cvGetSize(faceDwn),8,1);
	cvZero(lipExtractMask);
	for (int y=0;y<lip_similar->height;y++)
	{
		for (int x=0;x<lip_similar->width;x++)
		{
			int pixel = cvGetReal2D(lip_similar,y,x);
			cvSetReal2D(lipExtractMask,rect_Lip.y+y,rect_Lip.x+x,pixel);
		}
	}
	cvSmooth(lipExtractMask,lipExtractMask,CV_MEDIAN,5);//中值滤波去除椒盐噪声,使嘴唇边缘平滑
	//cvShowImage("faceDwnMask",lipExtractMask);

	//release
	cvReleaseImage(&lipBi);
	cvReleaseImage(&lipImageScale);
	cvReleaseImage(&lipBiScale);
	cvReleaseImage(&lipScale_Lab);
	delete mComplexionGMM1;
	delete mComplexionGMM2;
	cvReleaseImage(&lip_Lab);
	cvReleaseImage(&lip_Gauss);
	cvReleaseImage(&lip_similar);
}

//迭代检测嘴唇
bool LipSegmentation::DetectLipRegion(IplImage*& image_LipBi,CvRect& rect_Lip)
{
	//IplImage* faceDwnCpy = cvCloneImage(faceDwn);
	IplImage* faceDwn_Gray = cvCreateImage(cvGetSize(faceDwn),8,1);
	cvZero(faceDwn_Gray);
	cvAddS(faceDwn_Gray,cvScalar(255),faceDwn_Gray);
	double iterate_Threshold[16]={0.3,0.2,0.1,0.09,0.08,0.07,0.06,
		0.05,0.04,0.03,0.02,0.01,0.009,0.007,0.005,0.003};

	//粗略确定嘴唇中心点所在区域
	//cout<<"--嘴唇粗定位--"<<endl;
	CvRect lip_region = cvRect(0,0,0,0);
	bool flag_Lip = false;
	for (int i = 6;i<16;i++)
	{
		//IplImage* face_down2 = cvCloneImage(faceDwn);
		IplImage* faceDown_Gray2 = cvCloneImage(faceDwn_Gray);
		for (int y=0;y<faceDwn_Gauss->height;y++)
		{
			for (int x=0;x<faceDwn_Gauss->width;x++)
			{
				if(cvGetReal2D(faceDwn_Gauss,y,x)>=iterate_Threshold[i])
				{
					//cvSet2D(face_down2,y,x,CV_RGB(255,255,255));
					cvSetReal2D(faceDown_Gray2,y,x,0);
				}
			}
		}
		//cout<<"--"<<i<<"--"<<endl;
		//cvShowImage("itface",face_down2);
		//cvShowImage("itfaceBI",faceDown_Gray2);
		//cvWaitKey(0);

		//寻找初嘴唇
		IplImage* faceDown_GrayT = cvCloneImage(faceDown_Gray2);
		lip_region = LipLocate(faceDown_GrayT,flag_Lip);

		cvReleaseImage(&faceDown_GrayT);
		cvReleaseImage(&faceDown_Gray2);
		//cvReleaseImage(&face_down2);

		if (flag_Lip)
		{
			/*lip_region.y -= lip_region.height/2;
			lip_region.height += lip_region.height*1/2;
			lip_region.x -= lip_region.width/2;
			lip_region.width += lip_region.width;*/
			break;
		}
	}
	if (!flag_Lip)
	{
		for (int i = 5;i>=0;i--)
		{
			//IplImage* face_down2 = cvCloneImage(faceDwn);
			IplImage* faceDown_Gray2 = cvCloneImage(faceDwn_Gray);
			for (int y=0;y<faceDwn_Gauss->height;y++)
			{
				for (int x=0;x<faceDwn_Gauss->width;x++)
				{
					if(cvGetReal2D(faceDwn_Gauss,y,x)>=iterate_Threshold[i])
					{
						//cvSet2D(face_down2,y,x,CV_RGB(255,255,255));
						cvSetReal2D(faceDown_Gray2,y,x,0);
					}
				}
			}
			//cout<<"--"<<i<<"--"<<endl;
			//cvShowImage("itface",face_down2);
			//cvShowImage("itfaceBI",faceDown_Gray2);
			//cvWaitKey(0);

			//寻找初嘴唇
			IplImage* faceDown_GrayT = cvCloneImage(faceDown_Gray2);
			lip_region = LipLocate(faceDown_GrayT,flag_Lip);

			cvReleaseImage(&faceDown_GrayT);
			cvReleaseImage(&faceDown_Gray2);
			//cvReleaseImage(&face_down2);

			if (flag_Lip)
			{
				/*lip_region.y -= lip_region.height/2;
				lip_region.height += lip_region.height*1/2;
				lip_region.x -= lip_region.width/2;
				lip_region.width += lip_region.width;*/
				break;
			}
		}
	}
	//嘴唇初定位异常，异常处理
	if (!flag_Lip)
	{
		//cout<<"嘴唇初定位异常！"<<endl;
		return flag_Lip;
	}

	flag_Lip = false;
	//精确确定嘴唇区域
	//cout<<"--嘴唇精定位--"<<endl;
	int it;
	for (it=0;it<16;it=it+1)
	{
		for (int y=0;y<faceDwn_Gauss->height;y++)
		{
			for (int x=0;x<faceDwn_Gauss->width;x++)
			{
				if(cvGetReal2D(faceDwn_Gauss,y,x)>=iterate_Threshold[it])
				{
					//cvSet2D(faceDwnCpy,y,x,CV_RGB(255,255,255));
					cvSetReal2D(faceDwn_Gray,y,x,0);
				}
			}
		}
		//cout<<"--"<<it<<"--"<<endl;
		//cvShowImage("itface",faceDwnCpy);
		//cvShowImage("itfaceBI",faceDwn_Gray);
		//cvWaitKey(0);

		//寻找初嘴唇
		IplImage* faceDown_GrayT = cvCloneImage(faceDwn_Gray);
		rect_Lip = LipAccurateLocate(faceDown_GrayT,flag_Lip,lip_region);
		cvReleaseImage(&faceDown_GrayT);

		if (flag_Lip)
		{
			break;
		}
	}
	//嘴唇初定位异常，异常处理
	if (!flag_Lip)
	{
		//cout<<"嘴唇初定位异常！"<<endl;
		return flag_Lip;
	}

	cvSetImageROI(faceDwn_Gray,rect_Lip);
	IplImage* image_Lip = cvCreateImage(cvGetSize(faceDwn_Gray),8,1);
	cvCopy(faceDwn_Gray,image_Lip);
	cvResetImageROI(faceDwn_Gray);

	cvSetImageROI(faceDwn_Gauss,rect_Lip);
	IplImage* image_Gauss = cvCreateImage(cvGetSize(faceDwn_Gauss),faceDwn_Gauss->depth,1);
	cvCopy(faceDwn_Gauss,image_Gauss);
	cvResetImageROI(faceDwn_Gauss);

	//继续迭代优化
	//cout<<"--继续迭代优化--"<<endl;
	image_LipBi = cvCloneImage(image_Lip);
	int s1,s2,sd;
	s1 = LipLocateRefine(image_Lip);

	int sdFlagPre,sdFlagCur;
	int sdItPre = it,sdItCur = it;
	int sdPre,sdCur;

	int continuousIncreaseCnt = 0;
	int continuousDecreaseCnt = 0;

	int nLipRefine;
	for (int i = it+1;i<16;i++)
	{
		for (int y=0;y<image_Lip->height;y++)
		{
			for (int x=0;x<image_Lip->width;x++)
			{
				if(cvGetReal2D(image_Gauss,y,x)>=iterate_Threshold[i])
				{
					//cvSet2D(faceDwnCpy,rect_Lip.y+y,rect_Lip.x+x,CV_RGB(255,255,255));
					cvSetReal2D(image_Lip,y,x,0);
				}
			}
		}

		//cout<<"--"<<i<<"--"<<endl;
		//cvShowImage("itface",faceDwnCpy);
		//cvShowImage("itLipBI",image_Lip);
		//cvWaitKey(0);

		s2 = LipLocateRefine(image_Lip);
		sd = s1 - s2;
		s1 = s2;

		if (i==it+1)
		{
			sdPre = sd;
			sdFlagPre = 0;
			sdItPre = i;
			sdItCur = i;
		}
		else
		{
			sdCur = sd;
			sdItCur = i;
			if (sdCur-sdPre>0)//单调递增
			{
				sdFlagCur = INCREASE;
				++continuousIncreaseCnt;
				continuousDecreaseCnt = 0;
			}
			else              //单调递减
			{
				sdFlagCur = DECREASE;
				continuousIncreaseCnt = 0;
				++continuousDecreaseCnt;
			}

			if (continuousIncreaseCnt>=3||continuousDecreaseCnt>=3)//连续多次上升或者下降，使嘴唇丢失太多，停止迭代
			{
				break;
			}

			if ((sdFlagPre == DECREASE)&&(sdFlagCur == INCREASE))//寻找第一个局部最小值点
			{
				//if (sdPre<=750)//突变点sd值<=700->750   改进10
				//{
				break;
				//}
				//else
				//{
				//sdPre = sdCur;//更新
				//sdItPre = sdItCur;
				//sdFlagPre = sdFlagCur;
				//}

			}
			else
			{
				sdPre = sdCur;//更新
				sdItPre = sdItCur;
				sdFlagPre = sdFlagCur;
			}
		}
		//判断嘴唇能否检测到
		{
			IplImage* faceDownGrayRefine = cvCreateImage(cvGetSize(faceDwn),8,1);
			cvZero(faceDownGrayRefine);
			for (int y=0;y<image_Lip->height;y++)
			{
				for (int x=0;x<image_Lip->width;x++)
				{
					int pixel = cvGetReal2D(image_Lip,y,x);
					cvSetReal2D(faceDownGrayRefine,rect_Lip.y+y,rect_Lip.x+x,pixel);
				}
			}
			nLipRefine = LipDetect(faceDownGrayRefine);//获取疑似嘴唇的个数
			if (nLipRefine == 0)//未检测到嘴唇，则返回上一步迭代的嘴唇
			{
				if (sdItPre>0)
				{
					sdItPre--;
				}
				sdItCur = (it+sdItPre)/2;
				break;
			}
			cvReleaseImage(&faceDownGrayRefine);
		}
	}
	//局部最优值的嘴唇
	double T = 0.0;
	if (nLipRefine==0)
	{
		T = (iterate_Threshold[sdItPre] + iterate_Threshold[sdItCur]/*+iterate_Threshold[it]*/)/2;
	}
	else
	{
		T = (iterate_Threshold[sdItPre] + iterate_Threshold[sdItPre]/*+iterate_Threshold[it]*/)/2;
	}
	for (int y=0;y<image_LipBi->height;y++)
	{
		for (int x=0;x<image_LipBi->width;x++)
		{
			if(cvGetReal2D(image_Gauss,y,x)>=T)
			{
				cvSetReal2D(image_LipBi,y,x,0);
			}
		}
	}
	cvSmooth(image_LipBi,image_LipBi,CV_MEDIAN);//中值滤波去除椒盐噪声	
	//cvMorphologyEx(image_LipBi,image_LipBi,NULL,NULL,CV_MOP_CLOSE,1);
	//去除粗嘴唇mask中杂质，只留下嘴唇粗mask
	InitLipMakRefine(image_LipBi);
	//cvShowImage("lipRegion",image_LipBi);

	//release
	cvReleaseImage(&faceDwn_Gray);
	//cvReleaseImage(&faceDwnCpy);
	cvReleaseImage(&image_Lip);
	cvReleaseImage(&image_Gauss);

	return flag_Lip;
}


void LipSegmentation::LipContourExtract(const int maxStep)
{
	//去除突出杂质
	IplImage* lipContourAnd = FivePointAverage(lipExtractMask,5);
	FillContour(lipContourAnd);
	for (int num=10;num<=maxStep;num=num+5)
	{
		IplImage* lipContour_num = FivePointAverage(lipExtractMask,num);
		FillContour(lipContour_num);
		//and运算
		cvAnd(lipContourAnd,lipContour_num,lipContourAnd);
		cvReleaseImage(&lipContour_num);
	}
	InitLipMakRefine(lipContourAnd);
	//cvShowImage("and",lipContourAnd);

	//填补内凹
	IplImage* lipContourOr = FivePointAverage(lipContourAnd,5);
	FillContour(lipContourOr);
	//cvShowImage("or",lipContourOr);

	for (int num=5;num<=maxStep;num=num+5)
	{
		IplImage* lipContour_num = FivePointAverage(lipContourAnd,num);
		FillContour(lipContour_num);
		//or运算
		cvOr(lipContourOr,lipContour_num,lipContourOr);
		cvReleaseImage(&lipContour_num);
	}

	//cvShowImage("or",lipContourOr);

	//更新嘴唇lipExtractMask，获得最优嘴唇mask
	cvReleaseImage(&lipExtractMask);
	lipExtractMask = cvCloneImage(lipContourOr);
	//cvShowImage("LipMask",lipExtractMask);

	//分割出嘴唇
	SegLip();
	//cvShowImage("lipExtract",lipExtractImage);

	//release
	cvReleaseImage(&lipContourAnd);
	cvReleaseImage(&lipContourOr);
}

//嘴唇初定位
CvRect LipSegmentation::LipLocate(IplImage* faceDown_Gray,bool &falg_Lip)
{
	CvSeq *pContour = NULL; 
	CvMemStorage *pStorage = cvCreateMemStorage(0); 
	int n=cvFindContours(faceDown_Gray,pStorage,&pContour,sizeof(CvContour),CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
	CvRect lipRect = cvRect(0,0,0,0);
	int nlip = 0;//疑似嘴唇的个数
	for(;pContour!=NULL;pContour=pContour->h_next)
	{
		int area=(int)cvContourArea(pContour);
		CvRect rect=cvBoundingRect(pContour);//计算点集的最外面（up-right）矩形边界 
		double ratio_WH = 0;//外界矩形的宽高比
		double ratio_FILL = 0;//目标填充率
		double ratio_AREA  = 0;//目标区域面积与整个图像面积之比
		CvPoint center;
		ratio_WH = ((double)(rect.width))/((double)(rect.height));
		ratio_FILL = ((double)area)/((double)(rect.width*rect.height));
		ratio_AREA = ((double)area)/((double)(faceDown_Gray->width*faceDown_Gray->height));
		center.x = rect.x+rect.width/2;
		center.y = rect.y+rect.height/2;
		if ((ratio_WH>1)&&(ratio_WH<=6)&&(ratio_FILL>=0.3)&&(ratio_FILL<1)&&(ratio_AREA>=0.02)&&(ratio_AREA<=0.15)&&(center.x>=faceDown_Gray->width/3)&&(center.x<=faceDown_Gray->width*2/3)&&(center.y>=faceDown_Gray->height/3))
		{
			lipRect = rect;
			nlip++;
		}
	}
	if (nlip == 1)
	{
		falg_Lip = true;//找到嘴唇
	}
	else
	{
		falg_Lip = false;//未找到嘴唇
	}
	//释放内存
	cvReleaseMemStorage(&pStorage);
	return lipRect;
}

//嘴唇精确定位
CvRect LipSegmentation::LipAccurateLocate(IplImage* faceDown_Gray,bool &falg_Lip,CvRect lip_region)
{
	CvSeq *pContour = NULL; 
	CvMemStorage *pStorage = cvCreateMemStorage(0); 
	int n=cvFindContours(faceDown_Gray,pStorage,&pContour,sizeof(CvContour),CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
	CvRect lipRect = cvRect(0,0,0,0);
	int nlip = 0;//疑似嘴唇的个数
	for(;pContour!=NULL;pContour=pContour->h_next)
	{
		int area=(int)cvContourArea(pContour);
		CvRect rect=cvBoundingRect(pContour);//计算点集的最外面（up-right）矩形边界 
		double ratio_WH = 0;//外界矩形的宽高比
		double ratio_FILL = 0;//目标填充率
		double ratio_AREA  = 0;//目标区域面积与整个图像面积之比
		CvPoint center;
		ratio_WH = ((double)(rect.width))/((double)(rect.height));
		ratio_FILL = ((double)area)/((double)(rect.width*rect.height));
		ratio_AREA = ((double)area)/((double)(faceDown_Gray->width*faceDown_Gray->height));
		center.x = rect.x+rect.width/2;
		center.y = rect.y+rect.height/2;
		if ((ratio_WH>1)&&(ratio_WH<=6)&&(ratio_FILL>=0.3)&&(ratio_FILL<1)&&(ratio_AREA>=0.02)&&(ratio_AREA<=0.15)&&(center.x>=faceDown_Gray->width/3)&&(center.x<=faceDown_Gray->width*2/3)&&(center.y>=faceDown_Gray->height/3)&&(center.y<=lip_region.y+lip_region.height)&&(center.y>=lip_region.y)&&(center.x<=lip_region.x+lip_region.width)&&(center.x>=lip_region.x))
		{
			lipRect = rect;
			nlip++;
		}
	}
	if (nlip == 1)
	{
		falg_Lip = true;//找到嘴唇
	}
	else
	{
		falg_Lip = false;//未找到嘴唇
	}
	//释放内存
	cvReleaseMemStorage(&pStorage);
	return lipRect;
}

//优化嘴唇
int LipSegmentation::LipLocateRefine(IplImage* image_lip)
{
	int s = 0;
	for (int y=0;y<image_lip->height;y++)
	{
		for (int x=0;x<image_lip->width;x++)
		{
			if (cvGetReal2D(image_lip,y,x)>200)
			{
				s++;
			}
		}
	}
	return s;
}

//嘴唇检测，用于嘴唇优化时检测嘴唇是否存在
int LipSegmentation::LipDetect(IplImage* faceDown_Gray)
{
	CvSeq *pContour = NULL; 
	CvMemStorage *pStorage = cvCreateMemStorage(0); 
	int n=cvFindContours(faceDown_Gray,pStorage,&pContour,sizeof(CvContour),CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
	int nlip = 0;//疑似嘴唇的个数
	for(;pContour!=NULL;pContour=pContour->h_next)
	{
		int area=(int)cvContourArea(pContour);
		CvRect rect=cvBoundingRect(pContour);//计算点集的最外面（up-right）矩形边界 
		double ratio_WH = 0;//外界矩形的宽高比
		double ratio_FILL = 0;//目标填充率
		double ratio_AREA  = 0;//目标区域面积与整个图像面积之比
		CvPoint center;
		ratio_WH = ((double)(rect.width))/((double)(rect.height));
		ratio_FILL = ((double)area)/((double)(rect.width*rect.height));
		ratio_AREA = ((double)area)/((double)(faceDown_Gray->width*faceDown_Gray->height));
		if ((ratio_WH>1)&&(ratio_WH<=6)&&(ratio_FILL>=0.25)&&(ratio_FILL<1)&&(ratio_AREA>=0.022)&&(ratio_AREA<=0.15))
		{
			nlip++;
		}
	}
	//释放内存
	cvReleaseMemStorage(&pStorage);
	return nlip;
}

//去除粗嘴唇mask中杂质干扰，得到唯一嘴唇mask
void LipSegmentation::InitLipMakRefine(IplImage* mask)
{
	IplImage* mask1 = cvCloneImage(mask);
	IplImage* mask2 = cvCloneImage(mask);

	//确定嘴唇区域
	CvSeq *pContour1 = NULL; 
	CvMemStorage *pStorage1 = cvCreateMemStorage(0); 
	int n1=cvFindContours(mask1,pStorage1,&pContour1,sizeof(CvContour),CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
	int areaMax = 0;
	for(;pContour1!=NULL;pContour1=pContour1->h_next)
	{
		int area=(int)cvContourArea(pContour1);
		if (area>areaMax)
		{
			areaMax = area;
		}
	}

	//去除面积较小的轮廓
	CvSeq *pContour2 = NULL; 
	CvMemStorage *pStorage2 = cvCreateMemStorage(0); 
	int n2=cvFindContours(mask2,pStorage2,&pContour2,sizeof(CvContour),CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
	for(;pContour2!=NULL;pContour2=pContour2->h_next)
	{
		int area=(int)cvContourArea(pContour2);
		CvRect rect=cvBoundingRect(pContour2);//计算点集的最外面（up-right）矩形边界 
		if (area<areaMax)
		{
			cvSetImageROI(mask,rect);
			cvSetZero(mask);
			cvResetImageROI(mask);
		}
	}

	//释放内存
	cvReleaseMemStorage(&pStorage1);
	cvReleaseMemStorage(&pStorage2);
	cvReleaseImage(&mask1);
	cvReleaseImage(&mask2);
}

//五点平均法，平滑嘴唇边缘
IplImage* LipSegmentation::FivePointAverage(IplImage* mask,int step)//mask为嘴唇掩膜（唯一轮廓，无内空洞），step为步长
{
	IplImage* lipContour = cvCreateImage(cvGetSize(mask),8,1);
	cvZero(lipContour);
	/***********对上嘴唇的上轮廓，从左往右每隔step个点取一个点***************/
	CvPoint lipTopStartPoint = cvPoint(0,0),lipTopEndPoint = cvPoint(0,0);
	//求上嘴唇上轮廓最左边的点
	for (int x=0;x<mask->width;x++)
	{
		bool flag = false;
		for (int y=0;y<mask->height;y++)
		{
			int pixel = cvGetReal2D(mask,y,x);
			if (pixel>200)
			{
				lipTopStartPoint.x = x;
				lipTopStartPoint.y = y;
				flag = true;
				break;
			}
		}
		if (flag)
		{
			break;
		}
	}
	//求上嘴唇上轮廓最右边的点
	for (int x=mask->width-1;x>=0;x--)
	{
		bool flag = false;
		for (int y=0;y<mask->height;y++)
		{
			int pixel = cvGetReal2D(mask,y,x);
			if (pixel>200)
			{
				lipTopEndPoint.x = x;
				lipTopEndPoint.y = y;
				flag = true;
				break;
			}
		}
		if (flag)
		{
			break;
		}
	}
	//对上嘴唇的上轮廓每隔step个点取一个典型点，并将这些典型点连接起来
	CvPoint point1 = lipTopStartPoint,point2 = cvPoint(0,0);
	for (int x=lipTopStartPoint.x+step;x<=lipTopEndPoint.x;x=x+step)
	{
		for (int y=0;y<mask->height;y++)
		{
			int pixel = cvGetReal2D(mask,y,x);
			if (pixel>200)
			{
				point2.x = x;
				point2.y = y;
				break;
			}
		}
		//用直线把point1和point2连接起来
		cvLine(lipContour,point1,point2,cvScalarAll(255),1,8);
		point1 = point2;
	}
	cvLine(lipContour,point2,lipTopEndPoint,cvScalarAll(255),1,8);
	/***********对下嘴唇的下轮廓，从左往右每隔step个点取一个点***************/
	CvPoint lipDwnStartPoint = cvPoint(0,0),lipDwnEndPoint = cvPoint(0,0);
	//求下嘴唇下轮廓最左边的点
	for (int x=0;x<mask->width;x++)
	{
		bool flag = false;
		for (int y=mask->height-1;y>=0;y--)
		{
			int pixel = cvGetReal2D(mask,y,x);
			if (pixel>200)
			{
				lipDwnStartPoint.x = x;
				lipDwnStartPoint.y = y;
				flag = true;
				break;
			}
		}
		if (flag)
		{
			break;
		}
	}
	//求下嘴唇下轮廓最右边的点
	for (int x=mask->width-1;x>=0;x--)
	{
		bool flag = false;
		for (int y=mask->height-1;y>=0;y--)
		{
			int pixel = cvGetReal2D(mask,y,x);
			if (pixel>200)
			{
				lipDwnEndPoint.x = x;
				lipDwnEndPoint.y = y;
				flag = true;
				break;
			}
		}
		if (flag)
		{
			break;
		}
	}
	//对下嘴唇的下轮廓每隔step个点取一个典型点，并将这些典型点连接起来
	point1 = lipDwnStartPoint;
	for (int x=lipDwnStartPoint.x+step;x<=lipDwnEndPoint.x;x=x+step)
	{
		for (int y=mask->height-1;y>=0;y--)
		{
			int pixel = cvGetReal2D(mask,y,x);
			if (pixel>200)
			{
				point2.x = x;
				point2.y = y;
				break;
			}
		}
		//用直线把point1和point2连接起来
		cvLine(lipContour,point1,point2,cvScalarAll(255),1,8);
		point1 = point2;
	}
	cvLine(lipContour,point2,lipDwnEndPoint,cvScalarAll(255),1,8);

	//把嘴唇左右两角连起来构成一个封闭轮廓
	cvLine(lipContour,lipTopStartPoint,lipDwnStartPoint,cvScalarAll(255),1,8);
	cvLine(lipContour,lipTopEndPoint,lipDwnEndPoint,cvScalarAll(255),1,8);

	//cvShowImage("contour",lipContour);

	return lipContour;
}

//填充轮廓
void LipSegmentation::FillContour(IplImage* contour)
{
	IplImage* contourCopy = cvCloneImage(contour);
	CvSeq *pContour = NULL; 
	CvSeq *pConInner = NULL;  
	CvMemStorage *pStorage = cvCreateMemStorage(0); 
	int n=cvFindContours(contourCopy,pStorage,&pContour,sizeof(CvContour),CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
	cvDrawContours(contour,pContour,CV_RGB(255,255,255),CV_RGB(255,255,255),2,CV_FILLED,8);
	// 外轮廓循环   
	for (; pContour != NULL; pContour = pContour->h_next)   
	{   
		// 内轮廓循环   
		for (pConInner = pContour->v_next; pConInner != NULL; pConInner = pConInner->h_next)   
		{      
			cvDrawContours(contour, pConInner, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 0, CV_FILLED, 8, cvPoint(0, 0));  
		}   
	}   
	cvReleaseMemStorage(&pStorage);  
	cvReleaseImage(&contourCopy);
}

//分割出嘴唇区域
void LipSegmentation::SegLip()
{
	lipExtractImage = cvCloneImage(faceDwn);
	for (int y=0;y<lipExtractMask->height;y++)
	{
		for (int x=0;x<lipExtractMask->width;x++)
		{
			if (cvGetReal2D(lipExtractMask,y,x)<100)
			{
				cvSet2D(lipExtractImage,y,x,CV_RGB(0,0,0));
			}
		}
	}
}

void LipSegmentation::calculateDarkAndBrightThreshold(const IplImage* mask,const IplImage* grayImage,int& i_dark,int& i_bright,double rate_dark,double rate_bright,float lowPart)
{
	//hash table
	float data[256] = {0};
	int total = 0;
	for (int y=mask->height*lowPart;y<mask->height;y++)
	{
		for (int x=0;x<mask->width;x++)
		{
			if (cvGetReal2D(mask,y,x)>200)
			{
				int intensity = cvRound(cvGetReal2D(grayImage,y,x));
				data[intensity]++;
				total++;
			}
		}
	}

	//舍去较暗的rate_dark
	float sum_dark =0;
	for (int i=0;i<256;i++)
	{
		sum_dark += data[i];
		if ((sum_dark)/((double)total)>rate_dark)
		{
			i_dark = i;
			break;
		}
	}
	//printf("T_dark=%d\n",i_dark);

	//舍去较亮的rate_bright
	float sum_bright =0;
	for (int i=255;i>=0;i--)
	{
		sum_bright += data[i];
		if ((sum_bright)/((double)total)>rate_bright)
		{
			i_bright = i;
			break;
		}
	}
	//printf("i_bright=%d\n",i_bright);

	//colorhist(data,i_dark,i_bright);//直方图显示
}

//int LipSegmentation::colorhist(float *data,int i_dark,int i_bright)//画直方图（实际应用程序中不需要）
//{
//	int hist_size = 256;
//	int hist_height = 200;
//	float range[] = {0,255};
//	float *ranges[] = {range};
//	//创建一维直方图
//	CvHistogram* hist = cvCreateHist(1,&hist_size,CV_HIST_ARRAY,ranges,1);
//
//	//根据已给定的数据创建直方图
//	cvMakeHistHeaderForArray(1,&hist_size,hist,data,ranges,1);
//	//归一化直方图
//	cvNormalizeHist(hist,1.0);
//
//	//创建一张一维直方图的“图”，横坐标为灰度级，纵坐标为像素个数
//	int scale = 2;
//	IplImage* hist_image = cvCreateImage(cvSize(hist_size*scale,hist_height),8,3);
//	cvZero(hist_image);
//
//	//统计直方图中的最大bin
//	float max_value = 0;
//	cvGetMinMaxHistValue(hist,0,&max_value,0,0);
//
//	//分别将每个bin的值绘制在图中
//	for (int i=0;i<hist_size;i++)
//	{
//		float bin_val = cvQueryHistValue_1D(hist,i);
//		int intensity = cvRound(bin_val*hist_height/max_value);//要绘制的高度
//		if (i==i_dark||i==i_bright)
//		{
//			cvRectangle(hist_image,cvPoint(i*scale,hist_height-1),cvPoint((i+1)*scale-1,hist_height-intensity),CV_RGB(255,0,0));
//		}
//		else
//		{
//			cvRectangle(hist_image,cvPoint(i*scale,hist_height-1),cvPoint((i+1)*scale-1,hist_height-intensity),CV_RGB(0,0,255));
//		}
//	}
//
//	cvShowImage("hist",hist_image);
//
//	return 0;
//}

IplImage* LipSegmentation::GetLipMask()
{
	return lipExtractMask;
}

IplImage* LipSegmentation::GetLipImage()
{
	return lipExtractImage;
}
