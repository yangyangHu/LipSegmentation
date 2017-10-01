/*****************************************************************************************
*
*					  去噪：RemoveNoise
*					By 胡洋洋 2015/10/15
*
*****************************************************************************************/

#include "RemoveNoise.h"

RemoveNoise::RemoveNoise(){}

RemoveNoise::~RemoveNoise(){}

//去除面积小于area的连通域（比区域增长法更高效）
void RemoveNoise::LessConnectedRegionRemove(IplImage* image,int area)
{
	IplImage* src = cvCloneImage(image);
	CvMemStorage* storage = cvCreateMemStorage(0);  
	CvSeq* contour = 0; 
	//提取轮廓   
	cvFindContours( src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); 
	cvZero( image );//清空
	double minarea = (double)area;  
	for( ; contour != 0; contour = contour->h_next )  
	{  

		double tmparea=fabs(cvContourArea(contour));  
		if(tmparea < minarea)   
		{  
			cvSeqRemove(contour,0); //删除面积小于设定值的轮廓   
			continue;  
		}  
  
		CvScalar color = CV_RGB( 255, 255,255 );  

		//max_level 绘制轮廓的最大等级。如果等级为0，绘制单独的轮廓。如果为1，绘制轮廓及在其后的相同的级别下轮廓。   
		//如果值为2，所有的轮廓。如果等级为2，绘制所有同级轮廓及所有低一级轮廓，诸此种种。   
		//如果值为负数，函数不绘制同级轮廓，但会升序绘制直到级别为abs(max_level)-1的子轮廓。    
		cvDrawContours( image, contour, color, color, -1, CV_FILLED, 8 , cvPoint(0, 0));//绘制外部和内部的轮廓   
	}  

	//release
	cvReleaseImage(&src);
	cvReleaseMemStorage(&storage);  
}

//去除裂纹检测图中的干扰噪声（用于裂纹检测）
//输入：image-->裂纹检测结果二值化图，area-->噪声最大的面积(或线的最小面积),lineLength-->线的最小长度
//返回值：返回满足条件的裂纹数量
int RemoveNoise::RemoveCrackImageNoise(IplImage* image,int area,int lineLength)
{
	IplImage* src = cvCloneImage(image);
	CvMemStorage* storage = cvCreateMemStorage(0);  
	CvSeq* contour = 0; 
	//提取轮廓   
	cvFindContours( src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); 
	cvZero( image );//清空
	double minarea = (double)area;  
	int nCracks = 0;//裂纹数量
	for( ; contour != 0; contour = contour->h_next )  
	{  
		CvRect rect=cvBoundingRect(contour);//计算点集的最外面（up-right）矩形边界 
		int len = rect.height>rect.width ? rect.height : rect.width;
		double tmparea=fabs(cvContourArea(contour));  
		if(tmparea < minarea || len < lineLength)   
		{  
			cvSeqRemove(contour,0); //删除面积小于设定值的轮廓   
			continue;  
		}  
		nCracks++;
		CvScalar color = CV_RGB( 255, 255,255 );  

		//max_level 绘制轮廓的最大等级。如果等级为0，绘制单独的轮廓。如果为1，绘制轮廓及在其后的相同的级别下轮廓。   
		//如果值为2，所有的轮廓。如果等级为2，绘制所有同级轮廓及所有低一级轮廓，诸此种种。   
		//如果值为负数，函数不绘制同级轮廓，但会升序绘制直到级别为abs(max_level)-1的子轮廓。    
		cvDrawContours( image, contour, color, color, -1, CV_FILLED, 8 , cvPoint(0, 0));//绘制外部和内部的轮廓   
	}  

	//release
	cvReleaseImage(&src);
	cvReleaseMemStorage(&storage);  
	return nCracks;
}