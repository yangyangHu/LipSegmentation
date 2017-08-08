/**********************************************************************************
*
*					 唇色识别 LipColorRecognition
*					   by Hu yangyang 2016/12/29
*
***********************************************************************************/
#ifndef LIPCOLORRECOGNITION_H
#define LIPCOLORRECOGNITION_H

#include "Global.h"

class LipColorRecognition
{
public:
	LipColorRecognition();
	~LipColorRecognition();

	//返回嘴唇颜色预测值：0->淡白，1->淡红，2->红,3->暗红，4->紫
	int colorPredict(const CvScalar lipColor);

private:
	char* colorClassifierPathName;
};

#endif
