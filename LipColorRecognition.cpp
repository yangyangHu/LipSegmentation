/**********************************************************************************
*
*					 唇色识别 LipColorRecognition
*					   by Hu yangyang 2016/12/29
*
***********************************************************************************/

#include "LipColorRecognition.h"

LipColorRecognition::LipColorRecognition()
{
	//工程应用时，修改成模型实际加载路径!!!
	colorClassifierPathName = "model\\svm_lipcolor.xml";
}

LipColorRecognition::~LipColorRecognition(){}

int LipColorRecognition::colorPredict(const CvScalar lipColor)
{
	int response = 2;
	float feature[3];//<L,a,b>
	feature[0] = lipColor.val[0];//L
	feature[1] = lipColor.val[1];//a
	feature[2] = lipColor.val[2];//b

	//设置测试数据
	Mat testDataMat(1,3,CV_32FC1,feature);  //测试数据
	//预测
	CvSVM svm = CvSVM();
	svm.load(colorClassifierPathName);
	response = (int)svm.predict(testDataMat);

	return response;
}
