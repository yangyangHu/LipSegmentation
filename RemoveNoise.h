/*****************************************************************************************
*
*				  去噪：RemoveNoise
*				 By 胡洋洋 2015/10/15
*
*****************************************************************************************/
#ifndef REMOVENOISE_H
#define REMOVENOISE_H

#include "Global.h"

class RemoveNoise
{
public:
	RemoveNoise();
	~RemoveNoise();
	//去除面积小于area的连通域（比区域增长法更高效）
	void LessConnectedRegionRemove(IplImage* image,int area);

	//去除裂纹检测图中的干扰噪声（用于裂纹检测）
	//输入：image-->裂纹检测结果二值化图，area-->噪声最大的面积(或线的最小面积),lineLength-->线的最小长度
	//返回值：返回满足条件的裂纹数量
	int RemoveCrackImageNoise(IplImage* image,int area,int lineLength);
private:
};
#endif