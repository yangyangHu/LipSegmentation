# LipSegmentation
implementation of the paper: Robust Lip Segmentation Based on Complexion Mixture Model

#Opencv 2.4.4

# example
    //faceImage is the ROI (Region Of Interest) in the input image using face detection
		LipSegmentation lipPro(faceImage);
		bool isLip = lipPro.ProcessFlow();
		if (isLip)
		{
			lipDetectRes = 1;//嘴唇检测成功设为1

			IplImage* lipExtractImage = lipPro.GetLipImage();
			//cvShowImage("lipExtractImage",lipExtractImage);

			CvScalar lipColorFeature = lipPro.ExtractLipColorFeature();
			
			LipColorRecognition lipColorRecognition;
			lipColor = lipColorRecognition.colorPredict(lipColorFeature);
		}
		else
		{
			lipDetectRes = 0;//嘴唇检测失败设为0
		}
