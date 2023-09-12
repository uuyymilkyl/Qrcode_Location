/* adapted from  https://github.com/bubbliiiing/QRcode-location */
/* author: uuyymilkyl */

#include <iostream>    
#include "QrLocate.h"

using namespace cv;
using namespace std;

int main()
{

	while (1)
	{
		Mat src;
		Mat QRimg;
		src = imread("8.bmp",0);

		Mat binary = DetectQrcode::DetQr_RotatePreprocess(src);
		vector<RotatedRect> rects = DetectQrcode::DetQr_filterByNestedContours(src, binary);
		QRimg = DetectQrcode::DetectQrcode::DetQr_CropRotateQrimg(src, rects);
			(src, rects);
		// Õ¹Ê¾Í¼Ïñ
		imshow("QRcode", QRimg);

		waitKey(10000);
	}
	return 0;
}