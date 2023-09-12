#include "QrLocate.h"

Mat DetectQrcode::DetQr_RotatePreprocess(Mat& _src)
{
	resize(_src, _src, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC); // ˫���β�ֵ

	Mat srcCopy = _src.clone();
	Mat srcGray = ImageIsGray(_src);;

	vector<Point> center_all;

	//threshold(srcGray, srcGray, 0, 255, THRESH_BINARY | THRESH_OTSU);
	threshold(srcGray, srcGray, 133, 255, THRESH_BINARY );
	Canny(srcGray, srcGray, 50, 160, 3);
	Mat erodeStruct = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(srcGray, srcGray, MORPH_CLOSE, erodeStruct);

	return srcGray;
}

vector<RotatedRect> DetectQrcode::DetQr_filterByNestedContours(Mat& _srcImg, Mat& _binaryImg)
{
	Mat image = _srcImg.clone();
	Mat show_image = _srcImg.clone();
	Mat binary_image = _binaryImg.clone();

	//�ҵ��Ӹ�����
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary_image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	//��ʼ��������ϻ��ֵ���ת����
	vector<RotatedRect> filterRect;


	//��ʼ��С���������
	int numOfRec = 0;

	// ��ⷽ��
	int ic = 0;
	int parentIdx = -1;
	for (int i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][2] != -1 && ic == 0)
		{
			parentIdx = i;
			ic++;
		}
		else if (hierarchy[i][2] != -1)
		{
			ic++;
		}
		else if (hierarchy[i][2] == -1)
		{
			parentIdx = -1;
			ic = 0;
		}
		if (ic >= 3)
		{
			if (IsQrPoint(contours[parentIdx], image))
			{
				RotatedRect rect = minAreaRect(Mat(contours[parentIdx]));

				// ��ͼ����
				Point2f points[4];
				rect.points(points);
				for (int j = 0; j < 4; j++)
				{
					line(show_image, points[j], points[(j + 1) % 4], Scalar(0, 255, 0), 2);
				}

				//drawContours(canvas, contours, parentIdx, Scalar(0, 0, 255), -1);

				// ������������������ת����
				filterRect.push_back(rect);


				numOfRec++;
			}
			ic = 0;
			parentIdx = -1;
		}
	}
	return filterRect;
}

// �� for i  { for j }����д��ʽ���е��ѿ�
// ����ÿ��RotatedRect����������ҽ��������ֵ��20���ڵķֳ�һ�� ���·Ž�  vector<vector<RotatedRect>> 

void DetectQrcode::DetQr_GetFourPoints(Point2f& TopLeft, Point2f& TopRight, Point2f& BottomRight, Point2f& BottomLeft,Point2f &srcCenter, vector<RotatedRect>& _filterAreaRect)
{
	vector<RotatedRect> filterRect = _filterAreaRect;
	vector < Point2f> Points3;
	Point2f Point4;
	for (int i = 0; i < 3; i++)
	{
		Points3.push_back(GetRelativePoint(filterRect[i], srcCenter));
	}
	//����������нǶ�˳ʱ������

	Points3 = sortClockwise(Points3);
	//������ĸ����λ��
	double dist12 = std::sqrt(std::pow(Points3[0].x - Points3[1].x, 2) + std::pow(Points3[0].y - Points3[1].y, 2));
	double dist13 = std::sqrt(std::pow(Points3[0].x - Points3[2].x, 2) + std::pow(Points3[0].y - Points3[2].y, 2));
	double dist23 = std::sqrt(std::pow(Points3[1].x - Points3[2].x, 2) + std::pow(Points3[1].y - Points3[2].y, 2));


	//�ҵ����
	double maxLength = std::max(dist12, std::max(dist13, dist23));
	if (maxLength == dist12)
	{
		Point4.x = (Points3[0].x + Points3[1].x - Points3[2].x);
		Point4.y = (Points3[0].y + Points3[1].y - Points3[2].y);
	}
	else if (maxLength == dist13)
	{
		Point4.x = Points3[0].x + Points3[2].x - Points3[1].x;
		Point4.y = Points3[0].y + Points3[2].y - Points3[1].y;
	}
	else
	{
		Point4.x = Points3[0].x + Points3[2].x - Points3[1].x;
		Point4.y = Points3[0].y + Points3[2].y - Points3[1].y;
	}
	Points3.push_back(Point4);

	Points3 = sortClociWiseByXY(Points3);
	TopLeft.x = Points3[0].x - 5;
	TopLeft.y = Points3[0].y - 5;

	TopRight.x = Points3[1].x + 5;
	TopRight.y = Points3[1].y - 5;

	BottomRight.x = Points3[2].x + 5;
	BottomRight.y = Points3[2].y + 5;

	BottomLeft.x = Points3[3].x - 5;
	BottomLeft.y = Points3[3].y + 5;

}

Mat DetectQrcode::DetQr_CropRotateQrimg(Mat &_srcImg, vector<RotatedRect> &_vRect)
{
	//���������
	Point2f topRight;
	Point2f topLeft;
	Point2f bottomRight;
	Point2f bottomLeft;

	//�������ͼ���������ĵ㣬�������ֽǵ�
	int srcCX = _srcImg.size().width / 2;
	int srcCY = _srcImg.size().height / 2;
	Point2f srcCenterPoint;
	srcCenterPoint.x = srcCX;
	srcCenterPoint.y = srcCY;

	// ����պõ���3 ��ֱ�Ӽ����3���ǵ��Լ����ĸ��ǵ� Ȼ�����͸�ӱ任
	if (_vRect.size() == 3) 
	{

		vector < Point2f> Points3;
		Point2f Point4;
		for (int i = 0; i < 3; i++)
		{
			Points3.push_back(GetRelativePoint(_vRect[i], srcCenterPoint));
		}
		//����������нǶ�˳ʱ������

		Points3 = sortClockwise(Points3);
		//������ĸ����λ��
		double dist12 = std::sqrt(std::pow(Points3[0].x - Points3[1].x, 2) + std::pow(Points3[0].y - Points3[1].y, 2));
		double dist13 = std::sqrt(std::pow(Points3[0].x - Points3[2].x, 2) + std::pow(Points3[0].y - Points3[2].y, 2));
		double dist23 = std::sqrt(std::pow(Points3[1].x - Points3[2].x, 2) + std::pow(Points3[1].y - Points3[2].y, 2));


		//�ҵ����
		double maxLength = std::max(dist12, std::max(dist13, dist23));
		if (maxLength == dist12)
		{
			Point4.x = (Points3[0].x + Points3[1].x - Points3[2].x);
			Point4.y = (Points3[0].y + Points3[1].y - Points3[2].y);
		}
		else if (maxLength == dist13)
		{
			Point4.x = Points3[0].x + Points3[2].x - Points3[1].x;
			Point4.y = Points3[0].y + Points3[2].y - Points3[1].y;
		}
		else
		{
			Point4.x = Points3[0].x + Points3[2].x - Points3[1].x;
			Point4.y = Points3[0].y + Points3[2].y - Points3[1].y;
		}
		Points3.push_back(Point4);

		Points3 = sortClociWiseByXY(Points3);
		topLeft.x = Points3[0].x - 5;
		topLeft.y = Points3[0].y - 5;

		topRight.x = Points3[1].x + 5;
		topRight.y = Points3[1].y - 5;

		bottomRight.x = Points3[2].x + 5;
		bottomRight.y = Points3[2].y + 5;

		bottomLeft.x = Points3[3].x - 5;
		bottomLeft.y = Points3[3].y + 5;
	}
	//���size �ĸ�����4-9֮�䣬 ����100/500 �ַ��Ķ�ά��
	else if (_vRect.size() >= 4 ) 
	{
		vector<vector<RotatedRect>>  classifiedRectangles;

		for (int i = 0; i < _vRect.size(); i++)
		{
			int area = _vRect[i].size.width * _vRect[i].size.height;

			bool foundClass = false;

			for (int j = 0; j < classifiedRectangles.size(); j++)
			{
				if (!classifiedRectangles[j].empty())
				{
					double classArea = classifiedRectangles[j][0].size.width * classifiedRectangles[j][0].size.height;

					if (abs(area - classArea) <= (area/7))
					{
						classifiedRectangles[j].push_back(_vRect[i]);
						foundClass = true;
						break;
					}
				}
			}

			if (!foundClass)
			{
				classifiedRectangles.push_back({ _vRect[i] });
			}
		}


		// ��һ��������ֵ �õ��������һ��
		double maxAvgArea = 0.0;
		vector<RotatedRect> filterAreaRect; //Ԥ���������õ�һ����ת�������������һ��
		for (int n = 0; n < classifiedRectangles.size(); n++)
		{
			double avgArea = calculateAverageArea(classifiedRectangles[n]); //����һ����ת�����е�ƽ�����
			if (avgArea > maxAvgArea)
			{
				maxAvgArea = avgArea;
				filterAreaRect = classifiedRectangles[n];
			}

		}
		//����һ���ǰ�����ó�����һ��ǵ�λ�ü��+������ĸ���Ĳ���
		vector <RotatedRect> RotaedRect3;
		vector <Point2f> Points3;
		Point2f Point4;

		// ������
		if (filterAreaRect.size() < 3)
			return _srcImg;

		// ������
		DetQr_GetFourPoints(topLeft, topRight, bottomRight, bottomLeft, srcCenterPoint, filterAreaRect);

	}
	else
	{
		topLeft.x = 0;
		topLeft.y = 0;
		topRight.x = srcCX * 2;
		topRight.y = 0;
		bottomRight.x = srcCX * 2;
		bottomRight.y = srcCY * 2;
		bottomLeft.x = 0;
		bottomLeft.y = srcCY * 2;
	}

	// ��ͼ
	Mat roi;
	vector<Point2f> finalPoint4 =
	{
			topLeft,
			topRight,
			bottomRight,
			bottomLeft
	};

	(finalPoint4, 6);

	int width = calculateDistance(topLeft, topRight) + 8;
	int height = calculateDistance(topLeft, bottomLeft) + 8;

	std::vector<cv::Point2f> dstPoints =
	{
			cv::Point2f(0, 0),
			cv::Point2f(width - 1, 0),
			cv::Point2f(width - 1, height - 1),
			cv::Point2f(0, height - 1)
	};
	cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(finalPoint4, dstPoints);

	// ����͸�ӱ任
	cv::Mat result;

	cv::warpPerspective(_srcImg, result, perspectiveMatrix, cv::Size(width, height));

	roi = result;



	return roi;
}



DetectQrcode::DetectQrcode()
{
}

DetectQrcode::~DetectQrcode()
{
}