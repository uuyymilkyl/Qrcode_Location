#ifndef _CODE93_H_
#define _CODE93_H_
#include <opencv2/opencv.hpp>
#include <iostream>    

using namespace cv;
using namespace std;

class  DetectQrcode
{
public:
	DetectQrcode();
	~DetectQrcode();

public:

	static Mat DetQr_RotatePreprocess(Mat& _src);

	static vector<RotatedRect> DetQr_filterByNestedContours(Mat& _srcImg, Mat& _binaryImg);   ///< ɸѡ��Ļ���

	static Mat DetQr_CropRotateQrimg(Mat &_srcImg, vector<RotatedRect> &_vRect);      ///< ��ԭͼ������ת���ü�



};

bool comparePoints(const cv::Point2f& point1, const cv::Point2f& point2) {
	return (point1.x + point1.y) < (point2.x + point2.y);
}
static bool compareXPoints(const Point2f& p1, const Point2f& p2)
{
	return p1.x < p2.x;
}

static bool compareYPoints(const Point2f& p1, const Point2f& p2)
{
	return p1.x < p2.x;
}

static bool comparePointsy(const cv::Point& p1, const cv::Point& p2)
{
	if (p1.y < p2.y)
	{
		return true;
	}
	else if (p1.y > p2.y)
	{

		return false;
	}
	else
	{
		return p1.x < p2.x;
	}

}

static void expandQuadrilateral(vector<Point2f>& points, int offset)
{

	// ������������
	Point2f leftVec = points[3] - points[0];
	// �����Ҳ������
	Point2f rightVec = points[2] - points[1];

	// �������ߵĵ�λ������
	Point2f leftNormal = Point2f(-leftVec.y, leftVec.x);
	leftNormal /= norm(leftNormal);

	// �����Ҳ�ߵĵ�λ������
	Point2f rightNormal = Point2f(rightVec.y, -rightVec.x);
	rightNormal /= norm(rightNormal);

	// ����ƫ�����������ĸ��µĶ�������
	Point2f topLeft = points[0] - leftNormal * offset;
	Point2f topRight = points[1] + rightNormal * offset;
	Point2f bottomRight = points[2] + leftNormal * offset;
	Point2f bottomLeft = points[3] - rightNormal * offset;

	// �����ĸ��������
	points[0] = topLeft;
	points[1] = topRight;
	points[2] = bottomRight;
	points[3] = bottomLeft;

}

static int calculateDistance(const Point2f& point1, const Point2f& point2)
{
	/* ����������֮���ֱ�߾��� */
	double distance = std::sqrt(std::pow(point2.x - point1.x, 2) + std::pow(point2.y - point1.y, 2));
	return static_cast<int>(std::round(distance));
}


// �ж��Ƿ�Ҷ�ͼ ������ת�ɻҶ�ͼ  ���ߣ�uuyymilkyl
static Mat ImageIsGray(const Mat& _image)
{
	Mat img = _image.clone();
	if (img.channels() == 3)
	{
		cvtColor(img, img, COLOR_BGR2GRAY);
	}
	return img;
}

static double calculateAverageArea(const vector<RotatedRect>& rects)
{
	double totalArea = 0.0;
	for (int i =0; i < rects.size(); i++)
	{
		double area = rects[i].size.width * rects[i].size.height;
		totalArea += area;
	}
	return totalArea / rects.size();
}

// �����ڲ����а�ɫ����ռȫ���ı���  ���ߣ�bubbliiiing
static double Rate(Mat& count)
{
	int number = 0;
	int allpixel = 0;
	for (int row = 0; row < count.rows; row++)
	{
		for (int col = 0; col < count.cols; col++)
		{
			if (count.at<uchar>(row, col) == 255)
			{
				number++;
			}
			allpixel++;
		}
	}
	//cout << (double)number / allpixel << endl;
	return (double)number / allpixel;
}

// �����ж��Ƿ����ڽ��ϵ�������  ���ߣ�bubbliiiing
static bool IsCorner(Mat& image)
{
	// ����mask
	Mat imgCopy, dstCopy;
	Mat dstGray;
	imgCopy = image.clone();

	// ת��Ϊ�Ҷ�ͼ��
	cvtColor(image, dstGray, COLOR_BGR2GRAY);
	// ���ж�ֵ��

	threshold(dstGray, dstGray, 0, 255, THRESH_BINARY | THRESH_OTSU);
	dstCopy = dstGray.clone();  //����

	Mat erodeStruct = getStructuringElement(MORPH_RECT, Size(3, 3));
	//morphologyEx(dstCopy, dstCopy, MORPH_OPEN, erodeStruct);
	// �ҵ������봫�ݹ�ϵ
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(dstCopy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);


	for (int i = 0; i < contours.size(); i++)
	{
		//cout << i << endl;
		if (hierarchy[i][2] == -1 && hierarchy[i][3] != -1)
		{

			Rect rect = boundingRect(Mat(contours[i]));
			rectangle(image, rect, Scalar(0, 0, 255), 2);

			// ������ľ�����������ľ��εĶԱ�
			if (rect.width < imgCopy.cols * 2 / 7)      //2/7��Ϊ�˷�ֹһЩ΢С�ķ���
				continue;
			if (rect.height < imgCopy.rows * 2 / 7)      //2/7��Ϊ�˷�ֹһЩ΢С�ķ���
				continue;

			// �ж����к�ɫ���ɫ�Ĳ��ֵı���
			if (Rate(dstGray) > 0.20)
			{
				return true;
			}
		}
	}
	return  false;
}

// ���ڵõ����ֽ�  ���ߣ�
static Mat transformCorner(Mat src, RotatedRect rect)
{
	// �����ת����
	Point center = rect.center;
	// ������ϽǺ����½ǵĽǵ㣬����Ҫ��֤������ͼƬ��Χ�����ڿ�ͼ
	Point TopLeft = Point(cvRound(center.x), cvRound(center.y)) - Point(rect.size.height / 2, rect.size.width / 2);  //��ת���Ŀ��λ��
	TopLeft.x = TopLeft.x > src.cols ? src.cols : TopLeft.x;
	TopLeft.x = TopLeft.x < 0 ? 0 : TopLeft.x;
	TopLeft.y = TopLeft.y > src.rows ? src.rows : TopLeft.y;
	TopLeft.y = TopLeft.y < 0 ? 0 : TopLeft.y;

	int after_width, after_height;
	if (TopLeft.x + rect.size.width > src.cols) {
		after_width = src.cols - TopLeft.x - 1;
	}
	else {
		after_width = rect.size.width - 1;
	}
	if (TopLeft.y + rect.size.height > src.rows) {
		after_height = src.rows - TopLeft.y - 1;
	}
	else {
		after_height = rect.size.height - 1;
	}
	// ��ö�ά���λ��
	Rect RoiRect = Rect(TopLeft.x, TopLeft.y, after_width, after_height);

	//	dst�Ǳ���ת��ͼƬ roiΪ���ͼƬ maskΪ��ģ
	double angle = rect.angle;
	Mat mask, roi, dst;
	Mat image;
	// �����н�ͼ��������ͼ��

	vector<Point> contour;
	// ��þ��ε��ĸ���
	Point2f points[4];
	rect.points(points);
	for (int i = 0; i < 4; i++)
		contour.push_back(points[i]);

	vector<vector<Point>> contours;
	contours.push_back(contour);
	// ���н�ͼ���л�������
	//drawContours(mask, contours, 0, Scalar(255, 255, 255), -1);
	// ͨ��mask��Ĥ��src���ض�λ�õ����ؿ�����dst�С�
	src.copyTo(dst, mask);
	// ��ת
	Mat M = getRotationMatrix2D(center, angle, 1);
	warpAffine(dst, image, M, src.size());
	// ��ͼ
	roi = image(RoiRect);

	return roi;
}

// �ò������ڼ���Ƿ��ǽǵ㣬����������������� ���ߣ�bubbliiiing
static bool IsQrPoint(vector<Point>& contour, Mat& img)
{
	double area = contourArea(contour);
	// �ǵ㲻����̫С
	if (area < 30)
		return 0;
	RotatedRect rect = minAreaRect(Mat(contour));
	double w = rect.size.width;
	double h = rect.size.height;
	double rate = min(w, h) / max(w, h);
	if (rate > 0.7)
	{
		// ������ת���ͼƬ�����ڰѡ��ء����������ڴ���
		Mat image = transformCorner(img, rect);
		if (IsCorner(image))
		{
			return 1;
		}
	}
	return 0;
}

// ���ڶ�λ����λ��������޵Ľǵ㣬����ڵ�һ������ȡ����Ϊ����ֵ������ڵڶ�������ȡ����Ϊ����ֵ������ڵ���������ȡ���£�����������ȡ���¡�

static Point2f GetRelativePoint(RotatedRect& _vRect, Point2f &_point) 
{

	Point2f arrRectPoint[4];
	vector<Point2f> vRectPoint;

	Point2f OutputPoint;
	int RectX = _vRect.center.x;
	int RectY = _vRect.center.y;
	int CenterX = _point.x;
	int CenterY = _point.y;

	_vRect.points(arrRectPoint);
	for (int i = 0; i < 4; i++)
	{
		vRectPoint.push_back(arrRectPoint[i]);
	}
	std::sort(vRectPoint.begin(), vRectPoint.end(), comparePointsy);

	if (RectX < CenterX && RectY < CenterY) //�ڵ�һ���� 
	{
		OutputPoint = vRectPoint[0];
	}
	else if (RectX > CenterX && RectY < CenterY) //�ڵڶ�����
	{
		OutputPoint = vRectPoint[1];
	}
	else if (RectX > CenterX && RectY > CenterY) //�ڵ������� [3]������
	{
		OutputPoint = vRectPoint[3]; 
	}
	else if (RectX < CenterX && RectY >CenterY) //�ڵ������� 
	{
		OutputPoint = vRectPoint[2];
	}

	return OutputPoint;
}
#endif 