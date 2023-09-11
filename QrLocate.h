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

	static vector<RotatedRect> DetQr_filterByNestedContours(Mat& _srcImg, Mat& _binaryImg);   ///< 筛选后的回字

	static Mat DetQr_CropRotateQrimg(Mat &_srcImg, vector<RotatedRect> &_vRect);      ///< 对原图进行旋转并裁剪



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

	// 计算左侧边向量
	Point2f leftVec = points[3] - points[0];
	// 计算右侧边向量
	Point2f rightVec = points[2] - points[1];

	// 计算左侧边的单位法向量
	Point2f leftNormal = Point2f(-leftVec.y, leftVec.x);
	leftNormal /= norm(leftNormal);

	// 计算右侧边的单位法向量
	Point2f rightNormal = Point2f(rightVec.y, -rightVec.x);
	rightNormal /= norm(rightNormal);

	// 根据偏移量，计算四个新的顶点坐标
	Point2f topLeft = points[0] - leftNormal * offset;
	Point2f topRight = points[1] + rightNormal * offset;
	Point2f bottomRight = points[2] + leftNormal * offset;
	Point2f bottomLeft = points[3] - rightNormal * offset;

	// 更新四个点的坐标
	points[0] = topLeft;
	points[1] = topRight;
	points[2] = bottomRight;
	points[3] = bottomLeft;

}

static int calculateDistance(const Point2f& point1, const Point2f& point2)
{
	/* 计算两个点之间的直线距离 */
	double distance = std::sqrt(std::pow(point2.x - point1.x, 2) + std::pow(point2.y - point1.y, 2));
	return static_cast<int>(std::round(distance));
}


// 判断是否灰度图 不是则转成灰度图  作者：uuyymilkyl
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

// 计算内部所有白色部分占全部的比率  作者：bubbliiiing
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

// 用于判断是否属于角上的正方形  作者：bubbliiiing
static bool IsCorner(Mat& image)
{
	// 定义mask
	Mat imgCopy, dstCopy;
	Mat dstGray;
	imgCopy = image.clone();

	// 转化为灰度图像
	cvtColor(image, dstGray, COLOR_BGR2GRAY);
	// 进行二值化

	threshold(dstGray, dstGray, 0, 255, THRESH_BINARY | THRESH_OTSU);
	dstCopy = dstGray.clone();  //备份

	Mat erodeStruct = getStructuringElement(MORPH_RECT, Size(3, 3));
	//morphologyEx(dstCopy, dstCopy, MORPH_OPEN, erodeStruct);
	// 找到轮廓与传递关系
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

			// 最里面的矩形与最外面的矩形的对比
			if (rect.width < imgCopy.cols * 2 / 7)      //2/7是为了防止一些微小的仿射
				continue;
			if (rect.height < imgCopy.rows * 2 / 7)      //2/7是为了防止一些微小的仿射
				continue;

			// 判断其中黑色与白色的部分的比例
			if (Rate(dstGray) > 0.20)
			{
				return true;
			}
		}
	}
	return  false;
}

// 用于得到回字角  作者：
static Mat transformCorner(Mat src, RotatedRect rect)
{
	// 获得旋转中心
	Point center = rect.center;
	// 获得左上角和右下角的角点，而且要保证不超出图片范围，用于抠图
	Point TopLeft = Point(cvRound(center.x), cvRound(center.y)) - Point(rect.size.height / 2, rect.size.width / 2);  //旋转后的目标位置
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
	// 获得二维码的位置
	Rect RoiRect = Rect(TopLeft.x, TopLeft.y, after_width, after_height);

	//	dst是被旋转的图片 roi为输出图片 mask为掩模
	double angle = rect.angle;
	Mat mask, roi, dst;
	Mat image;
	// 建立中介图像辅助处理图像

	vector<Point> contour;
	// 获得矩形的四个点
	Point2f points[4];
	rect.points(points);
	for (int i = 0; i < 4; i++)
		contour.push_back(points[i]);

	vector<vector<Point>> contours;
	contours.push_back(contour);
	// 再中介图像中画出轮廓
	//drawContours(mask, contours, 0, Scalar(255, 255, 255), -1);
	// 通过mask掩膜将src中特定位置的像素拷贝到dst中。
	src.copyTo(dst, mask);
	// 旋转
	Mat M = getRotationMatrix2D(center, angle, 1);
	warpAffine(dst, image, M, src.size());
	// 截图
	roi = image(RoiRect);

	return roi;
}

// 该部分用于检测是否是角点，与下面两个函数配合 作者：bubbliiiing
static bool IsQrPoint(vector<Point>& contour, Mat& img)
{
	double area = contourArea(contour);
	// 角点不可以太小
	if (area < 30)
		return 0;
	RotatedRect rect = minAreaRect(Mat(contour));
	double w = rect.size.width;
	double h = rect.size.height;
	double rate = min(w, h) / max(w, h);
	if (rate > 0.7)
	{
		// 返回旋转后的图片，用于把“回”摆正，便于处理
		Mat image = transformCorner(img, rect);
		if (IsCorner(image))
		{
			return 1;
		}
	}
	return 0;
}

// 用于定位矩形位置相对象限的角点，如果在第一象限则取左上为返回值，如果在第二象限则取右上为返回值，如果在第三象限则取右下，第四象限则取左下。

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

	if (RectX < CenterX && RectY < CenterY) //在第一象限 
	{
		OutputPoint = vRectPoint[0];
	}
	else if (RectX > CenterX && RectY < CenterY) //在第二象限
	{
		OutputPoint = vRectPoint[1];
	}
	else if (RectX > CenterX && RectY > CenterY) //在第三象限 [3]是右下
	{
		OutputPoint = vRectPoint[3]; 
	}
	else if (RectX < CenterX && RectY >CenterY) //在第四象限 
	{
		OutputPoint = vRectPoint[2];
	}

	return OutputPoint;
}
#endif 