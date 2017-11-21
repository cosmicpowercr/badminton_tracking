#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp> 
#include <iostream>

void fillHole(const cv::Mat src, cv::Mat&dst)
{
	cv::Size m_Size = src.size();
	cv::Mat temp = cv::Mat::zeros(m_Size.height + 2, m_Size.width + 2, src.type());
	src.copyTo(temp(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)));

	cv::floodFill(temp, cv::Point(0, 0), cv::Scalar(255));
	
	cv::Mat cuting;
	temp(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)).copyTo(cuting);

	dst = src | (~cuting);
}

int main()
{
	cv::Mat image= cv::imread("C:/Users/admin/Desktop/3.jpg");

	cv::namedWindow("in Image");
	cv::imshow("in Image", image);

	cv::cvtColor(image, image, CV_BGR2GRAY);
	cv::threshold(image, image, 100, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	
	std::vector<std::vector<cv::Point>> contours;

	cv::findContours(image, contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	cv::Mat result(image.size(), CV_8U, cv::Scalar(255));
	cv::drawContours(result, contours,-1, cv::Scalar(0), 2);
	cv::namedWindow("resultImage");
	cv::imshow("resultImage", result);

	cv::bitwise_xor(result, cv::Scalar(255), result);
	
	cv::Mat dst=result.clone();
	fillHole(result, dst);

	cv::namedWindow("dst Image");
	cv::imshow("dst Image", dst);
	
	cv::waitKey(0);
	return 0;
}