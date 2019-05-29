

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;
int thr = 20;
int dlt = 3;

bool okr = false, okl = false;

Vec3f seed = Vec3b(0, 0, 0);
Mat frame_HSV;

Point mleft, mright;

static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H - 1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H + 1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S - 1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S + 1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V - 1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V + 1);
    setTrackbarPos("High V", window_detection_name, high_V);
}
static void on_dilate_trackbar(int, void *)
{
    dlt = max(0, dlt);
    setTrackbarPos("Dilate", window_detection_name, dlt);
}
int sizeK = 15;
int cont = 0;
void CallBackFunc(int event, int x, int y, int flags, void *userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        okr = true;
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        mleft = Point(x, y);
        seed = frame_HSV.at<Vec3b>(y, x);
        Vec3f mean = Vec3f(0, 0, 0);
        cont = 0;
        for (size_t i = y - sizeK; i < y + sizeK; i++)
        {
            for (size_t j = x - sizeK; j < x + sizeK; j++)
            {
                if (i > 0 && j > 0 && i < frame_HSV.rows && j < frame_HSV.cols)
                {
                    Vec3b aux = frame_HSV.at<Vec3b>(i, j);
                    mean[0] += aux[0];
                    mean[1] += aux[1];
                    mean[2] += aux[2];
                    cont++;
                }
            }
        }
        mean[0] /= cont;
        mean[1] /= cont;
        mean[2] /= cont;

        seed = mean;

        cout << "Seed: " << seed << endl;

        low_H = seed[0] - thr;
        high_H = seed[0] + thr;
        low_S = seed[1] - thr;
        high_S = seed[1] + thr;
        low_V = seed[2] - thr;
        high_V = seed[2] + thr;
        on_low_H_thresh_trackbar(0, NULL);
        on_high_H_thresh_trackbar(0, NULL);
        on_low_S_thresh_trackbar(0, NULL);
        on_high_S_thresh_trackbar(0, NULL);
        on_low_V_thresh_trackbar(0, NULL);
        on_high_V_thresh_trackbar(0, NULL);
    }
    if (event == EVENT_RBUTTONDOWN)
    {

        okl = true;
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        seed = frame_HSV.at<Vec3b>(y, x);
        Vec3f mean = Vec3f(0, 0, 0);
        cont = 0;
        mright = Point(x, y);
        for (size_t i = y - sizeK; i < y + sizeK; i++)
        {
            for (size_t j = x - sizeK; j < x + sizeK; j++)
            {
                if (i > 0 && j > 0 && i < frame_HSV.rows && j < frame_HSV.cols)
                {
                    Vec3b aux = frame_HSV.at<Vec3b>(i, j);
                    mean[0] += aux[0];
                    mean[1] += aux[1];
                    mean[2] += aux[2];
                    cont++;
                }
            }
        }
        mean[0] /= cont;
        mean[1] /= cont;
        mean[2] /= cont;

        seed = (seed + mean) / 2;
        cout << "Seed: " << seed << endl;

        low_H = seed[0] - thr;
        high_H = seed[0] + thr;
        low_S = seed[1] - thr;
        high_S = seed[1] + thr;
        low_V = seed[2] - thr;
        high_V = seed[2] + thr;
        on_low_H_thresh_trackbar(0, NULL);
        on_high_H_thresh_trackbar(0, NULL);
        on_low_S_thresh_trackbar(0, NULL);
        on_high_S_thresh_trackbar(0, NULL);
        on_low_V_thresh_trackbar(0, NULL);
        on_high_V_thresh_trackbar(0, NULL);
        on_dilate_trackbar(0, NULL);
    }
    if (event = -99)
    {
        low_H = seed[0] - thr;
        high_H = seed[0] + thr;
        low_S = seed[1] - thr;
        high_S = seed[1] + thr;
        low_V = seed[2] - thr;
        high_V = seed[2] + thr;
    }
}

static void on_thr_trackbar(int, void *)
{
    thr = max(0, thr);
    setTrackbarPos("Thr", window_detection_name, thr);
    CallBackFunc(-99, 1, 1, 1, NULL);
}

bool distance(Vec3b a, Vec3b b, int value)
{
    float d = sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2));

    if (d < value)
        return true;
    else
        return false;
}

void inRange2(Mat &input, Mat &out, Vec3b referencia, int value, Point l, Point r)
{
    out = Mat::zeros(input.size(), CV_8UC1);
    int size = 150;

    for (size_t i = l.y - size; i < l.y + size; i++)
    {
        Vec3b *ptr = input.ptr<Vec3b>(i);
        uchar *outPtr = out.ptr<uchar>(i);

        for (size_t j = l.x - size; j < l.x + size; j++)
        {
            if (j > 0 && i > 0 && i < input.rows && j < input.cols)
            {
                if (distance(referencia, ptr[j], value))
                {
                    outPtr[j] = 255;
                }
                else
                {
                    outPtr[j] = 0;
                }
            }
        }
    }

    for (size_t i = r.y - size; i < r.y + size; i++)
    {
        Vec3b *ptr = input.ptr<Vec3b>(i);
        uchar *outPtr = out.ptr<uchar>(i);

        for (size_t j = r.x - size; j < r.x + size; j++)
        {
            if (j > 0 && i > 0 && i < input.rows && j < input.cols)
            {
                if (distance(referencia, ptr[j], value))
                {
                    outPtr[j] = 255;
                }
                else
                {
                    outPtr[j] = 0;
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    VideoCapture cap(argc > 1 ? argv[1] : 0);
    namedWindow(window_capture_name, 2);
    namedWindow(window_detection_name, 2);
    namedWindow("HSV", 2);
    cout << " Selecione seus olhos com o mouse " << endl
         << " olho esquerdo ->> click mouse esquerdo" << endl
         << " olho direito ->> click do mouse direito" << endl
         << " qual tecla do teclado para continuar" << endl
         << " tecle - \"P\" - para pausar"<<endl;

    setMouseCallback("HSV", CallBackFunc, NULL);

    int b_circles = 30, pr1 = 100, pr2 = 30;
    createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
    createTrackbar("Thr", window_detection_name, &thr, max_value, on_thr_trackbar);
    createTrackbar("Dilate", window_detection_name, &dlt, max_value, on_dilate_trackbar);
    createTrackbar("Between Circles", window_detection_name, &b_circles, max_value);
    createTrackbar("pr1", window_detection_name, &pr1, max_value);
    createTrackbar("pr2", window_detection_name, &pr2, max_value);

    Mat frame, frame_threshold;

    int fps = cap.get(CAP_PROP_FPS);
    cap >> frame;
    cvtColor(frame, frame_HSV, CV_BGR2HSV_FULL);

    VideoWriter video("cladio_eye.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, frame.size());
    imshow(window_capture_name, frame);
    imshow("HSV", frame_HSV);

    waitKey();
    video.set(VIDEOWRITER_PROP_QUALITY, 5);

    vector<vector<Point>> contornos;

    while (true)
    {

        if (okl && okr)
        {
            cap >> frame;
            if (frame.empty())
            {
                break;
            }
            cvtColor(frame, frame_HSV, CV_BGR2HSV_FULL);
            inRange2(frame_HSV, frame_threshold, seed, thr, mleft, mright);

            Mat element = getStructuringElement(MORPH_RECT,
                                                Size(2 * dlt + 1, 2 * dlt + 1),
                                                Point(-1, -1));
            dilate(frame_threshold, frame_threshold, element);
            dilate(frame_threshold, frame_threshold, element);
            dilate(frame_threshold, frame_threshold, element);
           

            findContours(frame_threshold, contornos, RETR_EXTERNAL, CHAIN_APPROX_NONE);

            int nc = contornos.size() > 2 ? 2 : contornos.size();
            for (size_t i = 0; i < nc; i++)
            {
                Point center;
                Point max = Point(0, 0), min = Point(9999, 9999);
                for (size_t j = 0; j < contornos.at(i).size(); j++)
                {
                    center.x += contornos.at(i).at(j).x;
                    center.y += contornos.at(i).at(j).y;
                    if (max.x < contornos.at(i).at(j).x)
                        max.x = contornos.at(i).at(j).x;
                    if (max.y < contornos.at(i).at(j).y)
                        max.y = contornos.at(i).at(j).y;
                    if (min.x > contornos.at(i).at(j).x)
                        min.x = contornos.at(i).at(j).x;
                    if (min.x > contornos.at(i).at(j).y)
                        min.y = contornos.at(i).at(j).y;
                }
                center.x /= contornos.at(i).size() - 1;
                center.y /= contornos.at(i).size() - 1;
                int raio1 = sqrt(pow(max.x - center.x, 2) + pow(max.y - center.y, 2));
                int raio2 = sqrt(pow(min.x - center.x, 2) + pow(min.y - center.y, 2));
                circle(frame, center, 2, Scalar(0, 0, 255), 4);
                circle(frame, center, abs(raio1 + raio2)/6, Scalar(255, 0, 255), 4);
            }
            cout << "Size: " << contornos.size() << endl;
            video.write(frame);

            imshow("HSV", frame_HSV);
            imshow(window_capture_name, frame);
            imshow(window_detection_name, frame_threshold);

            char key = (char)waitKey(5);
            if (key == 'q' || key == 27)
            {
                break;
            }
            if (key == 'p')
            {
                key = waitKey();
            }
        }
    }
    return 0;
}