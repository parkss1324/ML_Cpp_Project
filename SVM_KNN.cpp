#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

// HOG 특징 벡터를 추출하는 함수
void computeHOG(Mat &image, vector<float> &featureVector) {
    HOGDescriptor hog(
        Size(64, 64),  // 윈도우 크기
        Size(16, 16),  // 블록 크기
        Size(8, 8),    // 블록 스트라이드
        Size(8, 8),    // 셀 크기
        9              // 히스토그램 빈 개수
    );
    vector<Point> locations;
    hog.compute(image, featureVector, Size(8, 8), Size(0, 0), locations);
}

// 데이터셋을 로드하는 함수
void loadDataset(vector<Mat> &images, vector<int> &labels, string folder, int label) {
    vector<String> filenames;
    glob(folder, filenames); // 폴더 내 모든 이미지 파일을 가져옴
    for (size_t i = 0; i < filenames.size(); i++) {
        Mat img = imread(filenames[i], IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        resize(img, img, Size(64, 64));
        images.push_back(img);
        labels.push_back(label);
    }
}

int main() {
    // 1️⃣ 데이터 로드 및 전처리
    vector<Mat> images;
    vector<int> labels;

    // 예제: 자동차 종류별 폴더
    loadDataset(images, labels, "/Users/parksungsu/Desktop/full-stack/mv_car", 0);  // 승용차
    loadDataset(images, labels, "/Users/parksungsu/Desktop/full-stack/mv_truck", 1); // 트럭
    loadDataset(images, labels, "/Users/parksungsu/Desktop/full-stack/mv_bus", 2);  // 버스

    cout << "데이터셋 로드 완료: " << images.size() << "개의 이미지" << endl;

    // 2️⃣ HOG 특징 추출
    vector<float> featureVector;
    Mat trainingData, labelMat;
    for (size_t i = 0; i < images.size(); i++) {
        computeHOG(images[i], featureVector);
        Mat featureMat(1, featureVector.size(), CV_32FC1, featureVector.data());
        trainingData.push_back(featureMat);
        labelMat.push_back(labels[i]);
    }

    // 3️⃣ SVM 모델 학습
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    cout << "SVM 학습 시작..." << endl;
    svm->train(trainingData, ROW_SAMPLE, labelMat);
    svm->save("vehicle_svm.xml");  // 학습된 모델 저장
    cout << "SVM 학습 완료!" << endl;

    // 4️⃣ KNN 모델 학습
    Ptr<KNearest> knn = KNearest::create();
    knn->train(trainingData, ROW_SAMPLE, labelMat);
    cout << "KNN 학습 완료!" << endl;

    // 5️⃣ 테스트 데이터 예측
    Mat testImg = imread("/Users/parksungsu/Desktop/full-stack/test.jpg", IMREAD_GRAYSCALE);
    resize(testImg, testImg, Size(64, 64));

    vector<float> testFeature;
    computeHOG(testImg, testFeature);
    Mat testMat(1, testFeature.size(), CV_32FC1, testFeature.data());

    float svmResult = svm->predict(testMat);
    float knnResult = knn->findNearest(testMat, 3, noArray());

    cout << "🚗 SVM 예측 결과: " << svmResult << endl;
    cout << "🚗 KNN 예측 결과: " << knnResult << endl;

    return 0;
}
