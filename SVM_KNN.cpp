#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

void computeHOG(Mat &image, vector<float> &featureVector);
void loadDataset(vector<Mat> &images, vector<int> &labels, string folder, int label);

int main() {
    // 1️⃣ 데이터 로드 및 전처리
    vector<Mat> images;
    vector<int> labels;

    // 자동차 종류별 폴더에서 데이터셋 로드
    loadDataset(images, labels, "/Users/parksungsu/Desktop/full-stack/mv_car", 0);  // 승용차
    loadDataset(images, labels, "/Users/parksungsu/Desktop/full-stack/mv_truck", 1); // 트럭
    loadDataset(images, labels, "/Users/parksungsu/Desktop/full-stack/mv_bus", 2);  // 버스

    cout << "데이터셋 로드 완료: " << images.size() << "개의 이미지" << endl; // 학습된 개수를 표시

    // 2️⃣ HOG 특징 추출
    vector<float> featureVector;
    Mat trainingData, labelMat;
    for (size_t i = 0; i < images.size(); i++) { // 45개의 이미지
        computeHOG(images[i], featureVector); // images[i]의 특징 벡터를 계산하고 featureVector에 저장
        Mat featureMat(1, featureVector.size(), CV_32FC1, featureVector.data()); 
        // 1 x featureVector.size() 크기의 행렬을 생성
        // featureVector.data()를 이용해 기존 vector<float> 데이터를 직접 사용

        trainingData.push_back(featureMat); // featureMat을 trainingData에 추가
        labelMat.push_back(labels[i]); // labelMat에 현재 이미지의 클래스 값(0 승용차, 1 트럭, 2 버스) 추가
    }

    // 3️⃣ SVM 모델 학습
    Ptr<SVM> svm = SVM::create(); // SVM 모델 생성
    svm->setType(SVM::C_SVC); // SVM 유형 C-Support Vector Classification로 설정(다중 클래스 분류)
    svm->setKernel(SVM::LINEAR); // 선형 커널(직선 혹은 평면으로 나눌 수 있을 때 적합)
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    // TermCriteria::MAX_ITER : 최대 반복 횟수를 기준으로 학습 종료
    // 100 : 최대 반복 횟수(100번 반복 후 학습 종료)
    // 1e-6 학습 중 손실(Loss)이 10의 -6승 이하로 떨어지면 학습 종료

    cout << "SVM 학습 시작..." << endl;
    svm->train(trainingData, ROW_SAMPLE, labelMat); // 차량 데이터 학습
    svm->save("vehicle_svm.xml");  // 학습된 모델 저장
    cout << "SVM 학습 완료!" << endl;

    // 4️⃣ KNN 모델 학습
    Ptr<KNearest> knn = KNearest::create(); // KNN 모델 생성
    cout << "KNN 학습 시작..." << endl;
    knn->train(trainingData, ROW_SAMPLE, labelMat); // 차량 데이터 학습
    cout << "KNN 학습 완료!" << endl;

    // 5️⃣ 테스트 데이터 예측
    Mat testImg = imread("/Users/parksungsu/Desktop/full-stack/test.jpg", IMREAD_GRAYSCALE);
    resize(testImg, testImg, Size(64, 64)); // 이미지 크기 변환

    vector<float> testFeature;
    computeHOG(testImg, testFeature); // testImg의 특징 벡터를 계산하고 testFeature에 저장
    Mat testMat(1, testFeature.size(), CV_32FC1, testFeature.data());
    // 1 x testFeature.size() 크기의 행렬을 생성
    // testFeature.data()를 이용해 기존 vector<float> 데이터를 직접 사용

    float svmResult = svm->predict(testMat); // SVM 분류 결과 출력
    float knnResult = knn->findNearest(testMat, 3, noArray()); // kNN 분류 결과 출력

    // SVM 결과에 대한 라벨 출력
    cout << "🚗 SVM 예측 결과: ";
    switch (static_cast<int>(svmResult)) {
        case 0:
            cout << "승용차" << endl;
            break;
        case 1:
            cout << "트럭" << endl;
            break;
        case 2:
            cout << "버스" << endl;
            break;
        default:
            cout << "알 수 없는 객체" << endl;
            break;
    }

    // KNN 결과에 대한 라벨 출력
    cout << "🚗 KNN 예측 결과: ";
    switch (static_cast<int>(knnResult)) {
        case 0:
            cout << "승용차" << endl;
            break;
        case 1:
            cout << "트럭" << endl;
            break;
        case 2:
            cout << "버스" << endl;
            break;
        default:
            cout << "알 수 없는 객체" << endl;
            break;
    }

    return 0;
}

// HOG 특징 벡터를 추출하는 함수
void computeHOG(Mat &image, vector<float> &featureVector) {
    HOGDescriptor hog(
        Size(64, 64),  // 윈도우 크기
        Size(16, 16),  // 블록 크기
        Size(8, 8),    // 블록 이동 크기
        Size(8, 8),    // 셀 크기
        9              // 히스토그램 빈 개수
    );
    
    vector<Point> locations;
    hog.compute(
        image,          // 입력 영상
        featureVector,  // 출력 HOG 기술자(HOGDescriptor 객체 생성시 자동으로 결정)
        Size(16, 16),   // 윈도우 이동 크기
        Size(0, 0),     // 영상 가장자리 패딩 크기
        locations);     // 계산 시작 위치
}

// 데이터셋을 로드하는 함수
void loadDataset(vector<Mat> &images, vector<int> &labels, string folder, int label) {
    vector<String> filenames;
    glob(folder, filenames); // 폴더 내 모든 이미지 파일을 가져옴
    for (size_t i = 0; i < filenames.size(); i++) {
        Mat img = imread(filenames[i], IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        resize(img, img, Size(64, 64)); // 이미지 크기 변환
        images.push_back(img); // 요소 추가
        labels.push_back(label); // 요소 추가
    }
}