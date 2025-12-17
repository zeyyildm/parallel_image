//adım 1: görseli okuma
#include <opencv2/opencv.hpp>
#include <vector> //matrise çevirmek için

cv::Mat toGrayFloat01(const cv::Mat& bgr){ //& kopyalamıyoruz orijinal görüntüde çalışıyoruz
    CV_Assert(bgr.type() == CV_8UC3); //yanlışlıkla farklı tip gelirse diye
    //her kanal 8 bit C3 = 3 kanal(RGB)
    //yani bu satır ile gerçekten renkli bi görüntü geldi mi ona bakıyorum
    cv::Mat gray(bgr.rows, bgr.cols, CV_32F); //gri görüntü tek kanal ve her piksel 32-bit float
    for(int r = 0; r < bgr.rows; r++){ //görüntüyü satır satır gez
        const cv::Vec3b* srcRow = bgr.ptr<cv::Vec3b>(r); //ilk pikselin adresini alıyoruz
        float* dstRow = gray.ptr<float>(r); //Artık dstRow[c] dediğimde r satırı, c.sütunundaki gri değer demek

        for(int c = 0; c < bgr.cols; c++){ //iç döngü ile sütunda geziyoruz
            float B = srcRow[c][0]; //rgb değerleri char gelir
            float G = srcRow[c][1];
            float R = srcRow[c][2];

            float y = (0.114f * B + 0.587f * G + 0.299f * R) / 255.0f; //biz griye dönüştürdük
            //255 ile 0-1 arasına getirdik
            dstRow[c] = y; //gri değeri gray görüntüsüne gidip attım

        }
    }
    return gray; //gri görüntü döndü
}
int main(){
    cv::Mat img = cv::imread("araba.jpg", cv::IMREAD_COLOR); //görseli oku
    if(img.empty()){ //görsel okunabildi mi
        //burda okunan görsel RGB, 8 BİT, 3 KANALLI
        std::cerr << "ımage yuklenemedııııı" << "\n";
        return (1);
    }
    cv::Mat gray01 = toGrayFloat01(img);
    //burdaki gray01 TEK KANALLI DEĞERLER -1 ARASI 32 BİT
    //ama bu şekilde görseli kaydedemeyiz jpg 8 bit ister
    //kontrol için tekrar 8 bite dönüştürüp kaydedeceğiz ki kontrol sağlayalım griye dönmüş mü diye
    cv::Mat gray8u;
    gray01.convertTo(gray8u, CV_8U, 255.0);
    cv::imwrite("serial_gray_copy.png", gray8u); //görseli düzgün okkuyabildik mi diye bir kopyasını kaydediyorum
    std::cout << "serial gray copy gorsel olusturuldu." << "\n";
    return (0);
} 