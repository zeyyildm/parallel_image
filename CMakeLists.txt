cmake_minimum_required(VERSION 3.16)
project(parallel_image_pipeline CXX) #projenin adı ve hangi dili kullandığımızı söylüyoruz

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) #derleyici desteklemiyorsa buildi durdur

find_package(OpenCV REQUIRED) #sistemde opencv kütüphanesini ara

add_executable(serial_app #bir çalıştırılabilir exe dosyası üret
    serial/main.cpp #ismini ve kaynak dosyasını yazıyoruz
)

target_link_libraries(serial_app PRIVATE ${OpenCV_LIBS})
#serial_app programını OPENCV kütüphanelerine bağla