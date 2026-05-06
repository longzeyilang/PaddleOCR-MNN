cmake_minimum_required(VERSION 3.12)
project(MNNInferenceDemo)

set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

# 2. 设置OpenCV头文件路径
include_directories(/home/gzy/3rdparty/include)

# 3. 设置OpenCV库文件路径
link_directories(/home/gzy/3rdparty/lib)


# 添加可执行文件
add_executable(MNNInferenceDemo 
                clipper.cpp
                utility.cpp
                postprocess_op.cpp
                ocr_det.cpp
                main.cpp
                ocr_rec.cpp
            )

# 链接库
target_link_libraries(MNNInferenceDemo 
                opencv_world
                dlib
                MNN_Express
                MNN
                pthread
                )