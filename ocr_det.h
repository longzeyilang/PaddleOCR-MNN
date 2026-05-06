#ifndef MNN_OCR_DET_H
#define MNN_OCR_DET_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>
#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "postprocess_op.h"

class MNNOCRDet
{
public:
    MNNOCRDet();
    ~MNNOCRDet();

    bool Init(const std::string &mnnModelPath);
    std::vector<std::vector<std::vector<int>>> Detect(const cv::Mat &srcImage);

private:
    cv::Mat Preprocess(const cv::Mat &src,
                       int &dstHeight,
                       int &dstWidth,
                       float &ratioH,
                       float &ratioW);

    std::vector<std::vector<std::vector<int>>> PostProcess(const cv::Mat src, MNN::Tensor *output,
                                                           float ratioH,
                                                           float ratioW,
                                                           int srcHeight,
                                                           int srcWidth);

private:
    std::shared_ptr<MNN::Interpreter> m_interpreter;
    MNN::Session *m_session;

    const int m_inputSize = 960;   // 960 
    const float m_meanVals[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    const float m_normVals[3] = {1.0f / (255.0f * 0.229), 1.0f / (255.0f * 0.224), 1.0f / (255.0f * 0.225)};

    // 参数
    const float m_thresh = 0.3f;
    const float m_boxThresh = 0.6f;
    const float unclip_ratio = 1.5;
    bool use_dilation = true;
    std::string det_db_score_mode = "slow";

    // post-process
    PaddleOCR::DBPostProcessor post_processor_;
};

#endif