#ifndef MNN_OCR_REC_H
#define MNN_OCR_REC_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>
#include <utility>
#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/Tensor.hpp"

class MNNOCRRec
{
public:
    MNNOCRRec();
    ~MNNOCRRec();

    bool Init(const std::string &modelPath, const std::string &dictPath);
    std::pair<std::string, float> RunRec(const cv::Mat &srcImg);

private:
    cv::Mat Preprocess(const cv::Mat &src, int &dstH, int &dstW, std::vector<float> &ratioHw);
    std::pair<std::string, float> PostProcess(MNN::Tensor *output);
    bool LoadDict(const std::string &path);

private:
    std::shared_ptr<MNN::Interpreter> m_interpreter;
    MNN::Session *m_session;

    // 识别模型固定配置
    const int m_inputHeight = 48;
    const int m_maxWidth = 320;

    const float m_meanVals[3] = {0.5 * 255.0f, 0.5 * 255.0f, 0.5 * 255.0f};
    const float m_normVals[3] = {1.0f / (255.0f * 0.5), 1.0f / (255.0f * 0.5), 1.0f / (255.0f * 0.5)};

    std::vector<std::string> m_dict;
};

#endif