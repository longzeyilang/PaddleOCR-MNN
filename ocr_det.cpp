#include "ocr_det.h"
#include <algorithm>
#include <cmath>

MNNOCRDet::MNNOCRDet()
    : m_interpreter(nullptr),
      m_session(nullptr)
{
}

MNNOCRDet::~MNNOCRDet()
{
    if (m_interpreter && m_session)
    {
        m_interpreter->releaseSession(m_session);
        m_session = nullptr;
    }
    if (m_interpreter)
    {
        m_interpreter.reset();
    }
}

bool MNNOCRDet::Init(const std::string &mnnModelPath)
{
    m_interpreter.reset(MNN::Interpreter::createFromFile(mnnModelPath.c_str()));

    MNN::ScheduleConfig config;
    config.numThread = 4;
    config.type = MNN_FORWARD_CPU;

    MNN::BackendConfig bConfig;
    bConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Normal;
    bConfig.power = MNN::BackendConfig::Power_High;
    bConfig.memory = MNN::BackendConfig::Memory_High;
    config.backendConfig = &bConfig;

    m_session = m_interpreter->createSession(config);
    return m_session != nullptr;
}

cv::Mat MNNOCRDet::Preprocess(const cv::Mat &src,
                              int &dstHeight,
                              int &dstWidth,
                              float &ratioH,
                              float &ratioW)
{
    int w = src.cols;
    int h = src.rows;

    float ratio = 1.f;
    int max_wh = std::max(w, h);
    if (max_wh > m_inputSize)
    {
        if (h > w)
        {
            ratio = static_cast<float>(m_inputSize) / static_cast<float>(h);
        }
        else
        {
            ratio = static_cast<float>(m_inputSize) / static_cast<float>(w);
        }
    }

    int resize_h = static_cast<int>(h * ratio);
    int resize_w = static_cast<int>(w * ratio);

    if (resize_h % 32 == 0)
    {
        resize_h = resize_h;
    }
    else if (static_cast<float>(resize_h) / 32 < 1 + 1e-5)
    {
        resize_h = 32;
    }
    else
    {
        resize_h = (resize_h / 32 - 1) * 32;
    }

    if (resize_w % 32 == 0)
    {
        resize_w = resize_w;
    }
    else if (static_cast<float>(resize_w) / 32 < 1 + 1e-5)
    {
        resize_w = 32;
    }
    else
    {
        resize_w = (resize_w / 32 - 1) * 32;
    }

    cv::Mat resize_img;
    cv::resize(src, resize_img, cv::Size(resize_w, resize_h));

    dstHeight = resize_h;
    dstWidth = resize_w;
    ratioH = static_cast<float>(resize_h) / h;
    ratioW = static_cast<float>(resize_w) / w;

    return resize_img;
}

std::vector<std::vector<std::vector<int>>> MNNOCRDet::Detect(const cv::Mat &srcImage)
{
    std::vector<std::vector<std::vector<int>>> resultBoxes;
    if (!m_interpreter || !m_session)
    {
        return resultBoxes;
    }

    int srcH = srcImage.rows;
    int srcW = srcImage.cols;
    int dstH = 0;
    int dstW = 0;
    float ratioH = 0.0f;
    float ratioW = 0.0f;

    cv::Mat resizedImage = Preprocess(srcImage, dstH, dstW, ratioH, ratioW);
    MNN::Tensor *inputTensor = m_interpreter->getSessionInput(m_session, nullptr);
    m_interpreter->resizeTensor(inputTensor, {1, 3, dstH, dstW});
    m_interpreter->resizeSession(m_session);
    std::vector<int> shape = inputTensor->shape();

    std::shared_ptr<MNN::Tensor> inputTensorHost(
        new MNN::Tensor(inputTensor, MNN::Tensor::CAFFE));

    std::vector<float> mean = {m_meanVals[0], m_meanVals[1], m_meanVals[2]};
    std::vector<float> var = {m_normVals[0], m_normVals[1], m_normVals[2]};

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
        MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean.data(), 3, var.data(), 3));

    pretreat->convert(resizedImage.data, resizedImage.cols, resizedImage.rows,
                      resizedImage.step[0], inputTensorHost.get());

    inputTensor->copyFromHostTensor(inputTensorHost.get());
    m_interpreter->runSession(m_session);

    MNN::Tensor *outputTensor = m_interpreter->getSessionOutput(m_session, nullptr);
    MNN::Tensor outputHost(outputTensor, MNN::Tensor::CAFFE);
    outputTensor->copyToHostTensor(&outputHost);

    resultBoxes = PostProcess(srcImage, &outputHost, ratioH, ratioW, srcH, srcW);
    return resultBoxes;
}

std::vector<std::vector<std::vector<int>>> MNNOCRDet::PostProcess(const cv::Mat src, MNN::Tensor *output,
                                                                  float ratioH,
                                                                  float ratioW,
                                                                  int srcHeight,
                                                                  int srcWidth)
{
    float *outputData = output->host<float>();
    int outputH = output->height();
    int outputW = output->width();

    cv::Mat pred_map(outputH, outputW, CV_32FC1, outputData);
    cv::Mat cbuf_map;
    pred_map.convertTo(cbuf_map, CV_8UC1, 255.0);

    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, m_thresh * 255, 255, cv::THRESH_BINARY);

    if (use_dilation)
    {
        cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, bit_map, dila_ele);
    }

    std::vector<std::vector<std::vector<int>>> boxes = post_processor_.BoxesFromBitmap(
        pred_map, bit_map, m_boxThresh, unclip_ratio, det_db_score_mode);

    boxes = post_processor_.FilterTagDetRes(boxes, ratioH, ratioW, src);

    if (!boxes.empty())
    {
        // 针对box进行排序
        std::sort(boxes.begin(), boxes.end(), [](const std::vector<std::vector<int>> &box1, const std::vector<std::vector<int>> &box2)
                  {
            // OCR检测框一般是4个点，第一个点是左上角 (x, y)
            int y1 = box1[0][1];
            int y2 = box2[0][1];
            int x1 = box1[0][0];
            int x2 = box2[0][0];

            // 定义y值“几乎相同”的阈值，可根据需求调整（默认20像素）
            const int Y_THRESHOLD = 20;

            // 如果y相差不大 → 按x排序
            if (std::abs(y1 - y2) < Y_THRESHOLD) {
                return x1 < x2;
            }
            // y差距大 → 按y排序
            return y1 < y2; });
    }
    return boxes;
}