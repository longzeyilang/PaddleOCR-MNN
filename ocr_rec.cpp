#include "ocr_rec.h"
#include <fstream>
#include <algorithm>
#include <iostream>

MNNOCRRec::MNNOCRRec()
    : m_interpreter(nullptr),
      m_session(nullptr)
{
}

MNNOCRRec::~MNNOCRRec()
{
    if (m_interpreter && m_session)
    {
        m_interpreter->releaseSession(m_session);
        m_session = nullptr;
    }
    m_interpreter.reset();
}

bool MNNOCRRec::LoadDict(const std::string &path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
    {
        return false;
    }

    m_dict.clear();
    m_dict.push_back("blank");
    std::string line;
    while (std::getline(ifs, line))
    {
        if (!line.empty())
        {
            m_dict.push_back(line);
        }
    }
    ifs.close();
    return true;
}

bool MNNOCRRec::Init(const std::string &modelPath, const std::string &dictPath)
{
    if (!LoadDict(dictPath))
    {
        std::cerr << "字典加载失败" << std::endl;
        return false;
    }

    m_interpreter.reset(MNN::Interpreter::createFromFile(modelPath.c_str()));
    if (!m_interpreter)
    {
        return false;
    }

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

cv::Mat MNNOCRRec::Preprocess(const cv::Mat &src, int &dstH, int &dstW, std::vector<float> &ratioHw)
{
    int h = src.rows;
    int w = src.cols;

    float wh_ratio = (float)w / (float)h;
    int imgW = static_cast<int>(m_inputHeight * wh_ratio); // m_inputHeight = 48

    cv::Mat resize_img;
    cv::resize(src, resize_img, cv::Size(imgW, m_inputHeight), 0, 0, cv::INTER_LINEAR);

    // 计算需要填充到的目标宽度（向上取最近 32 的倍数）
    int target_width = ((imgW + 15) / 16) * 16; // 向上取整为32倍数
    int pad_w = target_width - imgW;            // 需要填充的宽度

    // 右侧填充（左0、上0、下0、右pad_w），填充颜色为 (127,127,127) 中性灰
    cv::copyMakeBorder(
        resize_img,
        resize_img,
        0,     // 顶部不填充
        0,     // 底部不填充
        0,     // 左侧不填充
        pad_w, // 右侧填充
        cv::BORDER_CONSTANT,
        cv::Scalar(127, 127, 127));

    dstH = m_inputHeight; // 48
    dstW = target_width;  // 32的倍数

    // 缩放比例（原图 → 预处理图）
    ratioHw.clear();
    ratioHw.push_back(static_cast<float>(m_inputHeight) / h); // height比例
    ratioHw.push_back(static_cast<float>(imgW) / w);          // width比例

    return resize_img;
}

std::pair<std::string, float> MNNOCRRec::RunRec(const cv::Mat &srcImg)
{
    if (!m_interpreter || !m_session || srcImg.empty())
    {
        return {"", 0.0f};
    }

    int dstH = 0;
    int dstW = 0;
    std::vector<float> ratioHw;
    cv::Mat resizedImage = Preprocess(srcImg, dstH, dstW, ratioHw);

    MNN::Tensor *inputTensor = m_interpreter->getSessionInput(m_session, nullptr);

    m_interpreter->resizeTensor(inputTensor, {1, 3, dstH, dstW});
    m_interpreter->resizeSession(m_session);

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

    return PostProcess(&outputHost);
}

std::pair<std::string, float> MNNOCRRec::PostProcess(MNN::Tensor *output)
{
    // 你的输出 shape: [1, 60, 18385, 1]
    int seq_len = output->channel();  // 正确：序列长度
    int dict_size = output->height(); // 正确：字典大小
    float *data = output->host<float>();

    std::string text;
    float total_score = 1.0f;
    int last_idx = -1;

    // 遍历每个时间步
    for (int t = 0; t < seq_len; t++)
    {
        // 定位到第 t 个时间步的概率向量
        float *step_prob = data + t * dict_size;

        // 查找最大概率索引
        int max_idx = 0;
        float max_prob = step_prob[0];
        for (int i = 1; i < dict_size; i++)
        {
            if (step_prob[i] > max_prob)
            {
                max_prob = step_prob[i];
                max_idx = i;
            }
        }

        // CTC 解码：去重 + 跳过 blank (0)
        if (max_idx != 0 && max_idx != last_idx)
        {
            if (max_idx < (int)m_dict.size())
            {
                text += m_dict[max_idx];
                total_score *= max_prob;
            }
        }
        last_idx = max_idx;
    }

    return {text, total_score};
}