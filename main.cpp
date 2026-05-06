#include "ocr_det.h"
#include "ocr_rec.h"
#include <iostream>
#include <chrono>
#include <dirent.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>


// 检查是否为图片文件
bool isImageFile(const std::string &filename)
{
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::string img_exts[] = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"};
    for (const auto &img_ext : img_exts)
    {
        if (ext == img_ext)
            return true;
    }
    return false;
}

// 获取目录下所有图片文件
std::vector<std::string> getImageFiles(const std::string &dir_path)
{
    std::vector<std::string> image_files;
    DIR *dir = opendir(dir_path.c_str());
    if (!dir)
        return image_files;

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        if (filename != "." && filename != ".." && isImageFile(filename))
        {
            image_files.push_back(dir_path + "/" + filename);
        }
    }
    closedir(dir);
    return image_files;
}

// 创建输出目录（如果不存在）
bool createDirIfNotExists(const std::string &dir_path)
{
    struct stat st;
    if (stat(dir_path.c_str(), &st) != 0)
    {
        return mkdir(dir_path.c_str(), 0755) == 0;
    }
    return S_ISDIR(st.st_mode);
}

// 单张图片处理
bool processSingleImage(
    MNNOCRDet &ocrDet,
    MNNOCRRec &ocrRec,
    cv::Ptr<cv::freetype::FreeType2> &ft,
    const std::string &imgPath,
    const std::string &savePath)
{
    cv::Mat img = cv::imread(imgPath);
    if (img.empty())
    {
        std::cout << "[Error] Read image failed: " << imgPath << std::endl;
        return false;
    }

    // 检测耗时
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::vector<int>>> boxes = ocrDet.Detect(img);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cost = end - start;

    std::cout << "========================================" << std::endl;
    std::cout << "Det cost: " << cost.count() << " ms" << std::endl;
    std::cout << "Det boxes num: " << boxes.size() << std::endl;
    std::cout << "========================================" << std::endl;

    cv::Mat drawimg = img.clone();
    std::chrono::high_resolution_clock::duration total_rec_duration{};
    for (int i = 0; i < boxes.size(); i++)
    {
        auto box = boxes[i];
        cv::Mat crop_img = PaddleOCR::Utility::GetRotateCropImage(img, box);

        auto rec_start = std::chrono::high_resolution_clock::now();
        auto rec_result = ocrRec.RunRec(crop_img);
        auto rec_end = std::chrono::high_resolution_clock::now();
        total_rec_duration += (rec_end - rec_start);

        std::chrono::duration<double, std::milli> rec_cost = rec_end - rec_start;

        std::string text = rec_result.first;
        float score = rec_result.second;

        int width = box[2][0] - box[0][0];
        int height = box[2][1] - box[0][1];
        float ratio = width * 1.0f / height;

        std::cout
            << "Text: " << text << "  Score: " << score
            << "  Rec cost: " << rec_cost.count() << "ms" << "  宽高比: " << ratio << std::endl;

        std::vector<cv::Point> pts;
        for (const auto &p : box)
        {
            int x = p[0];
            int y = p[1];
            pts.emplace_back(x, y);
        }
        cv::polylines(drawimg, pts, true, cv::Scalar(0, 255, 0), 2);

        if (!pts.empty())
        {
            cv::Point text_pos = pts[0];
            text_pos.x = std::max(0, text_pos.x);
            text_pos.y = std::max(20, text_pos.y - 10);

            ft->putText(drawimg, text, text_pos, 40, cv::Scalar(0, 0, 255), -1, cv::LINE_AA, false);
        }
    }
    std::chrono::duration<double, std::milli> total_rec_cost(total_rec_duration);
    std::cout << "总识别耗时（不含绘制/保存）: " << total_rec_cost.count() << " ms" << std::endl;

    cv::imwrite(savePath, drawimg);
    std::cout << "[Save] " << savePath << "\n"
              << std::endl;
    return true;
}


int main()
{
    // 初始化模型
    MNNOCRDet ocrDet;
    MNNOCRRec ocrRec;

    bool det_ret = ocrDet.Init("./PP-OCRv5_mobile_det.mnn");
    bool rec_ret = ocrRec.Init("./PP-OCRv5_mobile_rec.mnn", "ppocr_keys_v5.txt");


    if (!det_ret || !rec_ret)
    {
        std::cout << "[Error] Model init failed!" << std::endl;
        return -1;
    }

    // 初始化中文FreeType
    cv::Ptr<cv::freetype::FreeType2> ft = cv::freetype::createFreeType2();
    ft->loadFontData("SimHei.ttf", 0);

    processSingleImage(ocrDet, ocrRec, ft, "./1.jpeg", "result.jpg");
    return 0;
}