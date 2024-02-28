/**

The MIT License

Copyright (c) 2017 Rafael Muñoz-Salinas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "fbow.h"
#include "cmd_line_parser.h"
#include "json.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

cv::Mat loadSuperpointFeatures(const std::string& path_to_msg) {
    std::cout << "extracting features ..." << std::endl;
    std::cout << "reading msg: " << path_to_msg << std::endl;
    // Load msg binary file
    std::ifstream ifs(path_to_msg, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "could not open msg: " << path_to_msg << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<uint8_t> msgpack;
    while (true) {
        uint8_t buffer;
        ifs.read(reinterpret_cast<char*>(&buffer), sizeof(uint8_t));
        if (ifs.eof()) {
            break;
        }
        msgpack.push_back(buffer);
    }
    ifs.close();

    // Convert msg binary file to json
    const auto json = nlohmann::json::from_msgpack(msgpack);
    // Load keypoints and descriptors from json into vector
    const auto json_descs = json.at("descriptors");
    std::vector<std::vector<std::vector<float>>> descs_vecs = json_descs.get<std::vector<std::vector<std::vector<float>>>>();
    
    // Convert features from vector to cv::Mat type
    cv::Mat feature = cv::Mat::zeros(descs_vecs[0].size(), 256, CV_32F);
    for (unsigned int i = 0; i < descs_vecs[0].size(); i++) {
        for (unsigned int j = 0; j < 256; ++j) {
            feature.at<float>(i, j) = descs_vecs[0][i][j];
        }
    }
    std::cout << "extracted features: total = " << descs_vecs[0].size() << std::endl;

    return feature;
}

int main(int argc, char** argv) {
    CmdLineParser cml(argc, argv);
    try {
        if (argc < 3 || cml["-h"]) {
            std::cerr << "Usage: VOCABULARY IMAGE" << std::endl;
            std::cerr << std::endl;
            std::cerr << "Loads a vocabulary and am image." << std::endl;
            std::cerr << "Extracts image features and then compute the BoW of the image." << std::endl;
            std::cerr << std::endl;
            return EXIT_FAILURE;
        }
        fbow::Vocabulary vocab;
        vocab.readFromFile(argv[1]);

        std::string desc_name = vocab.getDescName();
        std::cout << "vocabulary descriptor: " << desc_name << std::endl;
        auto features = loadSuperpointFeatures(std::string(argv[2]));
        std::cout << "size: " << features.rows << " " << features.cols << std::endl;

        fbow::BoWVector bow_vec;
        auto t_start = std::chrono::high_resolution_clock::now();
        bow_vec = vocab.transform(features);
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "time: " << (double)(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) << "ms" << std::endl;
        std::cout << std::endl;

        for (const auto& v : bow_vec) {
            std::cout << v.first << "(" << (float)v.second << ")" << " ";
        }
        std::cout << std::endl;
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }

    return EXIT_SUCCESS;
}
