#include "cmd_line_parser.h"
#include "vocabulary_creator.h"
#include "json.hpp"
#include "dir_reader.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

std::vector<cv::Mat> readSuperpointFeaturesFromFile(const std::vector<std::string>& path_to_msgs, std::string& desc_name) {
    std::vector<cv::Mat> features;
    desc_name = "Superpoint";

    std::cout << "extracting features ..." << std::endl;
    for (const auto& path_to_msg : path_to_msgs) {
        std::cout << "reading msg: " << path_to_msg << std::endl;
        // Load msg binary file
        std::ifstream ifs(path_to_msg, std::ios::in | std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "could not open msg: " << path_to_msg << std::endl;
            continue;
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
        features.push_back(feature);
    }

    return features;
}

int main(int argc, char** argv) {
    try {
        CmdLineParser cml(argc, argv);
        if (cml["-h"] || argc < 3) {
            std::cerr << "Usage: FEATURE_INPUT_DIR OUTPUT_VOCABULARY [-k K] [-l L] [-t NUM_THREADS] [--max-iters NUM_ITER] [-v]" << std::endl;
            std::cerr << std::endl;
            std::cerr << "Second step is creating the vocabulary of K^L from the set of features." << std::endl;
            std::cerr << "By default, we employ a random selection center without running a single iteration of the k means." << std::endl;
            std::cerr << "As indicated by the authors of the FLANN library in their paper, the result is not very different from using k-means, but speed is much better." << std::endl;
            std::cerr << std::endl;
            return EXIT_FAILURE;
        }

        std::string desc_name;
        auto msgs = DirReader::read(argv[1]);
        auto features = readSuperpointFeaturesFromFile(msgs, desc_name);

        std::cout << "descriptor name: " << desc_name << std::endl;
        std::cout << "num of features: " << features.size() << std::endl;
        std::cout << "feature shape: " << features[0].rows << " " << features[0].cols << std::endl;

        fbow::VocabularyCreator::Params params;
        params.k = stoi(cml("-k", "10"));
        params.L = stoi(cml("-l", "6"));
        params.nthreads = stoi(cml("-t", "4"));
        params.maxIters = std::stoi(cml("--max-iters", "0"));
        params.verbose = cml["-v"];

        srand(0);
        fbow::VocabularyCreator vocab_creator;
        fbow::Vocabulary vocab;

        std::cout << "creating a " << params.k << "^" << params.L << " vocabulary ..." << std::endl;

        auto t_start = std::chrono::high_resolution_clock::now();
        vocab_creator.create(vocab, features, desc_name, params);
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "time: " << (double)(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) << "ms" << std::endl;
        std::cout << "number of blocks: " << vocab.size() << std::endl;
        std::cout << "saving the vocabulary: " << argv[2] << std::endl;

        vocab.saveToFile(argv[2]);
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }

    return EXIT_SUCCESS;
}