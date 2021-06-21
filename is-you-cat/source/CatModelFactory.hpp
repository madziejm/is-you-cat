#pragma once

#include <memory>
#include <string>
#include "AbstractCatModel.hpp"

class CatModelFactory {
public:
    static std::unique_ptr<AbstractCatModel> produce(std::string model_filename, size_t frame_timeout);
    static std::unique_ptr<AbstractCatModel> produce(std::string model_filename, std::string second_model_filename, size_t frame_timeout);
};
