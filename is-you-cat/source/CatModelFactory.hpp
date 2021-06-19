#pragma once

#include <memory>
#include <string>
#include "CatModelInterface.hpp"

class CatModelFactory {
public:
    static std::unique_ptr<CatModelInterface> produce(std::string model_filename);
};
