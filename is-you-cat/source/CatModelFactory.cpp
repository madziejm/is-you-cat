#include "CatModelFactory.hpp"
#include "ONNXModel.hpp"
#include "TFModel.hpp"
#include "TorchModel.hpp"

std::unique_ptr<CatModelInterface> CatModelFactory::produce(std::string model_filename) {
    if(ONNXModel::accepts(model_filename))
        return std::make_unique<ONNXModel>(model_filename);
    if(TFModel::accepts(model_filename))
        return std::make_unique<TFModel>(model_filename);
    if(TorchModel::accepts(model_filename))
        return std::make_unique<TorchModel>(model_filename);
    return nullptr;
}
