#include "CatModelFactory.hpp"
#include "ONNXModel.hpp"
#include "ONNXBimodel.hpp"
#include "TFModel.hpp"
#include "TorchModel.hpp"

std::unique_ptr<AbstractCatModel> CatModelFactory::produce(std::string model_filename, size_t frame_timeout) {
    if(ONNXModel::accepts(model_filename))
        return std::make_unique<ONNXModel>(model_filename, frame_timeout);
    if(TFModel::accepts(model_filename))
        return std::make_unique<TFModel>(model_filename, frame_timeout);
    if(TorchModel::accepts(model_filename))
        return std::make_unique<TorchModel>(model_filename, frame_timeout);
    return nullptr;
}

std::unique_ptr<AbstractCatModel> CatModelFactory::produce(std::string model_filename, std::string second_model_filename, size_t frame_timeout) {
    if(ONNXBimodel::accepts(model_filename) && ONNXBimodel::accepts(second_model_filename))
        return std::make_unique<ONNXBimodel>(model_filename, second_model_filename, frame_timeout);
    return nullptr;
}
