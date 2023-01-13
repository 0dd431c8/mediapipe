#include "mediagraph.h"
#include "mediagraph_impl.h"

namespace mediagraph {

Detector* Detector::Create(const char* graph_config, const Output* outputs, uint8_t num_outputs) {
    DetectorImpl* mediagraph = new DetectorImpl();

    absl::Status status = mediagraph->Init(graph_config, outputs, num_outputs);
    if (status.ok()) {
        return mediagraph;
    } else {
        LOG(INFO) << "Error initializing graph " << status.ToString();
        delete mediagraph;
        return nullptr;
    }
}

Detector::~Detector() {
    DetectorImpl* det = static_cast<DetectorImpl*>(this);

    if (det == nullptr) return;

    det->Dispose();
}

Landmark* Detector::Process(uint8_t* data, int width, int height, uint8_t* num_features) {
    return dynamic_cast<DetectorImpl*>(this)->Process(data, width, height, num_features);
}
}
