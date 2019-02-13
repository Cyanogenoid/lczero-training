#include "chess/board.h"
#include "chess/position.h"
#include "neural/encoder.h"
#include "utils/bititer.h"
//#include "proto/chunk.pb.h"

#include <torch/extension.h>


torch::Tensor input_planes_to_tensor(lczero::InputPlanes planes) {
    torch::Tensor tensor = torch::zeros({112, 8, 8});

    float* data = tensor.data<float>();
    int plane_idx = 0;
    for (const auto& plane : planes) {
        for (auto bit : lczero::IterateBits(plane.mask)) {
            data[plane_idx * 112 + bit] = plane.value;
        }
        plane_idx++;
    }
    return tensor;
}


torch::Tensor parse_chunk() {
    // Create board at starting position.
    lczero::ChessBoard board;
    board.SetFromFen(lczero::ChessBoard::kStartposFen);
    // Keep history of positions.
    lczero::PositionHistory history;
    history.Reset(board, 0, 0);

    lczero::InputPlanes planes = lczero::EncodePositionForNN(history, 8, lczero::FillEmptyHistory::NO);
    return input_planes_to_tensor(planes);
}

torch::Tensor load_position() {
    return parse_chunk();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_position", &load_position, "Load a position from a chunk");
}
