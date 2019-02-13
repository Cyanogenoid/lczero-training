#include "chess/board.h"
#include "chess/position.h"
#include "neural/encoder.h"
//#include "proto/chunk.pb.h"

#include <torch/extension.h>


at::Tensor parse_chunk() {
    // Create board at starting position.
    lczero::ChessBoard board;
    board.SetFromFen(lczero::ChessBoard::kStartposFen);
    // Keep history of positions.
    lczero::PositionHistory history;
    history.Reset(board, 0, 0);

    lczero::InputPlanes planes = lczero::EncodePositionForNN(history, 8, lczero::FillEmptyHistory::NO);
    torch::Tensor tensor = torch::ones({112, 8, 8});
    return tensor;
}

torch::Tensor load_position() {
    return parse_chunk();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_position", &load_position, "Load a position");
}
