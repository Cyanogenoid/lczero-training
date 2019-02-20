#include <torch/extension.h>

#include "chunk_generated.h"
#include "flatbuffers.h"

#include <iostream>
#include <fstream>
#include <random>


int build_wdl(const flatlczero::Game* game, const flatlczero::Side side_to_move) {
    // from side-to-move's perspective
    // win: 0
    // draw: 1
    // loss: 2
    auto winner = game->winner();
    if (winner == flatlczero::Result::Result_Draw) {
        return 1;
    }
    if ((winner == flatlczero::Result::Result_White) == (side_to_move == flatlczero::Side::Side_White)) {
        return 0;
    }
    return 2;
}

std::tuple<torch::Tensor, torch::Tensor> build_policy(const flatlczero::Policy* policy) {
    auto targets = torch::zeros(1858);
    auto legals = torch::zeros(1858);
    auto targets_a = targets.accessor<float, 1>();
    auto legals_a = targets.accessor<float, 1>();
    for (size_t i = 0; i < policy->index()->Length(); i++) {
        auto index = policy->index()->Get(i);
        auto probability = policy->probability()->Get(i);
        targets_a[index] = probability;
        legals_a[index] = 1;
    }
    return {targets, legals};
}

void build_pieces(torch::Tensor& planes, const flatlczero::Pieces* pieces, bool flip) {
    auto planes_a = planes.accessor<float, 2>();
    for (size_t i = 0; i < pieces->indices()->Length(); i++) {
        auto index = pieces->indices()->Get(i);
        auto type = pieces->types()->Get(i);
        if (flip) {
            index ^= 0b111000;  // col unchanged, row flipped
        }
        // assumes that piece types are in the order p,n,b,r,q,k, starting from 0
        planes_a[type][index] = 1;
    }
}

void build_position(torch::Tensor& planes, const flatlczero::Position* position) {
    auto white = position->white();
    auto black = position->black();
    auto our_planes = planes.slice(0, 0, 6);
    auto their_planes = planes.slice(0, 6, 12);
    if (position->side_to_move() == flatlczero::Side::Side_White) {
        build_pieces(our_planes, white, false);
        build_pieces(their_planes, black, false);
    } else {
        build_pieces(our_planes, black, true);
        build_pieces(their_planes, white, true);
    }
    if (position->repetitions() >= 1) {
        planes[12].fill_(1);
    }
}

torch::Tensor build_input(const flatlczero::Game* game, const int position_index) {
    const int planes_per_position = 13;
    const int history = 8;
    auto planes = torch::zeros({history * planes_per_position + 8, 64});

    for (int i = position_index, plane_index = 0;
         i > position_index - history && i >= 0;
         i--, plane_index += planes_per_position) {
        auto position_planes = planes.slice(0, plane_index, plane_index + planes_per_position);
        auto position = game->states()->Get(i)->position();
        build_position(position_planes, position);
    }
    int offset = history * planes_per_position;
    auto position = game->states()->Get(position_index)->position();
    planes[offset + 0].fill_(position->us_ooo());
    planes[offset + 1].fill_(position->us_oo());
    planes[offset + 2].fill_(position->them_ooo());
    planes[offset + 3].fill_(position->them_oo());
    planes[offset + 4].fill_(position->side_to_move());
    planes[offset + 5].fill_(position->rule_50());
    planes[offset + 6].fill_(0);
    planes[offset + 7].fill_(1);

    return planes.view({planes.size(0), 8, 8});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int> load(char* data) {
    auto game = flatlczero::GetGame(data);

    // select random position from the game
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(0, game->states()->Length() - 1);
    int position_index = distr(eng);

    auto state = game->states()->Get(position_index);

    torch::Tensor input = build_input(game, position_index);
    torch::Tensor policy, legals;
    std::tie(policy, legals) = build_policy(state->policy());
    int wdl = build_wdl(game, state->position()->side_to_move());
    return {input, policy, legals, wdl};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load", &load, "Load a position from a chunk");
}
