#pragma once

#include <tuple>
#include <algorithm>
#include <ranges>

#include "tensor.h"


namespace makemore {

namespace functional {


// auto create_char_mappings() -> std::pair<std::map<std::size_t, char>, std::map<char, size_t>> {
//     auto itos = std::map<size_t, char>{};
//     auto stoi = std::map<char, size_t>{};

//     // Special character . for start and end of words
//     itos[0] = '.';
//     stoi['.'] = 0;

//     auto idx = std::size_t{ 1 };
//     for (char c = 'a'; c <= 'z'; ++c) {
//         itos[idx] = c;
//         stoi[c] = idx;
//         ++idx;
//     }

//     return { std::move(itos), std::move(stoi) };
// }

auto create_char_mapping_arrays() -> std::pair<std::array<char, 27>, std::array<std::size_t, 256>> {
    std::array<char, 27> itos_arr{};
    std::array<std::size_t, 256> stoi_arr{};

    // Special character . for start and end of words
    itos_arr[0] = '.';
    stoi_arr[static_cast<unsigned char>('.')] = 0;

    for (char c = 'a'; c <= 'z'; ++c) {
        auto idx = static_cast<std::size_t>(c - 'a' + 1);
        itos_arr[idx] = c;
        stoi_arr[static_cast<unsigned char>(c)] = idx;
    }

    return { std::move(itos_arr), std::move(stoi_arr) };
}



template<Arithmetic T>
auto one_hot_1D(const Tensor<T>& t, std::size_t num_classes) -> Tensor<T>
requires std::is_integral_v<T>
{
    if (t.shape().size() != 1) {
        throw std::invalid_argument(
            std::format(
                "[functional::one_hot_1D] only works for 1D tensors, tensor shape size: {}",
                t.shape().size()
            )
        );  
    }

    auto result = Tensor<T>::zeros({ t.shape()[0], num_classes }, false);

    #pragma omp parallel for
    for (std::size_t i = 0; i < t.size(); ++i) {
        auto data_idx = i * t.strides()[0] + t.offset();
        auto class_idx = static_cast<std::size_t>(t.data()[data_idx]);

        if (class_idx >= num_classes) {
            throw std::out_of_range(
                std::format(
                    "[functional::one_hot_1D] class idx {} out of class range {} at position {}",
                    class_idx, num_classes, i
                )
            );
        }

        result[i, class_idx] = T{ 1 };
    }

    return result;
}


template<std::size_t N>
auto build_dataset(
    const auto& names,
    const std::array<std::size_t, 256>& stoi_arr
) -> std::tuple<Tensor<std::size_t>, Tensor<std::size_t>, std::size_t> 
{
    static constexpr auto start_end_token_idx = std::size_t{ 0 };

    auto xs = std::vector<std::size_t>{};
    auto ys = std::vector<std::size_t>{};

    auto total = std::accumulate(
        names.begin(),
        names.end(),
        std::size_t{ 0 },
        [](std::size_t acc, auto& name){ return acc + name.size() + 1; }
    );
    xs.reserve(total * N);
    ys.reserve(total);

    auto context = std::array<std::size_t, N>{};

    // auto i = std::size_t{ 0 };
    for (const auto& name: names) {
        // if (i == 5) break;

        context.fill(start_end_token_idx);
        // std::cout << "\nname: " << name << "\n";

        for (char c: name) {
            if (c < 'a' || c > 'z') throw std::runtime_error("Found char outside a-z range");

            auto idx = stoi_arr[static_cast<unsigned char>(c)];

            // add context to xs and corresponding label idx to ys
            for (std::size_t i = 0; i < N; ++i) xs.push_back(context[i]);
            ys.push_back(idx);

            // shift context window to the left, i.e. pop first element and add current char idx to the end
            for (std::size_t i = 0; i < N - 1; ++i) context[i] = context[i + 1];
            context[N - 1] = idx;

            // for (const auto& e: context) std::cout << itos_arr[e] << " ";
            // std::cout << " ----> " << itos_arr[idx] << "\n";
        }

        
        for (std::size_t i = 0; i < N; ++i) xs.push_back(context[i]);
        ys.push_back(start_end_token_idx);

        // for (const auto& e: context) std::cout << itos_arr[e] << " ";
        // std::cout << " ----> " << itos_arr[start_end_token_idx] << "\n";

        // ++i;
    }

    return {
        Tensor<std::size_t>::from(xs, std::vector<std::size_t>{ total, N }),
        Tensor<std::size_t>::from(ys, std::vector<std::size_t>{ total }),
        total
    };
}


} // namespace functional

} // namespace makemore

