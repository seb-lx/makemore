#pragma once

#include <iostream>
#include <random>
#include <array>
#include <chrono>

#include "input.h"
#include "tensor.h"
#include "functional.h"


namespace makemore {

namespace models {


struct Config {
    static constexpr auto LETTERS_IN_ALPHABET = std::size_t{ 26 };
    static constexpr auto N_TOKENS = std::size_t{ 27 };
    static constexpr auto N_NAMES_TO_GENERATE = std::size_t{ 20 };

    static constexpr auto CONTEXT_LENGTH = std::size_t{ 3 }; // BLOCK_SIZE
    static constexpr auto EMBEDDING_DIM = std::size_t{ 10 };
    static constexpr auto HIDDEN_DIM = std::size_t{ 200 };
    
    static constexpr auto BATCH_SIZE = std::size_t{ 32 };

    static constexpr auto N_ITERATIONS_NN = std::size_t{ 10 };
    static constexpr auto N_ITERATIONS_MLP = std::size_t{ 10000 };

    static constexpr auto LEARNING_RATE_NN = 50.0f;
    static constexpr auto LEARNING_RATE_MLP = 0.1f;
    static constexpr auto LEARNING_RATE_MLP_DECAYED = 0.01f;

    static constexpr auto RANDOM_SEED = 2147483647u;
};


struct NN {
    static auto run() -> void {
        using makemore::Tensor;

        auto gen = std::mt19937{ Config::RANDOM_SEED };

        ////////////////////////////
        ////// Create Dataset //////
        ////////////////////////////
        const auto filename = std::string{ "data/names.txt" };
        auto sf = makemore::SourceFile(filename);
        auto names = sf.get_lines();
        std::cout << "Read " << filename << " with size " << names.size() << "\n";
        if (names.size() == 0) return;

        auto [itos_arr, stoi_arr] = makemore::functional::create_char_mapping_arrays();


        //auto N = Tensor<int>::zeros({ N_TOKENS, N_TOKENS });

        auto xs = std::vector<std::size_t>{};
        auto ys = std::vector<std::size_t>{};

        // Fill N with statistics from names
        const auto start_end_token_idx = std::size_t{ 0 };
        for (const auto& name: names) {
            auto idx1 = start_end_token_idx; 

            for (char c: name) {
                if (c < 'a' || c > 'z') throw std::runtime_error("Found char outside a-z range");

                auto idx2 = stoi_arr[static_cast<unsigned char>(c)];
                //N(idx1, idx2) += 1;
                xs.push_back(idx1);
                ys.push_back(idx2);
                
                idx1 = idx2;
            }

            //N(idx1, start_end_token_idx) += 1;
            xs.push_back(idx1);
            ys.push_back(start_end_token_idx);
        }

        auto xsT = Tensor<std::size_t>::from(xs, { xs.size() });
        auto ysT = Tensor<std::size_t>::from(ys, { ys.size() });

        // == "W = torch.randn((27, 27), generator=g, requires_grad=True)"
        auto W = Tensor<float>::randn({ Config::N_TOKENS, Config::N_TOKENS }, gen, true);


        //////////////////////
        ////// TRAINING //////
        //////////////////////

        auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < Config::N_ITERATIONS_NN; ++i) {
            std::cout << "Iteration = " << i << "\n";

            // ===== Forward Pass =====
            // encode inputs as one hot vectors
            auto xenc = makemore::functional::one_hot_1D<std::size_t>(
                xsT, Config::N_TOKENS
            ).cast<float>();

            //xenc.print(false);

            // == "logits = xenc @ W"
            auto logits = xenc.matmul(W);

            // == "counts = logits.exp()"
            auto counts = logits.exp();

            // == "counts / counts.sum(1, keepdims=True)"
            auto probs = counts / counts.sum(1, true); 

            // == "-probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()"
            auto ys_one_hot = makemore::functional::one_hot_1D<std::size_t>(ysT, Config::N_TOKENS).cast<float>();
            auto correct_probs = (probs * ys_one_hot).sum(1);
            auto nll_loss = correct_probs.log().mean().neg();
            auto reg_loss = (W * W).mean() * 0.01f;
            auto loss = nll_loss + reg_loss;

            std::cout << "\tLoss = " << loss.item() << "\n";

            // ===== Backward Pass =====
            W.zero_grad();
            loss.backward();

            // ===== Update =====
            // == "W.data += -50 * W.grad"
            auto& w_data = W.data_mut();
            const auto& w_grad = *W.grad_ptr();
            for (std::size_t j = 0; j < w_data.size(); ++j) {
                w_data[j] += -Config::LEARNING_RATE_NN * w_grad[j];
            }
        }

        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start
        );
        std::cout << "\nTraining Time: " << duration.count() << " s" << "\n\n";

        //////////////////////
        ////// SAMPLING //////
        //////////////////////
        gen.seed(Config::RANDOM_SEED);

        start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < Config::N_NAMES_TO_GENERATE; ++i) {
            auto out = std::vector<char>{};
            auto idx = std::size_t{ 0 };

            while (true) {

                // ==
                // "
                // xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                // logits = xenc @ W # predict log-counts
                // counts = logits.exp() # counts, equivalent to N
                // p = counts / counts.sum(1, keepdims=True) # probabilities for next character
                // "
                auto idx_tensor = Tensor<std::size_t>::from({ idx }, std::vector<size_t>{ 1 });
                auto xenc = makemore::functional::one_hot_1D<std::size_t>(
                    idx_tensor, Config::N_TOKENS
                ).cast<float>();

                auto logits = xenc.matmul(W);
                auto counts = logits.exp();
                auto p = counts / counts.sum(1, true);

                // == "ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()"
                auto p_row = p.row(0);
                auto probs_vec = std::vector<float>{};
                for (std::size_t j = 0; j < Config::N_TOKENS; ++j) probs_vec.push_back(p_row.at_flat(j));
                    
                auto dist = std::discrete_distribution<std::size_t>(
                    probs_vec.begin(),
                    probs_vec.end()
                );

                idx = dist(gen);

                if (idx == 0) break;
                out.push_back(itos_arr[idx]);
            }

            for (const auto& c: out) std::cout << c;
            std::cout << "\n";
        }

        duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start
        );
        std::cout << "\nSampling Time: " << duration.count() << " s" << "\n\n";
    }
};


struct MLP {
    static auto run() -> void {
        using makemore::Tensor;

        auto gen = std::mt19937{ Config::RANDOM_SEED };

        const auto filename = std::string{ "data/names.txt" };
        auto sf = makemore::SourceFile(filename);
        auto names = sf.get_lines();
        std::cout << "Read " << filename << " with size " << names.size() << "\n";
        if (names.size() == 0) return;

        auto [itos_arr, stoi_arr] = makemore::functional::create_char_mapping_arrays();
        
        // Shuffle data
        auto rng = std::default_random_engine{};
        std::ranges::shuffle(names, rng);

        const auto n = names.size();

        const auto n1 = static_cast<std::size_t>(0.8 * static_cast<double>(n));
        const auto n2 = static_cast<std::size_t>(0.9 * static_cast<double>(n));

        const auto X_train_view = names | std::views::take(n1);
        const auto X_val_view = names | std::views::drop(n1) | std::views::take(n2 - n1);
        const auto X_test_view = names | std::views::drop(n2);

        const auto [X_train, Y_train, total_train] = makemore::functional::build_dataset<Config::CONTEXT_LENGTH>(
            X_train_view,
            stoi_arr
        );
        const auto [X_val, Y_val, total_val] = makemore::functional::build_dataset<Config::CONTEXT_LENGTH>(
            X_val_view,
            stoi_arr
        );
        const auto [X_test, Y_ttest, total_test] = makemore::functional::build_dataset<Config::CONTEXT_LENGTH>(
            X_test_view,
            stoi_arr
        );

        // make sure to scale down to avoid nan
        auto C = Tensor<float>::randn({ Config::N_TOKENS, Config::EMBEDDING_DIM }, gen, true) * 0.1f;
        // Xavier initialization: scale by sqrt(1 / in)
        auto W1 = Tensor<float>::randn({ Config::CONTEXT_LENGTH * Config::EMBEDDING_DIM, Config::HIDDEN_DIM }, gen, true)
                * (1.0f / std::sqrt(static_cast<float>(Config::CONTEXT_LENGTH * Config::EMBEDDING_DIM)));
        auto W2 = Tensor<float>::randn({ Config::HIDDEN_DIM, Config::N_TOKENS }, gen, true)
                * (1.0f / std::sqrt(static_cast<float>(Config::HIDDEN_DIM)));
        auto b1 = Tensor<float>::zeros({ Config::HIDDEN_DIM }, true);
        auto b2 = Tensor<float>::zeros({ Config::N_TOKENS }, true);

        // auto W1 = Tensor<float>::randn({ Config::CONTEXT_LENGTH * Config::EMBEDDING_DIM, Config::HIDDEN_DIM }, gen, true) * 0.1f;
        // auto b1 = Tensor<float>::randn({ Config::HIDDEN_DIM }, gen, true);
        // auto W2 = Tensor<float>::randn({ Config::HIDDEN_DIM, Config::N_TOKENS }, gen, true) * 0.1f;
        // auto b2 = Tensor<float>::randn({ Config::N_TOKENS }, gen, true);
        auto params = std::vector<Tensor<float>>{ C, W1, b1, W2, b2 };

        //////////////////////
        ////// TRAINING //////
        //////////////////////

        auto start = std::chrono::steady_clock::now();

        auto index_dist = std::uniform_int_distribution<std::size_t>(0, total_train - 1);

        for (std::size_t i = 0; i < Config::N_ITERATIONS_MLP; ++i) {
            
            // create batches
            auto batch_indices = std::vector<std::size_t>(Config::BATCH_SIZE);
            for (auto& idx : batch_indices) idx = index_dist(gen);
            
            // x batch of shape (BATCH_SIZE, CONTEXT_LENGTH)
            // y batch of shape (BATCH_SIZE,)
            auto x_batch_data = std::vector<std::size_t>{};
            auto y_batch_data = std::vector<std::size_t>{};
            x_batch_data.reserve(Config::BATCH_SIZE * Config::CONTEXT_LENGTH);
            y_batch_data.reserve(Config::BATCH_SIZE);
            
            for (auto idx : batch_indices) {
                for (std::size_t j = 0; j < Config::CONTEXT_LENGTH; ++j) x_batch_data.push_back(X_train[idx, j]);
                y_batch_data.push_back(Y_train.at_flat(idx));
            }
            
            auto X_batch = Tensor<std::size_t>::from(
                x_batch_data,
                std::vector<std::size_t>{ Config::BATCH_SIZE, Config::CONTEXT_LENGTH }
            );
            auto Y_batch = Tensor<std::size_t>::from(
                y_batch_data,
                std::vector<std::size_t>{ Config::BATCH_SIZE }
            );
            
            // ===== Forward Pass =====
            auto one_hot_X = makemore::functional::one_hot_1D(X_batch.flatten(), Config::N_TOKENS).cast<float>();
            auto embeddings = one_hot_X.matmul(C).view({ Config::BATCH_SIZE, Config::CONTEXT_LENGTH * Config::EMBEDDING_DIM });
            auto h = (embeddings.matmul(W1) + b1).tanh();
            auto logits = h.matmul(W2) + b2;
            auto logits_stable = logits - logits.max(1, true);
            auto counts = logits_stable.exp();
            auto probs = counts / counts.sum(1, true);
            auto ys_one_hot = makemore::functional::one_hot_1D<std::size_t>(Y_batch, Config::N_TOKENS).cast<float>();
            auto correct_probs = (probs * ys_one_hot).sum(1);
            auto loss = correct_probs.log().mean().neg();


            // ===== Backward Pass =====
            for (auto& p: params) p.zero_grad();
            loss.backward();

            // ===== Update =====
            auto lr = (i < (Config::N_ITERATIONS_MLP / 2)) ? Config::LEARNING_RATE_MLP : Config::LEARNING_RATE_MLP_DECAYED;
            for (auto& p: params) {
                auto& p_data = p.data_mut();
                const auto& p_grad = *p.grad_ptr();
                for (std::size_t j = 0; j < p_data.size(); ++j) {
                    p_data[j] += -lr * p_grad[j];
                }
            }

            if (i % 100 == 0) std::cout << "Iteration " << i << ", Loss = " << loss.item() << ", using lr = " << lr << "\n";
        }

        // old training without mini batches
        // for (std::size_t i = 0; i < Config::N_ITERATIONS_MLP; ++i) {
        //     std::cout << "Iteration = " << i << "\n";

        //     // no minibatches for now

        //     // ===== Forward Pass =====
        //     // todo: for now use one hot and matmul for embeddings, later need better indexing: "C[X]"
        //     auto one_hot_X = makemore::functional::one_hot_1D(X_train.flatten(), Config::N_TOKENS).cast<float>();
        //     auto embeddings = one_hot_X.matmul(C).view({ total_train, Config::CONTEXT_LENGTH * Config::EMBEDDING_DIM });

        //     auto h = (embeddings.matmul(W1) + b1).tanh();
        //     auto logits = h.matmul(W2) + b2;
        //     auto logits_stable = logits - logits.max(1, true); // logits - max(logits) for numerical stability

        //     auto counts = logits_stable.exp();
        //     auto probs = counts / counts.sum(1, true);
        //     auto ys_one_hot = makemore::functional::one_hot_1D<std::size_t>(Y_train, Config::N_TOKENS).cast<float>();
        //     auto correct_probs = (probs * ys_one_hot).sum(1);
        //     auto loss = correct_probs.log().mean().neg();

        //     std::cout << "\tLoss = " << loss.item() << "\n";

        //     // ===== Backward Pass =====
        //     for (auto& p: params) p.zero_grad();
        //     loss.backward();

        //     // ===== Update =====
        //     auto lr = (i < (Config::N_ITERATIONS / 2)) ? Config::LEARNING_RATE_MLP : Config::LEARNING_RATE_MLP_DECAYED;
        //     for (auto& p: params) {
        //         auto& p_data = p.data_mut();
        //         const auto& p_grad = *p.grad_ptr();
        //         for (std::size_t j = 0; j < p_data.size(); ++j) {
        //             p_data[j] += -lr * p_grad[j];
        //         }
        //     }
        //     std::cout << "\t\tusing lr: " << lr << "\n";
        // }


        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start
        );
        std::cout << "\nTraining Time: " << duration.count() << " s" << "\n\n";
        std::cout << "\n\n";

        //////////////////////
        ////// LOSSES ////////
        //////////////////////   
        auto compute_loss = [&](const Tensor<std::size_t>& X, const Tensor<std::size_t>& Y, std::size_t total) {
            auto one_hot_X = makemore::functional::one_hot_1D(X.flatten(), Config::N_TOKENS).cast<float>();
            auto embeddings = one_hot_X.matmul(C).view({ total, Config::CONTEXT_LENGTH * Config::EMBEDDING_DIM });
            
            auto h = (embeddings.matmul(W1) + b1).tanh();
            auto logits = h.matmul(W2) + b2;
            auto logits_stable = logits - logits.max(1, true);
            
            auto counts = logits_stable.exp();
            auto probs = counts / counts.sum(1, true);
            auto ys_one_hot = makemore::functional::one_hot_1D<std::size_t>(Y, Config::N_TOKENS).cast<float>();
            auto correct_probs = (probs * ys_one_hot).sum(1);
            
            return correct_probs.log().mean().neg().item();
        };

        std::cout << "Train loss: " << compute_loss(X_train, Y_train, total_train) << "\n";
        std::cout << "Val loss:   " << compute_loss(X_val, Y_val, total_val) << "\n";
        std::cout << "Test loss:  " << compute_loss(X_test, Y_ttest, total_test) << "\n";
        std::cout << "\n\n";

        //////////////////////
        ////// SAMPLING //////
        //////////////////////
        gen.seed(Config::RANDOM_SEED);

        std::cout << "Sampling names after training:\n";

        start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < Config::N_NAMES_TO_GENERATE; ++i) {
            auto out = std::vector<char>{};

            auto context = std::vector<std::size_t>(Config::CONTEXT_LENGTH, 0);

            while (true) {
                auto context_tensor = Tensor<std::size_t>::from(context, { 1, Config::CONTEXT_LENGTH });
                
                auto one_hot_ctx = makemore::functional::one_hot_1D(context_tensor.flatten(), Config::N_TOKENS).cast<float>();
                auto emb = one_hot_ctx.matmul(C).view({ 1, Config::CONTEXT_LENGTH * Config::EMBEDDING_DIM });
                
                // Forward pass
                auto h = (emb.matmul(W1) + b1).tanh();
                auto logits = h.matmul(W2) + b2;
                
                // Softmax (with numerical stability)
                auto logits_stable = logits - logits.max(1, true);
                auto counts = logits_stable.exp();
                auto probs = counts / counts.sum(1, true);
                
                auto probs_row = probs.row(0);
                auto probs_vec = std::vector<float>{};
                for (std::size_t j = 0; j < Config::N_TOKENS; ++j) {
                    probs_vec.push_back(probs_row.at_flat(j));
                }
                
                auto dist = std::discrete_distribution<std::size_t>(probs_vec.begin(), probs_vec.end());
                auto next_idx = dist(gen);
                
                // Update context: drop first, append new
                context.erase(context.begin());
                context.push_back(next_idx);
                
                // Check for end token
                if (next_idx == 0) break;
                out.push_back(itos_arr[next_idx]);
            }
            
            // Print generated name
            for (const auto& c : out) std::cout << c;
            std::cout << "\n";
        }

        duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start
        );
        std::cout << "\nSampling Time: " << duration.count() << " s" << "\n\n";
    }
};


} // namespace models

} // namespace makemore
