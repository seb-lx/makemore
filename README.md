# makemore
This repository contains a C++ implementation based on the makemore project built in Python by Andrej Karpathy in his "Neural Networks: Zero to Hero" series (https://karpathy.ai/zero-to-hero.html). The goal is to implement all functionalities from scratch in C++ to match the Python code in the lecture.

The goal of the makemore project is to build language models that can generate new names based on the training on a list of over 30k names (see data/readme.md).

The following models are implemented (WIP):
- bigram model using one layer NN
- MLP
- ...


## tensor library
Since the goal is to follow the tutorial in modern C++, all the used Pytorch functionalities are also implemented from scratch. This includes a header-only tensor library (include/tensor.h) modeled according to the Pytorch tensor API. This tensor class is used in all the language model implementations.

Note that although similar features are implemented, it is of course not heavily optimized like Pytorch and thus significantly slower. Also no GPU support.

### features
- n-dimensional tensors with arbitrary shapes
- templated with arithmetic types
- automatic differentiation (autograd)
- broadcasting
- strided memory access of the underlying 1D data
- strided memory access allows for zero-copy views of tensors that share the same underlying data for memory efficiency (e.g. transpose contains same data but needs to be accessed differently - i.e. by swapping strides - since memory layout is now non-contigous with respect to the transpose)
- slicing to access only parts of a tensor (e.g. a row) by simply returning a view of the same underlying data with different offset
- some trivial loops are parallelized with OpenMP for some minor performance benefits

### design
A tensor is thus just a handle and points to a separate implementation containing the data.
```
┌─────────────┐      shared_ptr      ┌──────────────┐
│   Tensor    │ ──────────────────►  │  TensorImpl  │
│  (handle)   │                      │   (storage)  │
│ shape       │                      │ data[]       │
│ strides     │                      │ grad[]       │
│ offset      │                      │ requires_grad│
└─────────────┘                      └──────────────┘
```
A tensor also points to a gradient node in the computational graph. This is needed for views to have its own gradient node.
```
┌─────────────┐                    ┌────────────┐
│   Tensor    │      shared_ptr    │  GradNode  │
│             │ ─────────────────► │ parents[]  │
│             │                    │ backward() │
└─────────────┘                    └────────────┘
```

### examples
```cpp
#include "tensor.h"
using namespace makemore;

// ==================== Factory methods  ====================
auto a = Tensor<float>::zeros({ 3, 4 });                        // 3x4 tensor of zeros
auto b = Tensor<float>::ones({ 3, 4 });                         // 3x4 tensor of ones
auto c = Tensor<float>::full({ 2, 2 }, 5.0f);                   // 2x2 filled with 5.0
auto d1 = Tensor<float>::randn({ 3, 3 }, std::mt19937{ 42 });   // 3x3 random normal
auto d2 = Tensor<float>::randn(0, 10, { 3, 3 });                // 3x3 random uniform in [low, high] 

auto e = Tensor<float>::from({ 1, 2, 3, 4 }, { 2, 2 });         // from data
auto f = Tensor<float>::from(v, { 3, 4 });                      // from vector
auto g = Tensor<float>::from({{ 1, 2 }, { 3, 4 }});             // from 2d initializer list
auto h = Tensor<float>::from(t);                                // from tensor

auto i = Tensor<float>::zeros_like(t);                          // tensor of zeros with t's shape
auto j = Tensor<float>::ones_like(t);                           // tensor of ones with t's shape

auto W = Tensor<float>::zeros({ 27, 27 }, true);                // with gradients, requires_grad = true

// ==================== Element Access ====================
float val = a[1, 2];                                            // get element at row 1, col 2
a[0, 0] = 3.14f;                                                // set element
float scalar = t.item();                                        // extract from single-element tensor

// ==================== Views (zero-copy) ====================
auto row0 = a.row(0);                // first row (1D view)
auto col1 = a.col(1);                // second column (1D view)
auto sub = a.slice(0, 2);            // slice along dim 0 at index 2
auto aT = a.transpose();             // transpose (just swaps strides)

// ==================== Arithmetic (with broadcasting) ====================
auto sum = a + b;                    // element-wise add
auto diff = a - b;                   // element-wise subtract
auto prod = a * b;                   // element-wise multiply (Hadamard)
auto quot = a / b;                   // element-wise divide
auto scaled = a * 2.0f;              // scalar multiply
auto shifted = a + 1.0f;             // scalar add

a += b;                              // in-place (only for non-gradient tensors)
a *= 2.0f;                           // in-place (only for non-gradient tensors)

// ==================== Matrix Operations ====================
auto C = a.matmul(b.transpose());    // matrix multiply: (3x4) @ (4x3) = (3x3)
auto dot = x.matmul(y);              // 1D dot product

// ==================== Reductions ====================
auto total = a.sum();                // sum all elements -> scalar
auto row_sum = a.sum(1);             // sum along axis 1 i.e. collapse cols
auto row_sum_kd = a.sum(1, true);    // keepdim=true
auto mean_val = a.mean();            // mean of all elements
auto col_mean = a.mean(0);           // mean along axis 0
auto max_val = a.max();              // global max
auto min_per_row = a.min(1);         // min along axis 1

// ==================== Unary Operations ====================
auto ex = a.exp();                   // e^x
auto lg = a.log();                   // ln(x)
auto sq = a.sqrt();                  // sqrt(x)
auto th = a.tanh();                  // tanh(x)
auto sg = a.sigmoid();               // 1 / (1 + e^-x)
auto rl = a.relu();                  // max(0, x)
auto ng = a.neg();                   // -x

// ==================== Type Casting ====================
auto ints = Tensor<int>::zeros({ 3, 3 });
auto floats = ints.cast<float>();

// ==================== Autograd ====================
auto x = Tensor<float>::from({ 2.0f, 3.0f }, std::vector<std::size_t>{ 2 }, true);
auto y = x * x;           // y = x^2
auto z = y.sum();         // z = sum(x^2)
x.zero_grad();            // Zero gradients
z.backward();             // compute gradients

const auto* grad = x.grad_ptr();  // access gradients: dz/dx = 2x => grad = {4.0, 6.0}

// ==================== Example: NN Training Loop ====================
auto W = Tensor<float>::randn({ 27, 27 }, std::mt19937{ 42 }, true);

for (int iteration = 0; iteration < 100; ++iteration) {

    // Forward pass
    auto logits = xenc.matmul(W);
    auto probs = logits.exp() / logits.exp().sum(1, true);
    auto loss = (probs * targets).sum(1).log().mean().neg();
    
    // Backward pass
    W.zero_grad();
    loss.backward();
    
    // SGD update
    auto& data = W.data_mut();
    const auto& grad = *W.grad_ptr();
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= 0.1f * grad[i];
    }
}

// ==================== Utility ====================
a.print();                      // print tensor info and data
a.print(false);                 // print info only (no data dump for large tensors)
auto shape = a.shape();         // { 3, 4 }
auto size = a.size();           // 12
bool grad = a.requires_grad();  // false
```

### gradient computation
gradients are computed according to the chain rule. for scalar values this means ```dL/dx = dL/dy * dy/dx``` (ref [micrograd](https://github.com/seb-lx/micrograd)). for tensors this must be computed for every element and broadcasting/accumulation must be taken into account.

### comprehensive test suite
tests/test_tensor.cpp contains exhaustive test cases for the tensor class. It uses doctest via its header file include/doctest.h


## possible future improvements
- could also force contigous layout by creating a copy of the input tensors. this will cost copying the input tensor and uses more memory. need to evaluate if this is worth it compared to the current compute-intensive index calculations.
- compared to the pytorch version this tensor implementation is rather slow. this becomes apparent when training with the large input data set which leads to huge tensors and compute intensive operations. pytorch is of course heavily optimized and implemented for efficiency. this was not the focus here, some known peformance issues: mostly serial implementation (only some openmp pragmas for trivially parallelizable parts, backward pass completely serial), naive matrix multiplication, not cache-friendly / no cache optimizations, strided access with compute intensive index calculations, atm inefficient embedding (one hot and matmul), excessive heap memory allocations, ...


## build
This project can be build using cmake. Requires C++23.

### manual build
```bash
# debug
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug

# release
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

### build scripts
```bash
./build_and_run_debug.sh        # builds and runs main executable in debug mode
./build_and_run_release.sh      # builds and runs main executable in release mode
./test.sh                       # builds and runs test suite
```
