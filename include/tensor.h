#pragma once

#include <optional>
#include <stdexcept>
#include <numeric>
#include <concepts>
#include <type_traits>
#include <format>
#include <algorithm>
#include <random>
#include <functional>
#include <memory>
#include <cassert>
#include <unordered_set>
#include <iostream>
#include <cmath>
#include <tuple>

#include <omp.h>


namespace makemore {

namespace internal {


struct GradNode {
    std::vector<std::shared_ptr<GradNode>> parents; 
    std::function<void()> backward_fn;

    GradNode(): parents{}, backward_fn{} {};

    GradNode(
        std::vector<std::shared_ptr<GradNode>> parents,
        std::function<void()> backward_fn
    ):
        parents{ std::move(parents) },
        backward_fn{ std::move(backward_fn) }
    {}
};


template<typename T>
struct TensorImpl {
public:
    std::vector<T> data;
    std::unique_ptr<std::vector<T>> grad;
    bool requires_grad = false;


public:
    TensorImpl(std::size_t size, T fill_value, bool requires_grad = false):
        data{},
        grad{ nullptr },
        requires_grad{ requires_grad }
    {
        this->data.resize(size, fill_value);

        if (this->requires_grad) {
            this->grad = std::make_unique<std::vector<T>>();
            this->grad->resize(size, T{ 0 });
        }
    }

    TensorImpl(std::vector<T> data, bool requires_grad = false):
        data{ std::move(data) },
        grad{ nullptr },
        requires_grad{ requires_grad }
    {
        if (this->requires_grad) {
            this->grad = std::make_unique<std::vector<T>>();
            this->grad->resize(this->data.size(), T{ 0 });
        }
    }

public:
    auto zero_grad() -> void {
        if (grad) std::fill(grad->begin(), grad->end(), T{ 0 });
    }
};


} // namespace internal



// Create arithmetic concept to constraint the type of values 
// that can be stored within a tensor.
template<typename T> concept Arithmetic = std::is_arithmetic_v<T>;

template<Arithmetic T>
class Tensor {
public:
    friend struct internal::TensorImpl<T>; // todo: needed?

private:
    // A Tensor<T> itself does not hold and manage its data.
    // A separate TensorImpl is responsible for that, aswell as 
    // keep track of gradients. This enables efficient views of
    // tensors without copying the underlying data.
    std::shared_ptr<internal::TensorImpl<T>> impl_;

    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t impl_data_offset_;

    // In order to correctly build the computational graph
    // and compute the derivatives of each node, a GradNode<T>
    // is separate from the actual Tensor and TensorImpl.
    // This is needed to correctly compute gradients for views
    // of tensors. Otherwise different views of the same data would
    // refer to the same gradient state, i.e. both views would 
    // correspond to the same node.
    std::shared_ptr<internal::GradNode> grad_node_;


private: // Private constructors

    Tensor(): impl_{ nullptr }, grad_node_{ nullptr }, shape_{}, strides_{}, impl_data_offset_{ 0 } {}

    Tensor(
        std::shared_ptr<internal::TensorImpl<T>> impl,
        std::vector<std::size_t> shape,
        std::vector<std::size_t> strides,
        const std::size_t impl_data_offset,
        std::shared_ptr<internal::GradNode> grad_node
    ):
        impl_{ std::move(impl) },
        shape_{ (validate_shape(shape), std::move(shape)) },
        strides_{ std::move(strides) },
        impl_data_offset_{ impl_data_offset },
        grad_node_{ std::move(grad_node) }
    {}

    Tensor(std::vector<std::size_t> shape, const T fill_value, bool requires_grad = false):
        impl_{},
        shape_{},
        strides_{},
        impl_data_offset_{ 0 },
        grad_node_{ nullptr }
    {
        validate_shape(shape);

        impl_ = std::make_shared<internal::TensorImpl<T>>(
            compute_size(shape), fill_value, requires_grad
        );

        shape_ = std::move(shape);
        strides_ = compute_strides(shape_);

        if (requires_grad) grad_node_ = std::make_shared<internal::GradNode>();
    }

    static auto validate_shape(const std::vector<std::size_t>& shape) -> void {
        if (shape.empty()) {
            throw std::invalid_argument(
                std::format("[Tensor::validate_shape()] Cannot create tensor from empty shape!")
            );
        }

        for (const auto& d: shape) {
            if (d == std::size_t{ 0 }) {
                throw std::invalid_argument(
                    std::format("[Tensor::validate_shape()] Shape contains a zero at one dimension!")
                );
            }
        }
    }


public: // Static factory methods

    [[nodiscard]] static auto zeros(const std::vector<std::size_t>& shape, bool requires_grad = false) -> Tensor<T> {
        return Tensor<T>(shape, T{ 0 }, requires_grad);
    }

    [[nodiscard]] static auto ones(const std::vector<std::size_t>& shape, bool requires_grad = false) -> Tensor<T> {
        return Tensor<T>(shape, T{ 1 }, requires_grad);
    }

    [[nodiscard]] static auto full(
        const std::vector<std::size_t>& shape,
        const T fill_value,
        bool requires_grad = false
    ) -> Tensor<T> {
        return Tensor<T>(shape, fill_value, requires_grad);
    }

    [[nodiscard]] static auto randn(
        const std::vector<std::size_t>& shape,
        std::mt19937& gen,
        bool requires_grad = false
    ) -> Tensor<T>
    requires std::is_floating_point_v<T>
    {
        auto dist = std::normal_distribution<T>{ 0.0, 1.0 };

        auto impl = std::make_shared<internal::TensorImpl<T>>(
            compute_size(shape), T{ 0 }, requires_grad
        );

        std::for_each(
            impl->data.begin(),
            impl->data.end(),
            [&dist, &gen](T& e){ e = dist(gen); }
        );

        auto strides = compute_strides(shape);
        auto impl_data_offset = std::size_t{ 0 };

        return Tensor<T>(
            impl,
            shape,
            strides,
            impl_data_offset,
            (requires_grad ? std::make_shared<internal::GradNode>() : nullptr)
        );
    }

    [[nodiscard]] static auto randn(
        T low,
        T high,
        const std::vector<std::size_t>& shape,
        std::mt19937& gen,
        bool requires_grad = false
    ) -> Tensor<T>
    requires std::is_integral_v<T>
    {
        auto dist = std::uniform_int_distribution<T>{ low, high };

        auto impl = std::make_shared<internal::TensorImpl<T>>(
            compute_size(shape), T{ 0 }, requires_grad
        );

        std::for_each(
            impl->data.begin(),
            impl->data.end(),
            [&dist, &gen](T& e){ e = dist(gen); }
        );

        auto strides = compute_strides(shape);
        auto impl_data_offset = std::size_t{ 0 };

        return Tensor<T>(
            impl,
            shape,
            strides,
            impl_data_offset,
            (requires_grad ? std::make_shared<internal::GradNode>() : nullptr)
        );
    }

    [[nodiscard]] static auto from(
        const std::vector<T>& data,
        const std::vector<std::size_t>& shape,
        bool requires_grad = false
    ) -> Tensor<T> {
        auto size = compute_size(shape);
        if (data.size() != size) {
            throw std::invalid_argument(
                std::format(
                    "[Tensor::from()] Cannot create tensor from data: data.size() is {}, shape results in size {}",
                    data.size(), size
                )
            );
        }

        auto impl = std::make_shared<internal::TensorImpl<T>>(
            data, requires_grad
        );
        auto strides = compute_strides(shape);
        auto impl_data_offset = std::size_t{ 0 };

        return Tensor<T>(
            impl,
            shape,
            strides,
            impl_data_offset,
            (requires_grad ? std::make_shared<internal::GradNode>() : nullptr)
        );
    }   

    [[nodiscard]] static auto from(std::initializer_list<T> data, bool requires_grad = false) -> Tensor<T> {
        return from(std::vector<T>(data), { data.size() }, requires_grad);
    }

    [[nodiscard]] static auto from(
        std::initializer_list<T> data,
        std::vector<std::size_t> shape,
        bool requires_grad = false
    ) -> Tensor<T> {
        return from(std::vector<T>(data), shape, requires_grad);
    }

    [[nodiscard]] static auto from(
        std::initializer_list<std::initializer_list<T>> data,
        bool requires_grad = false
    ) -> Tensor<T> {
        if (data.size() == 0) {
            throw std::invalid_argument(
                std::format(
                    "[Tensor::from()] Cannot create tensor from initializer list with size {}", data.size()
                )
            );
        }

        auto rows = data.size();
        auto cols = data.begin()->size();

        auto v = std::vector<T>{};
        v.reserve(rows * cols);

        for (const auto& row: data) {
            if (row.size() != cols) {
                throw std::invalid_argument(
                    std::format(
                        "[Tensor::from()] Tensor must have same number of elements in each row"
                        "(mismatch for row size {}, should be {})",
                        row.size(), cols 
                    )
                );
            }
        
            v.insert(v.end(), row.begin(), row.end());
        }

        return from(v, { rows, cols }, requires_grad);
    }

    [[nodiscard]] static auto from(const Tensor<T>& other, bool requires_grad = false) -> Tensor<T> {
        // Since other could be a view, we cannot simply copy its underlying data
        // we need to copy over all elements in the view
        auto result_data = std::vector<T>{};
        result_data.reserve(other.size());

        other.for_each_tensor_element(
            [&result_data](const T& e) { result_data.push_back(e); }
        );

        return from(std::move(result_data), other.shape_, requires_grad);
    }

    [[nodiscard]] static auto zeros_like(const Tensor<T>& other, bool requires_grad = false) -> Tensor<T> {
        return zeros(other.shape_, requires_grad);
    }

    [[nodiscard]] static auto ones_like(const Tensor<T>& other, bool requires_grad = false) -> Tensor<T> {
        return ones(other.shape_, requires_grad);
    }


public: // Member attribute getter

    auto shape() const noexcept -> const std::vector<std::size_t>& {
        return shape_;
    }

    auto strides() const noexcept -> const std::vector<std::size_t>& {
        return strides_;
    }

    auto size() const noexcept -> std::size_t {
        return compute_size(shape_);
    }

    auto offset() const noexcept -> std::size_t {
        return impl_data_offset_;
    }

    auto requires_grad() const noexcept -> bool {
        return impl_->requires_grad;
    }

    auto data() const noexcept -> const std::vector<T>& {
        return impl_->data;
    }

    auto data_mut() noexcept -> std::vector<T>& {
        return impl_->data;
    }

    auto grad_ptr() const noexcept -> const std::vector<T>* {
        return impl_->grad.get();
    }


public: // Single element access

    template<typename... Indices>
    requires (sizeof...(Indices) > 0) && (std::convertible_to<Indices, std::size_t> && ...)
    [[nodiscard]] auto operator[](Indices... indices) noexcept -> T& {
        assert(sizeof...(indices) == shape_.size() && "dimension mismatch");

        auto indices_v = std::vector<std::size_t>{ static_cast<std::size_t>(indices)... };
        auto data_idx = ndim_to_flat_idx(indices_v, strides_) + impl_data_offset_;
        return impl_->data[data_idx];
    }

    template<typename... Indices>
    requires (sizeof...(Indices) > 0) && (std::convertible_to<Indices, std::size_t> && ...)
    [[nodiscard]] auto operator[](Indices... indices) const noexcept -> const T& {
        assert(sizeof...(indices) == shape_.size() && "dimension mismatch");

        auto indices_v = std::vector<std::size_t>{ static_cast<std::size_t>(indices)... };
        auto data_idx = ndim_to_flat_idx(indices_v, strides_) + impl_data_offset_;
        return impl_->data[data_idx];
    }

    [[nodiscard]] auto at(const std::vector<std::size_t>& indices) -> T& {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument(
                std::format(
                    "[Tensor::at()] Indices and tensor shape mismatch: indices size is {} and shape is {}",
                    indices.size(), shape_.size()
                )
            );
        }

        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::invalid_argument(
                    std::format(
                        "[Tensor::at()] Indices and tensor shape mismatch: indices[{}] = {} out of bounds for shape[{}] = {}",
                        i, indices[i], i, shape_[i]
                    )
                );
            }
        }

        return impl_->data[ndim_to_flat_idx(indices, strides_) + impl_data_offset_];
    }
    
    [[nodiscard]] auto at(const std::vector<std::size_t>& indices) const -> const T& {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument(
                std::format(
                    "[Tensor::at()] Indices and tensor shape mismatch: indices size is {} and shape is {}",
                    indices.size(), shape_.size()
                )
            );
        }

        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::invalid_argument(
                    std::format(
                        "[Tensor::at()] Indices and tensor shape mismatch: indices[{}] = {} out of bounds for shape[{}] = {}",
                        i, indices[i], i, shape_[i]
                    )
                );
            }
        }

        return impl_->data[ndim_to_flat_idx(indices, strides_) + impl_data_offset_];
    }

    [[nodiscard]] auto at_flat(std::size_t idx) const -> const T& {
        if (idx >= size()) {
            throw std::out_of_range(
                std::format(
                    "[Tensor::at_flat()] Provided index out of size range: idx is {} and size is {}",
                    idx, size()
                )
            );
        }

        return impl_->data[compute_physical_data_index(idx, compute_strides(shape_))];
    }

    [[nodiscard]] auto at_flat(std::size_t idx) -> T& {
        if (idx >= size()) {
            throw std::out_of_range(
                std::format(
                    "[Tensor::at_flat()] Provided index out of size range: idx is {} and size is {}",
                    idx, size()
                )
            );
        }

        return impl_->data[compute_physical_data_index(idx, compute_strides(shape_))];
    }

    [[nodiscard]] auto grad_at(const std::vector<std::size_t>& indices) const -> T {
        if (!impl_->grad) return T{ 0 };

        auto grad_idx = ndim_to_flat_idx(indices, strides_) + impl_data_offset_;

        return (*impl_->grad)[grad_idx];
    }

public: // View access

    auto slice(std::size_t dim, std::size_t idx) const -> Tensor<T> {
        if (dim >= shape_.size()) {
            throw std::out_of_range(
                std::format(
                    "[Tensor::slice()] Dim out of shape range: Dim is {} and shape size is {}",
                    dim, shape_.size()
                )
            );            
        }

        if (idx >= shape_[dim]) {
            throw std::out_of_range(
                std::format(
                    "[Tensor::slice()] Idx out of shape[dim] range: Idx is {} and shape[dim] is {}",
                    idx, shape_[dim]
                )
            );
        }

        auto result_shape = shape_;
        auto result_strides = strides_;

        result_shape.erase(result_shape.begin() + static_cast<std::ptrdiff_t>(dim));
        result_strides.erase(result_strides.begin() + static_cast<std::ptrdiff_t>(dim));

        auto result_offset = impl_data_offset_ + idx * strides_[dim];

        return Tensor<T>(
            impl_,
            result_shape,
            result_strides,
            result_offset,
            create_view_grad_node()
        );
    }

    auto row(std::size_t idx) const -> Tensor<T> {
        if (shape_.size() != 2) {
            throw std::runtime_error(
                std::format(
                    "[Tensor::row()] row() only works on 2D tensors: shape size is {}",
                    shape_.size()
                )
            );            
        }

        return slice(0, idx); 
    }

    auto col(std::size_t idx) const -> Tensor<T> {
        if (shape_.size() != 2) {
            throw std::runtime_error(
                std::format(
                    "[Tensor::col()] col() only works on 2D tensors: shape size is {}",
                    shape_.size()
                )
            );            
        }
        
        return slice(1, idx); 
    }

    [[nodiscard]] auto flatten() const -> Tensor<T> {
        return view({ size() });
    }

    [[nodiscard]] auto view(const std::vector<std::size_t>& new_shape) const -> Tensor<T> {
        auto new_size = compute_size(new_shape);

        if (new_size != size()) {
            throw std::invalid_argument(
                std::format(
                    "[Tensor::view()] view() new size based on given shape does not match data size"
                )
            );   
        }

        if (!is_contiguous()) {
            throw std::runtime_error(
                std::format(
                    "[Tensor::view()] view() requires contigous tensors"
                )
            );
        }

        return Tensor<T>(
            impl_,
            new_shape,
            compute_strides(new_shape),
            impl_data_offset_,
            create_view_grad_node()
        );
    }


public: // Tensor utilities

    auto item() const -> T {
        if (size() != 1) {
            throw std::runtime_error(
                std::format(
                    "[Tensor::item()] item() only works on tensors with one element: number of elements is {}",
                    size()
                )
            );            
        }  
        
        return at_flat(0);
    }
        
    template<Arithmetic TT>
    [[nodiscard]] auto cast(bool requires_grad = false) const -> Tensor<TT> {
        auto result_data = std::vector<TT>{};
        result_data.reserve(size());

        for_each_tensor_element([&result_data](const T& e) {
            result_data.push_back(static_cast<TT>(e));
        });

        return Tensor<TT>::from(std::move(result_data), shape_, requires_grad);
    }


public: // Tensor operations

    auto operator+(const T scalar) const -> Tensor<T> {
        return scalar_binary_operation(
            scalar, 
            [](T x, T s) { return x + s; },       // forward:  input + scalar
            [](T s) { return T{ 1 }; }            // backward: d(x+s)/dx = 1
        );
    }

    auto operator-(const T scalar) const -> Tensor<T> {
        return scalar_binary_operation(
            scalar, 
            [](T x, T s) { return x - s; },       // forward:  input - scalar
            [](T s) { return T{ 1 }; }            // backward: d(x-s)/dx = 1
        );
    }

    auto operator*(const T scalar) const -> Tensor<T> {
        return scalar_binary_operation(
            scalar, 
            [](T x, T s) { return x * s; },       // forward:  input * scalar
            [](T s) { return s; }                 // backward: d(x*s)/dx = s
        );
    }

    auto operator/(const T scalar) const -> Tensor<T> {
        return scalar_binary_operation(
            scalar, 
            [](T x, T s) { return x / s; },       // forward:  input / scalar
            [](T s) { return T{ 1 } / s; }        // backward: d(x/s)/dx = 1 / s
        );
    }

    auto operator+=(const T scalar) -> Tensor<T>& {
        return scalar_binary_operation_in_place(
            scalar,
            [](T x, T s) { return x + s; }        // forward:  input + scalar
        );        
    }

    auto operator-=(const T scalar) -> Tensor<T>& {
        return scalar_binary_operation_in_place(
            scalar,
            [](T x, T s) { return x - s; }        // forward:  input - scalar
        );         
    }

    auto operator*=(const T scalar) -> Tensor<T>& {
        return scalar_binary_operation_in_place(
            scalar,
            [](T x, T s) { return x * s; }        // forward:  input * scalar
        ); 
    }

    auto operator/=(const T scalar) -> Tensor<T>& {
        return scalar_binary_operation_in_place(
            scalar,
            [](T x, T s) { return x / s; }        // forward:  input / scalar
        );        
    }

    auto operator+(const Tensor<T>& other) const -> Tensor<T> {
        return broadcast_binary_operation(
            other,
            [](T a, T b) { return a + b; },
            [](T, T, T g) { return std::pair{ g, g }; }
        );
    }

    auto operator-(const Tensor<T>& other) const -> Tensor<T> {
        return broadcast_binary_operation(
            other,
            [](T a, T b) { return a - b; },
            [](T, T, T g) { return std::pair{ g, -g }; }
        );
    }

    auto operator*(const Tensor<T>& other) const -> Tensor<T> {
        return broadcast_binary_operation(
            other,
            [](T a, T b) { return a * b; },
            [](T a, T b, T g) { return std::pair{ b * g, a * g }; }            // d(a*b)/da = b and d(a*b)/db = a
        );
    }

    auto operator/(const Tensor<T>& other) const -> Tensor<T> {
        return broadcast_binary_operation(
            other,
            [](T a, T b) { return a / b; },
            [](T a, T b, T g) { return std::pair{ g / b, -g * a / (b * b) }; } // d(a/b)/da= 1/b and d(a/b)/db= -a/b^2
        );
    }

    auto operator+=(const Tensor<T>& other) -> Tensor<T>& {
        return broadcast_binary_operation_in_place(
            other,
            [](T a, T b) { return a + b; }
        );
    }

    auto operator-=(const Tensor<T>& other) -> Tensor<T>& {
        return broadcast_binary_operation_in_place(
            other,
            [](T a, T b) { return a - b; }
        );
    }

    auto operator*=(const Tensor<T>& other) -> Tensor<T>& {
        return broadcast_binary_operation_in_place(
            other,
            [](T a, T b) { return a * b; }
        );
    }

    auto operator/=(const Tensor<T>& other) -> Tensor<T>& {
        return broadcast_binary_operation_in_place(
            other,
            [](T a, T b) { return a / b; }
        );
    }

    [[nodiscard]] auto sum(std::optional<std::size_t> axis = std::nullopt, bool keepdim = false) const -> Tensor<T> {
        auto forward_fn = [](T acc, T val) { return acc + val; };
        auto backward_fn = [](T, T, T grad_out) { return grad_out; };         // d(sum)/d(x_i) = 1

        if (!axis.has_value()) {
            return reduce_operation_total(forward_fn, backward_fn, T{ 0 });
        }

        return reduce_operation(axis.value(), keepdim, forward_fn, backward_fn, T{ 0 });
    }

    [[nodiscard]] auto prod(std::optional<std::size_t> axis = std::nullopt, bool keepdim = false) const -> Tensor<T> {
        auto forward_fn = [](T acc, T val) { return acc * val; };
        auto backward_fn = [](T input, T result, T grad_out) { // d(prod)/d(x_i) = prod / x_i if x_i != 0
            if (input == T{ 0 }) return T{ 0 }; // undefined ?! TODO: how to fix?
            return (result / input) * grad_out;
        };         

        if (!axis.has_value()) {
            return reduce_operation_total(forward_fn, backward_fn, T{ 1 });
        }

        return reduce_operation(axis.value(), keepdim, forward_fn, backward_fn, T{ 1 });
    }

    [[nodiscard]] auto min(std::optional<std::size_t> axis = std::nullopt, bool keepdim = false) const -> Tensor<T> {
        auto forward_fn = [](T acc, T val) { return std::min(acc, val); };
        auto backward_fn = [](T, T, T) { // d(min)/d(x_i) = 1 if x_i == min, else 0 ??
            return T{ 0 }; // undefined ?! TODO:min/max not differentiable?
        };         

        if (!axis.has_value()) {
            return reduce_operation_total(forward_fn, backward_fn, std::numeric_limits<T>::max());
        }

        return reduce_operation(axis.value(), keepdim, forward_fn, backward_fn, std::numeric_limits<T>::max());
    }

    [[nodiscard]] auto max(std::optional<std::size_t> axis = std::nullopt, bool keepdim = false) const -> Tensor<T> {
        auto forward_fn = [](T acc, T val) { return std::max(acc, val); };
        auto backward_fn = [](T, T, T) { // d(max)/d(x_i) = 1 if x_i == max, else 0 ??
            return T{ 0 }; // undefined ?! TODO:min/max not differentiable?
        };         

        if (!axis.has_value()) {
            return reduce_operation_total(forward_fn, backward_fn, std::numeric_limits<T>::lowest());
        }

        return reduce_operation(axis.value(), keepdim, forward_fn, backward_fn, std::numeric_limits<T>::lowest());
    }

    [[nodiscard]] auto mean(std::optional<std::size_t> axis = std::nullopt, bool keepdim = false) const -> Tensor<T>
    requires std::is_floating_point_v<T>
    {
        auto s = sum(axis, keepdim);
        auto count = axis.has_value() ? shape_[axis.value()] : size();

        return s / static_cast<T>(count);
    }

    [[nodiscard]] auto exp() const -> Tensor<T> {
        return unary_operation(
            [](T x) { return std::exp(x); },                        // forward
            [](T, T y) { return y; }                              // backward: de^x/dx = e^x = y
        );
    }

    [[nodiscard]] auto log() const -> Tensor<T> {
        return unary_operation(
            [](T x) { return std::log(x); },                        // forward
            [](T x, T) { return T{ 1 } / x; }                     // backward: dlog(x)/dx = 1/x
        );
    }

    [[nodiscard]] auto sqrt() const -> Tensor<T> {
        return unary_operation(
            [](T x) { return std::sqrt(x); },                       // forward
            [](T, T y) { return T{ 1 } / (T{ 2 } * y); }          // backward: dsqrt(x)/dx = 1/(2sqrt(x)) = 1/(2y)
        );
    }

    [[nodiscard]] auto tanh() const -> Tensor<T> {
        return unary_operation(
            [](T x) { return std::tanh(x); },                       // forward
            [](T, T y) { return T{ 1 } - y * y; }                 // backward: dtanh(x)/dx = 1 - tanh^2(x) = 1 - y^2
        );
    }

    [[nodiscard]] auto sigmoid() const -> Tensor<T> {
        return unary_operation(
            [](T x) { return T{ 1 } / (T{ 1 } + std::exp(-x)); },   // forward
            [](T, T y) { return y * (T{ 1 } - y); }               // backward: dsigmoid(x)/dx = sigmoid(x) * (1 - sigmoid(x)) = y * (1 - y)
        );
    }

    [[nodiscard]] auto relu() const -> Tensor<T> {
        return unary_operation(
            [](T x) { return x > T{ 0 } ? x : T{ 0 }; },            // forward
            [](T x, T) { return x > T{ 0 } ? T{ 1 } : T{ 0 }; }   // backward: 1 if x > 0, else 0
        );
    }

    [[nodiscard]] auto neg() const -> Tensor<T> {
        return unary_operation(
            [](T x) { return -x; },                                 // forward
            [](T, T) { return T{ -1 }; }                        // backward: d-x/dx = -1
        );
    }

    auto transpose() const -> Tensor<T> {
        if (shape_.size() == 1) return *this;
        else if (shape_.size() == 2) return permute(0, 1);

        throw std::runtime_error(
            std::format(
                "[Tensor::transpose()] Transpose does not work for tensors with dimensions {}, use permute() instead",
                shape_.size()
            )
        );
            
    }

    auto permute(std::size_t dim0, std::size_t dim1) const -> Tensor<T> {
        if (dim0 >= shape_.size() || dim1 >= shape_.size()) {
            throw std::out_of_range(
                std::format(
                    "[Tensor::permute()] Given dimensions are out of range for shape of size {}: dim0 is {}, dim1 is {}",
                    shape_.size(), dim0, dim1
                )
            );
        }

        if (dim0 == dim1) return *this;

        auto result_shape = shape_;
        auto result_strides = strides_;

        std::swap(result_shape[dim0], result_shape[dim1]);
        std::swap(result_strides[dim0], result_strides[dim1]);

        return Tensor<T>(
            impl_,
            result_shape,
            result_strides,
            impl_data_offset_,
            create_view_grad_node()
        );
    }

    [[nodiscard]] auto matmul(const Tensor<T>& other) const -> Tensor<T> {
        if (shape().size() == 1 && other.shape().size() == 1) {
            return dot(other);

        } else if (shape().size() == 2 && other.shape().size() == 2) {
            return matmul2D(other);

        } else {
            throw std::runtime_error("not yet implemented!");
        }
    }

private: // Helper

    [[nodiscard]] static auto compute_size(const std::vector<std::size_t>& shape) -> std::size_t {
        if (shape.empty()) return 0;

        return std::accumulate(shape.begin(), shape.end(), std::size_t{ 1 }, std::multiplies<>());
    }

    [[nodiscard]] static auto compute_strides(const std::vector<std::size_t>& shape) -> std::vector<std::size_t> {
        if (shape.empty()) return {};
                
        auto strides = std::vector<std::size_t>(shape.size());

        auto current_stride = std::size_t{ 1 };
        for (auto i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i) {
            strides[static_cast<std::size_t>(i)] = current_stride;
            current_stride *= shape[static_cast<std::size_t>(i)];
        }

        return strides;
    }

    [[nodiscard]] auto is_2d() const noexcept -> bool {
        return shape_.size() == 2;
    }

    [[nodiscard]] auto is_contiguous() const noexcept -> bool {
        return strides_ == compute_strides(shape_) && impl_data_offset_ == 0;
    }

    // faster access for 2D, no vector/heap allocation
    [[nodiscard]] static auto compute_2d_physical_idx(
        std::size_t flat_idx,
        std::size_t logical_stride_0,
        const std::vector<std::size_t>& physical_strides,
        std::size_t offset
    ) -> std::size_t {
        auto row = flat_idx / logical_stride_0;
        auto col = flat_idx % logical_stride_0;
        return row * physical_strides[0] + col * physical_strides[1] + offset;
    }

    [[nodiscard]] auto compute_physical_data_index(
        std::size_t logical_idx, 
        const std::vector<std::size_t>& logical_strides
    ) const -> std::size_t {
        auto indices = flat_to_ndim_idx(logical_idx, logical_strides);
        return ndim_to_flat_idx(indices, strides_) + impl_data_offset_;
    }

    [[nodiscard]] static auto flat_to_ndim_idx(
        const std::size_t flat_idx,
        const std::vector<std::size_t>& strides
    ) -> std::vector<std::size_t> {
        // compute ndim indices from flat indices, e.g. (row 1, col 1) of 2x3 tensor from flat idx 4
        auto indices = std::vector<std::size_t>(strides.size());

        auto remaining = flat_idx;

        for (std::size_t i = 0; i < strides.size(); ++i) {
            indices[i] = remaining / strides[i]; // retrieve integer value
            remaining %= strides[i];
        }

        return indices;
    }

    [[nodiscard]] static auto ndim_to_flat_idx(
        const std::vector<std::size_t>& indices,
        const std::vector<std::size_t>& strides
    ) -> std::size_t
    {
        auto flat_idx = std::size_t{ 0 };

        for (std::size_t i = 0; i < indices.size(); ++i) {
            flat_idx += indices[i] * strides[i];
        }

        return flat_idx;
    }

    [[nodiscard]] static auto collapse_shape(
        const std::vector<std::size_t>& shape,
        std::size_t axis_idx,
        bool keepdim
    ) -> std::vector<std::size_t> {
        auto result = std::vector<std::size_t>{};

        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i != axis_idx) {
                result.push_back(shape[i]);
            }
            else {
                if (keepdim) result.push_back(std::size_t{ 1 });
            }
        }

        // if result shape is empty, this is a 1D tensor, need to add shape { 1 }
        if (result.empty()) result.push_back(std::size_t{ 1 });

        return result;
    }

    [[nodiscard]] static auto collapse_indices(
        const std::vector<std::size_t>& indices,
        std::size_t axis_idx,
        bool keepdim
    ) -> std::vector<std::size_t> {
        auto result = std::vector<std::size_t>{};

        for (std::size_t d = 0; d < indices.size(); ++d) {
            if (d != axis_idx) {
                result.push_back(indices[d]);
            }
            else {
                if (keepdim) result.push_back(std::size_t{ 0 });
            }
        }

        if (result.empty()) result.push_back(std::size_t{ 0 });

        return result;
    }

    template<typename Fn>
    auto for_each_tensor_element(Fn&& fn) const -> void {
        auto logical_strides = compute_strides(shape_);

        for (std::size_t i = 0; i < size(); ++i) {
            auto data_idx = compute_physical_data_index(i, logical_strides);

            const auto& value = impl_->data[data_idx]; // pass const reference
            fn(value);
        }
    }

    template<typename Fn>
    auto for_each_tensor_element_mut(Fn&& fn) -> void {
        auto logical_strides = compute_strides(shape_);

        for (std::size_t i = 0; i < size(); ++i) {
            auto data_idx = compute_physical_data_index(i, logical_strides);

            auto& value = impl_->data[data_idx]; // pass non-const reference
            fn(value);
        }
    }
     
    template<typename Fn>
    auto for_each_tensor_element_collapse_axis(Fn&& fn) const -> void { 
        auto logical_strides = compute_strides(shape_);

        for (std::size_t i = 0; i < size(); ++i) {
            auto indices = flat_to_ndim_idx(i, logical_strides);
            auto data_idx = ndim_to_flat_idx(indices, strides_) + impl_data_offset_;

            const auto& value = impl_->data[data_idx]; // pass const reference
            fn(indices, data_idx, value);
        }
    }

    template<typename Fn>
    auto for_each_tensor_element_collapse_axis_mut(Fn&& fn) -> void {
        auto logical_strides = compute_strides(shape_);

        for (std::size_t i = 0; i < size(); ++i) {
            auto indices = flat_to_ndim_idx(i, logical_strides);
            auto data_idx = ndim_to_flat_idx(indices, strides_) + impl_data_offset_;

            auto& value = impl_->data[data_idx];  // pass non-const reference
            fn(indices, data_idx, value);
        }        
    }

    [[nodiscard]] auto create_view_grad_node() const -> std::shared_ptr<internal::GradNode> {
        if (!requires_grad()) return nullptr;

        auto result_parents = std::vector<std::shared_ptr<internal::GradNode>>{};
        if (grad_node_) result_parents.push_back(grad_node_);

        // no gradient computation needed, they simply flow through since no computation done in views
        return std::make_shared<internal::GradNode>(
            std::move(result_parents),
            []() -> void {}
        );
    }

    [[nodiscard]] static auto get_broadcast_shape(
        const std::vector<std::size_t>& shape_a_in,
        const std::vector<std::size_t>& shape_b_in
    ) -> std::vector<std::size_t>
    {
        auto shape_a = shape_a_in;
        auto shape_b = shape_b_in;

        // pad with 1s at the beginning until same shape
        while (shape_a.size() < shape_b.size()) shape_a.insert(shape_a.begin(), 1);
        while (shape_b.size() < shape_a.size()) shape_b.insert(shape_b.begin(), 1);

        auto result = std::vector<std::size_t>(shape_a.size());

        // check each shape pair of input
        for (std::size_t i = 0; i < shape_a.size(); ++i) {
            // rules for successfull broadcasting:
            //  - same dim
            //  - either one a 1
            if (shape_a[i] != shape_b[i] && shape_a[i] != 1 && shape_b[i] != 1) {
                throw std::invalid_argument(
                    std::format(
                        "[Tensor::get_broadcast_shape()] Shapes are not broadcastable: shape_a[i] is {}, shape_b[i] is {}",
                        shape_a[i], shape_b[i]
                    )
                );
            }

            result[i] = std::max(shape_a[i], shape_b[i]);
        }

        return result;        
    }

private: // Concrete implementations of operations

    template<typename ForwardFn, typename BackwardFn>
    [[nodiscard]] auto scalar_binary_operation(
        const T scalar,
        ForwardFn&& forward_fn,   // function returning scalar operation
        BackwardFn&& backward_fn  // function returning local gradient
    ) const -> Tensor<T> {
        auto result_size = size();
        auto result_impl = std::make_shared<internal::TensorImpl<T>>(
            size(), T{ 0 }, requires_grad()
        );

        // forward pass
        // now copy over all elements from this tensor (or view of tensor), i.e. do not copy complete underlying data vector
        auto logical_strides = compute_strides(shape_);

        if (is_2d() && is_contiguous()) { // direct, non-strided access

            #pragma omp parallel for
            for (std::size_t i = 0; i < result_size; ++i) {
                result_impl->data[i] = forward_fn(impl_->data[i], scalar);
            }

        } else if (is_2d()) { // strided 2d access

            #pragma omp parallel for
            for (std::size_t i = 0; i < result_size; ++i) {
                auto data_idx = compute_2d_physical_idx(i, shape_[1], strides_, impl_data_offset_);
                result_impl->data[i] = forward_fn(impl_->data[data_idx], scalar);
            }

        } else { // default strided nd access
            
            #pragma omp parallel for
            for (std::size_t i = 0; i < result_size; ++i) {
                auto data_idx = compute_physical_data_index(i, logical_strides);
                result_impl->data[i] = forward_fn(impl_->data[data_idx], scalar);
            }

        }

        auto result_grad_node = std::shared_ptr<internal::GradNode>{ nullptr };
        if (requires_grad()) {
            auto result_parents = std::vector<std::shared_ptr<internal::GradNode>>{};

            // only one parent since other is a scalar
            if (grad_node_) result_parents.push_back(grad_node_);


            auto result_backward_fn = [
                result_impl,
                input_impl = impl_,
                input_shape = shape_,
                input_strides = strides_,
                input_offset = impl_data_offset_,
                scalar,
                backward_fn
            ]() -> void {
                if (!input_impl->grad) return;

                auto n = compute_size(input_shape);
                auto logical_strides = compute_strides(input_shape);

                //#pragma omp parallel for
                for (std::size_t i = 0; i < n; ++i) {
                    auto indices = flat_to_ndim_idx(i, logical_strides);
                    auto data_idx = ndim_to_flat_idx(indices, input_strides) + input_offset;

                    auto grad_out = (*result_impl->grad)[i];

                    (*input_impl->grad)[data_idx] += grad_out * backward_fn(scalar);
                }
            };

            result_grad_node = std::make_shared<internal::GradNode>(
                std::move(result_parents),
                std::move(result_backward_fn)
            );
        }

        return Tensor<T>(result_impl, shape_, logical_strides, std::size_t{ 0 }, std::move(result_grad_node));
    }

    template<typename ForwardFn>
    [[nodiscard]] auto scalar_binary_operation_in_place(const T scalar, ForwardFn&& forward_fn) -> Tensor<T>& {
        if (requires_grad()) {
            throw std::runtime_error(
                std::format(
                    "[Tensor::scalar_binary_operation_in_place()] In-place operators only work on tensors that do not "
                    "require gradients, this is because the computational graph would break because of data loss."
                )
            ); 
        }

        for_each_tensor_element_mut([&](T& value) { value = forward_fn(value, scalar); });
            
        return *this;
    }

    template<typename ForwardFn, typename BackwardFn>
    [[nodiscard]] auto broadcast_binary_operation(
        const Tensor<T>& other,
        ForwardFn&& forward_fn,
        BackwardFn&& backward_fn
    ) const -> Tensor<T> {
        // TODO:
        //  Need to decide whether forcing contigous layout of input tensors is worth it.
        //  For now use strided access using the accessor methods / offsets.
        //  Trade-off: strides access incurs more index calculation operations while
        //             forcing contigous layout incurs copying tensors.

        auto result_shape = get_broadcast_shape(shape_, other.shape_);
        auto result_strides = compute_strides(result_shape);
        auto result_size = compute_size(result_shape);
        auto result_requires_grad = requires_grad() || other.requires_grad();

        // Pad shapes with leading 1s and strides with leading 0s
        auto shape_a = shape_;
        auto shape_b = other.shape_;
        auto strides_a = strides_;
        auto strides_b = other.strides_;

        while (shape_a.size() < result_shape.size()) {
            shape_a.insert(shape_a.begin(), 1);
            strides_a.insert(strides_a.begin(), 0);
        }

        while (shape_b.size() < result_shape.size()) {
            shape_b.insert(shape_b.begin(), 1);
            strides_b.insert(strides_b.begin(), 0);
        }

        // zero strides for broadcast dim
        for (std::size_t i = 0; i < result_shape.size(); ++i) {
            if (shape_a[i] == 1) strides_a[i] = 0;
            if (shape_b[i] == 1) strides_b[i] = 0;
        }

        // create new tensor and allocate memory
        auto result_impl = std::make_shared<internal::TensorImpl<T>>(
            result_size, T{ 0 }, result_requires_grad
        );

        // forward pass
        auto both_2d = is_2d() && other.is_2d();
        auto both_contigous = is_contiguous() && other.is_contiguous();
        auto both_same_shape = shape_ == other.shape_;

        if (both_contigous && both_same_shape) { // direct, non-strided access

            #pragma omp parallel for
            for (std::size_t i = 0; i < result_size; ++i) {
                result_impl->data[i] = forward_fn(impl_->data[i], other.impl_->data[i]);
            }

        } else if (both_2d) { // strided 2d access

            #pragma omp parallel for collapse(2)
            for (std::size_t r = 0; r < result_shape[0]; ++r) {
                for (std::size_t c = 0; c < result_shape[1]; ++c) {
                    auto data_idx_a = r * strides_a[0] + c * strides_a[1] + impl_data_offset_;
                    auto data_idx_b = r * strides_b[0] + c * strides_b[1] + other.impl_data_offset_;
                    auto result_idx = r * result_shape[1] + c;
                    
                    result_impl->data[result_idx] = forward_fn(impl_->data[data_idx_a], other.impl_->data[data_idx_b]);
                }
            }

        } else { // default strided nd access

            #pragma omp parallel for
            for (std::size_t i = 0; i < result_size; ++i) {
                auto result_indices = flat_to_ndim_idx(i, result_strides);

                auto data_idx_a = ndim_to_flat_idx(result_indices, strides_a) + impl_data_offset_;
                auto data_idx_b = ndim_to_flat_idx(result_indices, strides_b) + other.impl_data_offset_;

                result_impl->data[i] = forward_fn(impl_->data[data_idx_a], other.impl_->data[data_idx_b]);
            }

        }

        auto result_grad_node = std::shared_ptr<internal::GradNode>{ nullptr };
        if (result_requires_grad) {
            auto result_parents = std::vector<std::shared_ptr<internal::GradNode>>{};

            // add both parents to new grad node
            if (grad_node_) result_parents.push_back(grad_node_);
            if (other.grad_node_) result_parents.push_back(other.grad_node_);

            auto result_backward_fn = [
                result_impl, result_shape, result_strides,
                impl_a = impl_, shape_a, strides_a, offset_a = impl_data_offset_,
                impl_b = other.impl_, shape_b, strides_b, offset_b = other.impl_data_offset_,
                backward_fn
            ]() -> void {
                auto n = compute_size(result_shape);
                
                for (std::size_t i = 0; i < n; ++i) {
                    auto result_indices = flat_to_ndim_idx(i, result_strides);

                    auto data_idx_a = ndim_to_flat_idx(result_indices, strides_a) + offset_a;
                    auto data_idx_b = ndim_to_flat_idx(result_indices, strides_b) + offset_b;

                    auto grad_out = (*result_impl->grad)[i];
                    auto val_a = impl_a->data[data_idx_a];
                    auto val_b = impl_b->data[data_idx_b];

                    auto [grad_a, grad_b] = backward_fn(val_a, val_b, grad_out);

                    if (impl_a->grad) (*impl_a->grad)[data_idx_a] += grad_a;
                    if (impl_b->grad) (*impl_b->grad)[data_idx_b] += grad_b;
                }
            };

            result_grad_node = std::make_shared<internal::GradNode>(
                std::move(result_parents),
                std::move(result_backward_fn)
            );
        }

        return Tensor<T>(
            result_impl,
            result_shape,
            result_strides,
            std::size_t{ 0 },
            std::move(result_grad_node)
        );        
    }

    template<typename ForwardFn>
    [[nodiscard]] auto broadcast_binary_operation_in_place(const Tensor<T>& other, ForwardFn&& forward_fn) -> Tensor<T>& {
        if (requires_grad()) {
            throw std::runtime_error(
                std::format(
                    "[Tensor::broadcast_binary_operation_in_place()] In-place operators only work on tensors that do not "
                    "require gradients, this is because the computational graph would break because of data loss."
                )
            ); 
        }

        if (shape_ != get_broadcast_shape(shape_, other.shape_)) {
            throw std::runtime_error(
                std::format(
                    "[Tensor::broadcast_binary_operation_in_place()] In-place operators only work when "
                    "broadcasting returns the same shape as the tensor it is performed on."
                )
            );             
        }

        // Pad shapes of other with leading 1s and strides with leading 0s
        auto shape_b = other.shape_;
        auto strides_b = other.strides_;

        while (shape_b.size() < shape_.size()) {
            shape_b.insert(shape_b.begin(), 1);
            strides_b.insert(strides_b.begin(), 0);
        }

        // zero strides for broadcast dim
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            if (shape_b[i] == 1) strides_b[i] = 0;
        }

        // compute result data with actual arithmetic operation
        const auto logical_strides = compute_strides(shape_);

        for (std::size_t i = 0; i < size(); ++i) {
            auto indices = flat_to_ndim_idx(i, logical_strides);
            auto data_idx_a = ndim_to_flat_idx(indices, strides_) + impl_data_offset_;
            auto data_idx_b = ndim_to_flat_idx(indices, strides_b) + other.impl_data_offset_;

            impl_->data[data_idx_a] = forward_fn(impl_->data[data_idx_a], other.impl_->data[data_idx_b]);
        }

        return *this;
    }

    template<typename ForwardFn, typename BackwardFn>
    [[nodiscard]] auto reduce_operation_total(
        ForwardFn&& forward_fn,
        BackwardFn&& backward_fn,
        T init_value
    ) const -> Tensor<T> {

        auto total = init_value;
        for_each_tensor_element([&total, &forward_fn](const T& value) { 
            total = forward_fn(total, value);
        });

        auto result = Tensor<T>::from({ total }, { 1 }, requires_grad());

        if (requires_grad()) {
            auto result_parents = std::vector<std::shared_ptr<internal::GradNode>>{};
            if (grad_node_) result_parents.push_back(grad_node_);
            
            auto result_backward_fn = [
                result_impl = result.impl_,
                input_impl = impl_,
                input_shape = shape_,
                input_strides = strides_,
                input_offset = impl_data_offset_,
                backward_fn,
                total
            ]() -> void {
                if (!input_impl->grad) return;
                
                auto grad_out = (*result_impl->grad)[0];
                auto input_size = compute_size(input_shape);
                auto logical_strides = compute_strides(input_shape);
                
                for (std::size_t i = 0; i < input_size; ++i) {
                    auto indices = flat_to_ndim_idx(i, logical_strides);
                    auto input_data_idx = ndim_to_flat_idx(indices, input_strides) + input_offset;
                    auto input_value = input_impl->data[input_data_idx];

                    auto grad_in = backward_fn(input_value, total, grad_out);

                    (*input_impl->grad)[input_data_idx] += grad_in;
                }
            };
            
            result.grad_node_ = std::make_shared<internal::GradNode>(
                std::move(result_parents),
                std::move(result_backward_fn)
            );
        }
        
        return result;
    }

    template<typename ForwardFn, typename BackwardFn>
    [[nodiscard]] auto reduce_operation(
        std::size_t axis,
        bool keepdim,
        ForwardFn&& forward_fn,
        BackwardFn&& backward_fn,
        T init_value
    ) const -> Tensor<T> {

        if (axis >= shape_.size()) {
            throw std::out_of_range(
                std::format(
                    "[Tensor::reduce_operation()] Given axis idx ({}) is out of range for shape of size {}",
                    axis, shape_.size()
                )
            );
        }

        // Create new result shape which has the given axis idx removed / collapsed
        //  If keepdim is true, then keep that axis with shape 1
        auto result_shape = collapse_shape(shape_, axis, keepdim);

        auto result = Tensor<T>::full(result_shape, init_value, requires_grad());

        for_each_tensor_element_collapse_axis(
            [&](const auto& indices, [[maybe_unused]] std::size_t data_idx, const T& value) {
                auto result_indices = collapse_indices(indices, axis, keepdim);
                auto result_flat_idx = ndim_to_flat_idx(result_indices, result.strides());

                auto& result_value = result.data_mut()[result_flat_idx + result.offset()];
                result_value = forward_fn(result_value, value);            
            }
        );

        auto result_grad_node = std::shared_ptr<internal::GradNode>{ nullptr };
        if (requires_grad()) {
            auto result_parents = std::vector<std::shared_ptr<internal::GradNode>>{};
            if (grad_node_) result_parents.push_back(grad_node_);

            auto result_backward_fn = [
                result_impl = result.impl_, result_shape, result_strides = result.strides(), result_offset = result.offset(),
                input_impl = impl_, input_shape = shape_, input_strides = strides_, input_offset = impl_data_offset_,
                axis, keepdim, backward_fn
            ]() -> void {
                if (!input_impl->grad) return;

                auto input_size = compute_size(input_shape);
                auto logical_strides = compute_strides(input_shape);

                for (std::size_t i = 0; i < input_size; ++i) {
                    auto indices = flat_to_ndim_idx(i, logical_strides);
                    auto input_data_idx = ndim_to_flat_idx(indices, input_strides) + input_offset;
                    auto input_value = input_impl->data[input_data_idx];

                    auto result_indices = collapse_indices(indices, axis, keepdim);
                    auto result_data_idx = ndim_to_flat_idx(result_indices, result_strides) + result_offset;
                    auto result_value = result_impl->data[result_data_idx];

                    auto grad_out = (*result_impl->grad)[result_data_idx];

                    auto grad_in = backward_fn(input_value, result_value, grad_out);

                    (*input_impl->grad)[input_data_idx] += grad_in;
                }   
            };

            result_grad_node = std::make_shared<internal::GradNode>(
                std::move(result_parents),
                std::move(result_backward_fn)
            );

            result.grad_node_ = result_grad_node;
        }

        return result;
    }

    [[nodiscard]] auto dot(const Tensor<T>& other) const -> Tensor<T> {
        if (shape_.size() != 1 || other.shape_.size() != 1) {
            throw std::invalid_argument(
                std::format(
                    "[Tensor::dot()] works only on 1D Tensors: tensor a is size {}, tensor b is size {}",
                    shape_.size(), other.shape_.size()
                )
            );           
        }

        if (size() != other.size()) {
            throw std::invalid_argument(
                std::format(
                    "[Tensor::dot()] tensors must have the same (data) size: tensor a is size {}, tensor b is size {}",
                    size(), other.size()
                )
            );
        }

        auto result_requires_grad = requires_grad() || other.requires_grad();
        auto result_shape = std::vector<std::size_t>{ 1 };
        auto result_strides = std::vector<std::size_t>{ 1 };
        auto result_size = size();
        
        auto result = T{ 0 };
        #pragma omp parallel for
        for (std::size_t i = 0; i < result_size; ++i) {
            auto data_idx_a = i * strides_[0] + impl_data_offset_;
            auto data_idx_b = i * other.strides_[0] + other.impl_data_offset_;

            result += impl_->data[data_idx_a] * other.impl_->data[data_idx_b];
        }

        auto result_impl = std::make_shared<internal::TensorImpl<T>>(
            std::vector<T>{ result }, result_requires_grad
        );
        
        auto result_grad_node = std::shared_ptr<internal::GradNode>{ nullptr };
        if (result_requires_grad) {
            auto result_parents = std::vector<std::shared_ptr<internal::GradNode>>{};
            if (grad_node_) result_parents.push_back(grad_node_);
            if (other.grad_node_) result_parents.push_back(other.grad_node_);

            auto result_backward_fn = [
                result_impl,
                result_size,
                
                impl_a = impl_,
                strides_a = strides_,
                offset_a = impl_data_offset_,
                
                impl_b = other.impl_,
                strides_b = other.strides_,
                offset_b = other.impl_data_offset_
            ]() -> void {
                // C = A dot B
                //  dL/dA = dL/dC dot B
                //  dL/dB = dL/dC dot A

                auto grad_out = (*result_impl->grad)[0];
                for (std::size_t i = 0; i < result_size; ++i) {
                    auto data_idx_a = i * strides_a[0] + offset_a;
                    auto data_idx_b = i * strides_b[0] + offset_b;

                    if (impl_a->grad) {
                        (*impl_a->grad)[data_idx_a] += grad_out * impl_b->data[data_idx_b];
                    }

                    if (impl_b->grad) {
                        (*impl_b->grad)[data_idx_b] += grad_out * impl_a->data[data_idx_a];
                    }                   
                }
            };

            result_grad_node = std::make_shared<internal::GradNode>(
                std::move(result_parents),
                std::move(result_backward_fn)
            );

        }

        return Tensor<T>(
            result_impl,
            result_shape,
            result_strides,
            std::size_t{ 0 },
            std::move(result_grad_node)
        );
    }

    [[nodiscard]] auto matmul2D(const Tensor<T>& other) const -> Tensor<T> {
        if (shape_.size() != 2 || other.shape_.size() != 2) {
            throw std::invalid_argument(
                std::format(
                    "[Tensor::matmul2D()] works only on 2D Tensors: tensor a is size {}, tensor b is size {}",
                    shape_.size(), other.shape_.size()
                )
            );
        }

        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument(
                std::format(
                    "[Tensor::matmul2D()] shape mismatch: first tensors second dimension does not match second tensors "
                    "first dimension. Tensor a shape[1] is {}, tensor b shape[0] is {}",
                    shape_[1], other.shape_[0]
                )
            );
        }

        // Matmul: Tensor 1 is AxB and tensor 2 is BxC, result tensor is AxC
        const auto A = shape_[0];
        const auto B = shape_[1];
        const auto C = other.shape_[1];

        auto result_requires_grad = requires_grad() || other.requires_grad();
        auto result_size = A * C;
        auto result_shape = std::vector<std::size_t>{ A, C };
        auto result_strides = compute_strides(result_shape);

        auto result_impl = std::make_shared<internal::TensorImpl<T>>(
            result_size, T{ 0 }, result_requires_grad
        );

        #pragma omp parallel for collapse(2)
        for (std::size_t i = 0; i < A; ++i) {
            for (std::size_t j = 0; j < C; ++j) {
                auto result_ij = T{ 0 };

                for (std::size_t k = 0; k < B; ++k) {
                    auto data_idx_a = i * strides_[0] + k * strides_[1] + impl_data_offset_;
                    auto data_idx_b = k * other.strides_[0] + j * other.strides_[1] + other.impl_data_offset_;

                    result_ij += impl_->data[data_idx_a] * other.impl_->data[data_idx_b];
                }

                auto data_idx_result = i * C + j; // result is contigous
                result_impl->data[data_idx_result] = result_ij;
            }
        }

        auto result_grad_node = std::shared_ptr<internal::GradNode>{ nullptr };
        if (result_requires_grad) {
            auto result_parents = std::vector<std::shared_ptr<internal::GradNode>>{};

            if (grad_node_) result_parents.push_back(grad_node_);
            if (other.grad_node_) result_parents.push_back(other.grad_node_);


            auto result_backward_fn = [
                result_impl,
                A, B, C,

                impl_a = impl_,
                strides_a = strides_,
                offset_a = impl_data_offset_,
                
                impl_b = other.impl_,
                strides_b = other.strides_,
                offset_b = other.impl_data_offset_
            ]() -> void {

                // gradient for matrix multiplication Z = X @ Y:
                //  dL/dX = dL/dZ @ Y.T
                //      -> shape AxB = shape AxC @ shape CxB
                //  dL/dY = X.T @ dL/dZ
                //      -> shape BxC = shape BxA @ shape AxC

                // dL/dX
                if (impl_a->grad) {
                    for (std::size_t i = 0; i < A; ++i) {
                        for (std::size_t j = 0; j < B; ++j) {
                            auto result_ij = T{ 0 };

                            for (std::size_t k = 0; k < C; ++k) {
                                auto grad_idx_z = i * C + k;
                                auto data_idx_b_T = j * strides_b[0] + k * strides_b[1] + offset_b;

                                result_ij += (*result_impl->grad)[grad_idx_z] * impl_b->data[data_idx_b_T];
                            }

                            auto grad_idx_a = i * strides_a[0] + j * strides_a[1] + offset_a;
                            (*impl_a->grad)[grad_idx_a] += result_ij;
                        }
                    }
                }

                // dL/dY
                if (impl_b->grad) {
                    for (std::size_t i = 0; i < B; ++i) {
                        for (std::size_t j = 0; j < C; ++j) {
                            auto result_ij = T{ 0 };

                            for (std::size_t k = 0; k < A; ++k) {
                                auto data_idx_a_T = k * strides_a[0] + i * strides_a[1] + offset_a;
                                auto grad_idx_z = k * C + j;

                                result_ij += impl_a->data[data_idx_a_T] * (*result_impl->grad)[grad_idx_z];
                            }

                            auto grad_idx_b = i * strides_b[0] + j * strides_b[1] + offset_b;
                            (*impl_b->grad)[grad_idx_b] += result_ij;
                        }
                    }
                }
            };

            result_grad_node = std::make_shared<internal::GradNode>(
                std::move(result_parents),
                std::move(result_backward_fn)
            );
        }

        return Tensor<T>(
            result_impl,
            result_shape,
            result_strides,
            std::size_t{ 0 },
            std::move(result_grad_node)
        );
    }
    
    template<typename ForwardFn, typename BackwardFn>
    [[nodiscard]] auto unary_operation(ForwardFn&& forward_fn, BackwardFn&& backward_fn) const -> Tensor<T> {
        auto result_size = size();
        auto result_requires_grad = requires_grad();

        auto result_impl = std::make_shared<internal::TensorImpl<T>>(
            result_size, T{ 0 }, result_requires_grad
        );

        auto result_shape = shape_;
        auto result_strides = compute_strides(result_shape);

        // forward pass
        if (is_2d() && is_contiguous()) { // direct, non-strided access

            #pragma omp parallel for
            for (std::size_t i = 0; i < result_size; ++i) {
                result_impl->data[i] = forward_fn(impl_->data[i]);
            }

        } else if (is_2d()) { // strided 2d access

            #pragma omp parallel for
            for (std::size_t i = 0; i < result_size; ++i) {
                auto data_idx = compute_2d_physical_idx(i, shape_[1], strides_, impl_data_offset_);
                result_impl->data[i] = forward_fn(impl_->data[data_idx]);
            }

        } else { // default strided nd access

            auto logical_strides = compute_strides(shape_);
            #pragma omp parallel for
            for (std::size_t i = 0; i < result_size; ++i) {
                auto data_idx = compute_physical_data_index(i, logical_strides);
                result_impl->data[i] = forward_fn(impl_->data[data_idx]);
            }

        }

        // backward pass
        auto result_grad_node = std::shared_ptr<internal::GradNode>{ nullptr };
        if (result_requires_grad) {
            auto result_parents = std::vector<std::shared_ptr<internal::GradNode>>{};

            if (grad_node_) result_parents.push_back(grad_node_);

            auto result_backward_fn = [
                result_impl,
                input_impl = impl_,
                input_strides = strides_,
                input_offset = impl_data_offset_,
                input_shape = shape_,
                backward_fn
            ]() -> void {
                if (!input_impl->grad) return;

                auto n = compute_size(input_shape);

                auto logical_strides = compute_strides(input_shape);
                for (std::size_t i = 0; i < n; ++i) {
                    auto indices = flat_to_ndim_idx(i, logical_strides);
                    auto input_data_idx = ndim_to_flat_idx(indices, input_strides) + input_offset;

                    auto grad_out = (*result_impl->grad)[i];
                    auto input_val = input_impl->data[input_data_idx];
                    auto output_val = result_impl->data[i];

                    // backward_fn(input_val, output_val) returns local gradient
                    (*input_impl->grad)[input_data_idx] += grad_out * backward_fn(input_val, output_val);
                }
            };

            result_grad_node = std::make_shared<internal::GradNode>(
                std::move(result_parents),
                std::move(result_backward_fn)
            );
        }

        return Tensor<T>(
            result_impl,
            result_shape,
            result_strides,
            std::size_t{ 0 },
            std::move(result_grad_node)
        );
    }


public: // Gradient computation

    auto backward(const Tensor<T>* gradient = nullptr) -> void {
        using internal::GradNode;

        auto topology = std::vector<std::shared_ptr<GradNode>>{};
        auto visited = std::unordered_set<GradNode*>{};

        auto build_topology = [&](auto&& self, std::shared_ptr<GradNode> gnp) -> void {
            if (!gnp) return;
            if (visited.find(gnp.get()) != visited.end()) return;

            visited.insert(gnp.get());
            for (const auto& parent: gnp->parents) {
                self(self, parent);
            }
            topology.push_back(gnp);
        };

        build_topology(build_topology, grad_node_);

        if (impl_->grad) {
            if (!gradient) { 
                if (size() != 1) {
                    throw std::runtime_error("Cannot implicitly create gradient for non-scalar tensors.");
                }

                (*impl_->grad)[0] = T{ 1 }; 
            } 
            else {
                if (gradient->shape() != shape()) {
                    throw std::invalid_argument("Provided gradient does not match shape.");
                }

                std::copy(gradient->data().begin(), gradient->data().end(), impl_->grad->begin());
            }
        }

        std::ranges::reverse(topology);
        for (const auto& gnp: topology) {
            if (gnp->backward_fn) {
                gnp->backward_fn();
            }
        }
    }

    auto zero_grad() -> void {
        impl_->zero_grad();
    }


public: // Printing

    auto print(bool with_data = true) -> void {
        std::cout << "Tensor:\n";
        std::cout << "\ttype = " << typeid(T).name() << "\n";
        std::cout << "\tshape = [ ";
        for (const auto& e: shape_) std::cout << e << " ";
        std::cout << "] (size = " << shape_.size() << ")\n";
        std::cout << "\tstrides = [ ";
        for (const auto& e: strides_) std::cout << e << " ";
        std::cout << "] (size = " << strides_.size() << ")\n";
        std::cout << "\timpl_data_offset_ = " << impl_data_offset_ << "\n";


        if (impl_) {
            std::cout << "\tTensorImpl:\n";

            if (with_data) {
                std::cout << "\t\tdata = [ ";
                for (const auto& e: impl_->data) std::cout << e << " ";
                std::cout << "] (size = " << impl_->data.size() << ")\n";
            } else {
                std::cout << "\t\tdata size = " << impl_->data.size() << "\n";
            }

            std::cout << "\t\trequires_grad = " << (impl_->requires_grad ? "true" : "false") << "\n";
            if (impl_->grad) {
                if (with_data) {
                    std::cout << "\t\tgrad = [ ";
                    for (const auto& e: *(impl_->grad)) std::cout << e << " ";
                    std::cout << "] (size = " << impl_->grad->size() << ")\n";
                } else {
                    std::cout << "\t\tgrad size = " << impl_->grad->size() << "\n";
                }
            }
        }

        if (grad_node_) {
            std::cout << "\tGradNode:\n";
            std::cout << "\t\tparents = [ ";
            for (const auto& e: grad_node_->parents) std::cout << e << " ";
            std::cout << "] (size = " << grad_node_->parents.size() << ")\n";
        }
    }
};


} // namespace makemore
