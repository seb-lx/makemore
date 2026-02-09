#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tensor.h"


using makemore::Tensor;
using sizeVec = std::vector<std::size_t>;


TEST_CASE("Tensor construction") {


    SUBCASE("factory zeros 1D") {
        auto t = Tensor<float>::zeros({ 4 }, true);

        CHECK(t.shape() == sizeVec{ 4 });
        CHECK(t.strides() == sizeVec{ 1 });
        CHECK(t.size() == std::size_t{ 4 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == true);
        for (std::size_t i = 0; i < 4; ++i) CHECK(t.at_flat(i) == doctest::Approx(0.0f));
    }

    SUBCASE("factory ones 1D") {
        auto t = Tensor<float>::ones({ 4 }, true);

        CHECK(t.shape() == sizeVec{ 4 });
        CHECK(t.strides() == sizeVec{ 1 });
        CHECK(t.size() == std::size_t{ 4 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == true);
        for (std::size_t i = 0; i < 4; ++i) CHECK(t.at_flat(i) == doctest::Approx(1.0f));
    }

    SUBCASE("factory randn float 1D") {
        auto gen = std::mt19937{ 2147483647 };

        auto t = Tensor<float>::randn({ 4 }, gen, true);

        CHECK(t.shape() == sizeVec{ 4 });
        CHECK(t.strides() == sizeVec{ 1 });
        CHECK(t.size() == std::size_t{ 4 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == true);
    }

    SUBCASE("factory randn integral 1D") {
        auto gen = std::mt19937{ 2147483647 };

        auto t = Tensor<int>::randn(0, 10, { 7 }, gen, false);

        CHECK(t.shape() == sizeVec{ 7 });
        CHECK(t.strides() == sizeVec{ 1 });
        CHECK(t.size() == std::size_t{ 7 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == false);
        for (std::size_t i = 0; i < 7; ++i) {
            auto& v = t.at_flat(i);
            CHECK(v >= 0); CHECK(v <= 10);
        }
    }

    SUBCASE("factory from 1D") {
        auto v = std::vector<float>{ 1, 2, 3, 4 };
        auto t1 = Tensor<float>::from(v, { 4 }, true);

        CHECK(t1.shape() == sizeVec{ 4 });
        CHECK(t1.strides() == sizeVec{ 1 });
        CHECK(t1.size() == std::size_t{ 4 });
        CHECK(t1.offset() == std::size_t{ 0 });
        CHECK(t1.requires_grad() == true);

        for (std::size_t i = 0; i < 4; ++i) CHECK(t1.at_flat(i) == doctest::Approx(static_cast<float>(i + 1)));


        auto t2 = Tensor<float>::from({ 1, 2, 3, 4 }, { 4 }, true);

        CHECK(t2.shape() == sizeVec{ 4 });
        CHECK(t2.strides() == sizeVec{ 1 });
        CHECK(t2.size() == std::size_t{ 4 });
        CHECK(t2.offset() == std::size_t{ 0 });
        CHECK(t2.requires_grad() == true);

        for (std::size_t i = 0; i < 4; ++i) CHECK(t2.at_flat(i) == doctest::Approx(static_cast<float>(i + 1)));


        auto t3 = Tensor<float>::from({ 1, 2, 3, 4 }, true);

        CHECK(t3.shape() == sizeVec{ 4 });
        CHECK(t3.strides() == sizeVec{ 1 });
        CHECK(t3.size() == std::size_t{ 4 });
        CHECK(t3.offset() == std::size_t{ 0 });
        CHECK(t3.requires_grad() == true);

        for (std::size_t i = 0; i < 4; ++i) CHECK(t3.at_flat(i) == doctest::Approx(static_cast<float>(i + 1)));

        
        auto t4 = Tensor<float>::from(t3, true);

        CHECK(t4.shape() == sizeVec{ 4 });
        CHECK(t4.strides() == sizeVec{ 1 });
        CHECK(t4.size() == std::size_t{ 4 });
        CHECK(t4.offset() == std::size_t{ 0 });
        CHECK(t4.requires_grad() == true);

        for (std::size_t i = 0; i < 4; ++i) CHECK(t4.at_flat(i) == doctest::Approx(static_cast<float>(i + 1)));
    }

    SUBCASE("factory zeros_like 1D") {
        auto gen = std::mt19937{ 2147483647 };

        auto t = Tensor<int>::randn(0, 10, { 7 }, gen, true);
        auto t2 = Tensor<int>::zeros_like(t, false);

        CHECK(t2.shape() == sizeVec{ 7 });
        CHECK(t2.strides() == sizeVec{ 1 });
        CHECK(t2.size() == std::size_t{ 7 });
        CHECK(t2.offset() == std::size_t{ 0 });
        CHECK(t2.requires_grad() == false);

        for (std::size_t i = 0; i < 7; ++i) CHECK(t2.at_flat(i) == 0);
    }

    SUBCASE("factory ones_like 1D") {
        auto gen = std::mt19937{ 2147483647 };

        auto t = Tensor<int>::randn(0, 10, { 7 }, gen, true);
        auto t2 = Tensor<int>::ones_like(t, false);

        CHECK(t2.shape() == sizeVec{ 7 });
        CHECK(t2.strides() == sizeVec{ 1 });
        CHECK(t2.size() == std::size_t{ 7 });
        CHECK(t2.offset() == std::size_t{ 0 });
        CHECK(t2.requires_grad() == false);

        for (std::size_t i = 0; i < 7; ++i) CHECK(t2.at_flat(i) == 1);
    }

    SUBCASE("factory zeros 2D") {
        auto t = Tensor<float>::zeros({ 2, 2 }, true);
        auto t1 = Tensor<float>::zeros({ 2, 1 }, true);
        auto t2 = Tensor<float>::zeros({ 1, 2 }, true);

        CHECK(t.shape() == sizeVec{ 2, 2 });
        CHECK(t.strides() == sizeVec{ 2, 1 });
        CHECK(t.size() == std::size_t{ 4 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == true);
        for (std::size_t i = 0; i < 4; ++i) CHECK(t.at_flat(i) == doctest::Approx(0.0f));

        CHECK(t1.shape() == sizeVec{ 2, 1 });
        CHECK(t1.strides() == sizeVec{ 1, 1 });
        CHECK(t1.size() == std::size_t{ 2 });
        CHECK(t1.offset() == std::size_t{ 0 });
        CHECK(t1.requires_grad() == true);
        for (std::size_t i = 0; i < 2; ++i) CHECK(t1.at_flat(i) == doctest::Approx(0.0f));

        CHECK(t2.shape() == sizeVec{ 1, 2 });
        CHECK(t2.strides() == sizeVec{ 2, 1 });
        CHECK(t2.size() == std::size_t{ 2 });
        CHECK(t2.offset() == std::size_t{ 0 });
        CHECK(t2.requires_grad() == true);
        for (std::size_t i = 0; i < 2; ++i) CHECK(t2.at_flat(i) == doctest::Approx(0.0f));
    }

    SUBCASE("factory ones 2D") {
        auto t = Tensor<float>::ones({ 4, 2 }, false);

        CHECK(t.shape() == sizeVec{ 4, 2 });
        CHECK(t.strides() == sizeVec{ 2, 1 });
        CHECK(t.size() == std::size_t{ 8 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == false);
        for (std::size_t i = 0; i < 8; ++i) CHECK(t.at_flat(i) == doctest::Approx(1.0f));
    }

    SUBCASE("factory randn float 2D") {
        auto gen = std::mt19937{ 2147483647 };

        auto t = Tensor<float>::randn({ 4, 4 }, gen, true);

        CHECK(t.shape() == sizeVec{ 4, 4 });
        CHECK(t.strides() == sizeVec{ 4, 1 });
        CHECK(t.size() == std::size_t{ 16 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == true);
    }

    SUBCASE("factory randn integral 2D") {
        auto gen = std::mt19937{ 2147483647 };

        auto t = Tensor<int>::randn(0, 10, { 7, 4 }, gen, false);

        CHECK(t.shape() == sizeVec{ 7, 4 });
        CHECK(t.strides() == sizeVec{ 4, 1 });
        CHECK(t.size() == std::size_t{ 28 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == false);
        for (std::size_t i = 0; i < 28; ++i) {
            auto& v = t.at_flat(i);
            CHECK(v >= 0); CHECK(v <= 10);
        }
    }

    SUBCASE("factory from 2D") {
        auto v = std::vector<float>{ 1, 2, 3, 4, 5, 6, 7, 8 };
        auto t1 = Tensor<float>::from(v, { 2, 4 }, true);

        CHECK(t1.shape() == sizeVec{ 2, 4 });
        CHECK(t1.strides() == sizeVec{ 4, 1 });
        CHECK(t1.size() == std::size_t{ 8 });
        CHECK(t1.offset() == std::size_t{ 0 });
        CHECK(t1.requires_grad() == true);

        for (std::size_t i = 0; i < 8; ++i) CHECK(t1.at_flat(i) == doctest::Approx(static_cast<float>(i + 1)));


        auto t2 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 4, 2 }, true);

        CHECK(t2.shape() == sizeVec{ 4, 2 });
        CHECK(t2.strides() == sizeVec{ 2, 1 });
        CHECK(t2.size() == std::size_t{ 8 });
        CHECK(t2.offset() == std::size_t{ 0 });
        CHECK(t2.requires_grad() == true);

        for (std::size_t i = 0; i < 8; ++i) CHECK(t2.at_flat(i) == doctest::Approx(static_cast<float>(i + 1)));


        auto t3 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 4 }, true);

        CHECK(t3.shape() == sizeVec{ 2, 4 });
        CHECK(t3.strides() == sizeVec{ 4, 1 });
        CHECK(t3.size() == std::size_t{ 8 });
        CHECK(t3.offset() == std::size_t{ 0 });
        CHECK(t3.requires_grad() == true);

        for (std::size_t i = 0; i < 8; ++i) CHECK(t3.at_flat(i) == doctest::Approx(static_cast<float>(i + 1)));


        auto t4 = Tensor<float>::from({ { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, true);

        CHECK(t4.shape() == sizeVec{ 2, 4 });
        CHECK(t4.strides() == sizeVec{ 4, 1 });
        CHECK(t4.size() == std::size_t{ 8 });
        CHECK(t4.offset() == std::size_t{ 0 });
        CHECK(t4.requires_grad() == true);

        for (std::size_t i = 0; i < 8; ++i) CHECK(t4.at_flat(i) == doctest::Approx(static_cast<float>(i + 1)));


        auto t5 = Tensor<float>::from(t4, true);

        CHECK(t5.shape() == sizeVec{ 2, 4 });
        CHECK(t5.strides() == sizeVec{ 4, 1 });
        CHECK(t5.size() == std::size_t{ 8 });
        CHECK(t5.offset() == std::size_t{ 0 });
        CHECK(t5.requires_grad() == true);

        for (std::size_t i = 0; i < 8; ++i) CHECK(t5.at_flat(i) == doctest::Approx(static_cast<float>(i + 1)));
    }

    SUBCASE("factory zeros_like 2D") {
        auto t1 = Tensor<float>::from({ { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, true);
        auto t2 = Tensor<float>::zeros_like(t1, false);
        
        CHECK(t2.shape() == sizeVec{ 2, 4 });
        CHECK(t2.strides() == sizeVec{ 4, 1 });
        CHECK(t2.size() == std::size_t{ 8 });
        CHECK(t2.offset() == std::size_t{ 0 });
        CHECK(t2.requires_grad() == false);

        for (std::size_t i = 0; i < 7; ++i) CHECK(t2.at_flat(i) == doctest::Approx(0.0f));
    }

    SUBCASE("factory ones_like 2D") {
        auto t1 = Tensor<float>::from({ { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, true);
        auto t2 = Tensor<float>::ones_like(t1, false);
        
        CHECK(t2.shape() == sizeVec{ 2, 4 });
        CHECK(t2.strides() == sizeVec{ 4, 1 });
        CHECK(t2.size() == std::size_t{ 8 });
        CHECK(t2.offset() == std::size_t{ 0 });
        CHECK(t2.requires_grad() == false);

        for (std::size_t i = 0; i < 7; ++i) CHECK(t2.at_flat(i) == doctest::Approx(1.0f));
    }

    SUBCASE("factory zeros 3D") {
        auto t = Tensor<float>::zeros({ 2, 2, 2 }, true);

        CHECK(t.shape() == sizeVec{ 2, 2, 2 });
        CHECK(t.strides() == sizeVec{ 4, 2, 1 });
        CHECK(t.size() == std::size_t{ 8 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == true);
        for (std::size_t i = 0; i < 8; ++i) CHECK(t.at_flat(i) == doctest::Approx(0.0f));
    }

    SUBCASE("factory ones 3D") {
        auto t = Tensor<float>::ones({ 4, 4, 4 }, false);

        CHECK(t.shape() == sizeVec{ 4, 4, 4 });
        CHECK(t.strides() == sizeVec{ 16, 4, 1 });
        CHECK(t.size() == std::size_t{ 64 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == false);
        for (std::size_t i = 0; i < 64; ++i) CHECK(t.at_flat(i) == doctest::Approx(1.0f));
    }

    SUBCASE("factory randn float 3D") {
        auto gen = std::mt19937{ 2147483647 };

        auto t = Tensor<float>::randn({ 3, 2, 3 }, gen, true);

        CHECK(t.shape() == sizeVec{ 3, 2, 3 });
        CHECK(t.strides() == sizeVec{ 6, 3, 1 });
        CHECK(t.size() == std::size_t{ 18 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == true);
    }

    SUBCASE("factory randn integral 3D") {
        auto gen = std::mt19937{ 2147483647 };

        auto t = Tensor<int>::randn(0, 10, { 3, 2, 3 }, gen, false);

        CHECK(t.shape() == sizeVec{ 3, 2, 3 });
        CHECK(t.strides() == sizeVec{ 6, 3, 1 });
        CHECK(t.size() == std::size_t{ 18 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == false);
        for (std::size_t i = 0; i < 18; ++i) {
            auto& v = t.at_flat(i);
            CHECK(v >= 0); CHECK(v <= 10);
        }
    }
}



TEST_CASE("Single element access") {

    SUBCASE("operator[] and at() 1D") {
        auto x = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, { 9 }, true);

        for (std::size_t i = 0; i < 9; ++i) {
            CHECK(x[i] == doctest::Approx(static_cast<float>(i + 1)));
            CHECK(x[i] == x.at({ i }));
        }
    }

    SUBCASE("operator[] and at() 2D") {
        auto x = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, { 3, 3 }, true);
        auto x1 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, { 9, 1 }, true);
        auto x2 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, { 1, 9 }, true);

        CHECK(x[0, 0] == doctest::Approx(1.0f)); CHECK(x[0, 0] == x.at({ 0, 0 }));
        CHECK(x[0, 1] == doctest::Approx(2.0f)); CHECK(x[0, 1] == x.at({ 0, 1 }));
        CHECK(x[0, 2] == doctest::Approx(3.0f)); CHECK(x[0, 2] == x.at({ 0, 2 }));
        CHECK(x[1, 0] == doctest::Approx(4.0f)); CHECK(x[1, 0] == x.at({ 1, 0 }));
        CHECK(x[1, 1] == doctest::Approx(5.0f)); CHECK(x[1, 1] == x.at({ 1, 1 }));
        CHECK(x[1, 2] == doctest::Approx(6.0f)); CHECK(x[1, 2] == x.at({ 1, 2 }));
        CHECK(x[2, 0] == doctest::Approx(7.0f)); CHECK(x[2, 0] == x.at({ 2, 0 }));
        CHECK(x[2, 1] == doctest::Approx(8.0f)); CHECK(x[2, 1] == x.at({ 2, 1 }));
        CHECK(x[2, 2] == doctest::Approx(9.0f)); CHECK(x[2, 2] == x.at({ 2, 2 }));

        CHECK(x1[0, 0] == doctest::Approx(1.0f)); CHECK(x1[0, 0] == x1.at({ 0, 0 }));
        CHECK(x1[1, 0] == doctest::Approx(2.0f)); CHECK(x1[1, 0] == x1.at({ 1, 0 }));
        CHECK(x1[2, 0] == doctest::Approx(3.0f)); CHECK(x1[2, 0] == x1.at({ 2, 0 }));
        CHECK(x1[3, 0] == doctest::Approx(4.0f)); CHECK(x1[3, 0] == x1.at({ 3, 0 }));
        CHECK(x1[4, 0] == doctest::Approx(5.0f)); CHECK(x1[4, 0] == x1.at({ 4, 0 }));
        CHECK(x1[5, 0] == doctest::Approx(6.0f)); CHECK(x1[5, 0] == x1.at({ 5, 0 }));
        CHECK(x1[6, 0] == doctest::Approx(7.0f)); CHECK(x1[6, 0] == x1.at({ 6, 0 }));
        CHECK(x1[7, 0] == doctest::Approx(8.0f)); CHECK(x1[7, 0] == x1.at({ 7, 0 }));
        CHECK(x1[8, 0] == doctest::Approx(9.0f)); CHECK(x1[8, 0] == x1.at({ 8, 0 }));

        CHECK(x2[0, 0] == doctest::Approx(1.0f)); CHECK(x2[0, 0] == x2.at({ 0, 0 }));
        CHECK(x2[0, 1] == doctest::Approx(2.0f)); CHECK(x2[0, 1] == x2.at({ 0, 1 }));
        CHECK(x2[0, 2] == doctest::Approx(3.0f)); CHECK(x2[0, 2] == x2.at({ 0, 2 }));
        CHECK(x2[0, 3] == doctest::Approx(4.0f)); CHECK(x2[0, 3] == x2.at({ 0, 3 }));
        CHECK(x2[0, 4] == doctest::Approx(5.0f)); CHECK(x2[0, 4] == x2.at({ 0, 4 }));
        CHECK(x2[0, 5] == doctest::Approx(6.0f)); CHECK(x2[0, 5] == x2.at({ 0, 5 }));
        CHECK(x2[0, 6] == doctest::Approx(7.0f)); CHECK(x2[0, 6] == x2.at({ 0, 6 }));
        CHECK(x2[0, 7] == doctest::Approx(8.0f)); CHECK(x2[0, 7] == x2.at({ 0, 7 }));
        CHECK(x2[0, 8] == doctest::Approx(9.0f)); CHECK(x2[0, 8] == x2.at({ 0, 8 }));
    }

    SUBCASE("operator[] and at() 3D") {
        auto x = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 2, 2 }, true);

        CHECK(x[0, 0, 0] == doctest::Approx(1.0f)); CHECK(x[0, 0, 0] == x.at({ 0, 0, 0 }));
        CHECK(x[0, 0, 1] == doctest::Approx(2.0f)); CHECK(x[0, 0, 1] == x.at({ 0, 0, 1 }));
        CHECK(x[0, 1, 0] == doctest::Approx(3.0f)); CHECK(x[0, 1, 0] == x.at({ 0, 1, 0 }));
        CHECK(x[0, 1, 1] == doctest::Approx(4.0f)); CHECK(x[0, 1, 1] == x.at({ 0, 1, 1 }));
        CHECK(x[1, 0, 0] == doctest::Approx(5.0f)); CHECK(x[1, 0, 0] == x.at({ 1, 0, 0 }));
        CHECK(x[1, 0, 1] == doctest::Approx(6.0f)); CHECK(x[1, 0, 1] == x.at({ 1, 0, 1 }));
        CHECK(x[1, 1, 0] == doctest::Approx(7.0f)); CHECK(x[1, 1, 0] == x.at({ 1, 1, 0 }));
        CHECK(x[1, 1, 1] == doctest::Approx(8.0f)); CHECK(x[1, 1, 1] == x.at({ 1, 1, 1 }));
    }
}


TEST_CASE("Tensor operations") {


    SUBCASE("Addition scalar 1D") {
        auto t = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 8 }, true);
        auto res = t + 1.0f;

        CHECK(res.shape() == sizeVec{ 8 });
        CHECK(res.strides() == sizeVec{ 1 });
        CHECK(res.size() == std::size_t{ 8 });
        CHECK(res.offset() == std::size_t{ 0 });
        CHECK(res.requires_grad() == true);
        for (std::size_t i = 0; i < 8; ++i) {
            CHECK(res[i] == t[i] + 1);
        }        
    }

    SUBCASE("Addition scalar 2D") {}
    SUBCASE("Addition scalar 3D") {}

    SUBCASE("Subtraction scalar 1D") {}
    SUBCASE("Subtraction scalar 2D") {}
    SUBCASE("Subtraction scalar 3D") {}

    SUBCASE("Multiplication scalar 1D") {}
    SUBCASE("Multiplication scalar 2D") {}
    SUBCASE("Multiplication scalar 3D") {}
    
    SUBCASE("Division scalar 1D") {}
    SUBCASE("Division scalar 2D") {}
    SUBCASE("Division scalar 3D") {}


    SUBCASE("Addition 1D") {}
    SUBCASE("Addition 3D") {}

    SUBCASE("Addition 2D, no broadcast") {
        auto x1 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 4 }, true);
        auto y1 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 4 }, true);
        auto z1 = x1 + y1;
        CHECK(z1.shape() == sizeVec{ 2, 4 });
        CHECK(z1.strides() == sizeVec{ 4, 1 });
        CHECK(z1.size() == std::size_t{ 8 });
        CHECK(z1.offset() == std::size_t{ 0 });
        CHECK(z1.requires_grad() == true);
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                CHECK(z1[i, j] == doctest::Approx(x1[i, j] + y1[i, j]));
            }
        }
        
        auto x2 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 4 }, true);
        auto y2 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 4 }, false);
        auto z2 = x2 + y2;
        CHECK(z2.requires_grad() == true);

        auto x3 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 4 }, false);
        auto y3 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 4 }, true);
        auto z3 = x3 + y3;
        CHECK(z3.requires_grad() == true);

        auto x4 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 4 }, false);
        auto y4 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 4 }, false);
        auto z4 = x4 + y4;
        CHECK(z4.requires_grad() == false);
    }

    SUBCASE("Addition 2D, broadcast") {
        auto a = Tensor<float>::from({ 1, 2, 3 }, { 3, 1 });
        auto b = Tensor<float>::from({ 10, 20, 30, 40 }, { 1, 4 });
        auto c = a + b; // should be shape [3, 4]

        CHECK(c.shape() == sizeVec{ 3, 4 });
        CHECK(c[0, 0] == doctest::Approx(11.0f));
        CHECK(c[0, 1] == doctest::Approx(21.0f));
        CHECK(c[1, 0] == doctest::Approx(12.0f));
        CHECK(c[1, 1] == doctest::Approx(22.0f));
        CHECK(c[2, 3] == doctest::Approx(43.0f));
    
        auto a2 = Tensor<float>::from({ 1, 2, 3, 4 }, { 2, 2 });
        auto b2 = Tensor<float>::from({ 2, 2 }, { 1, 2 });
        auto c2 = a2 + b2; // should be shape [2, 2]

        CHECK(c2.shape() == sizeVec{ 2, 2 });
        CHECK(c2[0, 0] == doctest::Approx(3.0f));
        CHECK(c2[0, 1] == doctest::Approx(4.0f));
        CHECK(c2[1, 0] == doctest::Approx(5.0f));
        CHECK(c2[1, 1] == doctest::Approx(6.0f));

        auto a3 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, { 3, 3 });
        auto b3 = Tensor<float>::from({ 1 }, { 1 });
        auto c3 = a3 + b3; // should be shape [3, 3]

        CHECK(c3.shape() == sizeVec{ 3, 3 });
        CHECK(c3[0, 0] == doctest::Approx(2.0f));
        CHECK(c3[0, 1] == doctest::Approx(3.0f));
        CHECK(c3[0, 2] == doctest::Approx(4.0f));
        CHECK(c3[2, 2] == doctest::Approx(10.0f));
    }

    SUBCASE("Addition 2D, transposed view") {
        auto x = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, { 3, 3 }, true);
        auto y = x.transpose();
        auto z = x + y;

        CHECK(z.shape() == sizeVec{ 3, 3 });
        CHECK(z[0, 0] == doctest::Approx(2.0f));
        CHECK(z[0, 1] == doctest::Approx(6.0f));
        CHECK(z[0, 2] == doctest::Approx(10.0f));
        CHECK(z[1, 0] == doctest::Approx(6.0f));
        CHECK(z[1, 1] == doctest::Approx(10.0f));
        CHECK(z[1, 2] == doctest::Approx(14.0f));
        CHECK(z[2, 0] == doctest::Approx(10.0f));
        CHECK(z[2, 1] == doctest::Approx(14.0f));
        CHECK(z[2, 2] == doctest::Approx(18.0f));
    }

    SUBCASE("Subtraction 1D") {}
    SUBCASE("Subtraction 2D") {}
    SUBCASE("Subtraction 3D") {}

    SUBCASE("Multiplication 1D") {}
    SUBCASE("Multiplication 2D") {}
    SUBCASE("Multiplication 3D") {}

    SUBCASE("Division 1D") {}
    SUBCASE("Division 2D") {}
    SUBCASE("Division 3D") {}

    SUBCASE("Transpose view 1D") {
        auto x = Tensor<int>::from({ 1, 2, 3, 4 }, { 4 }, true);
        auto t = x.transpose(); // returns same handle

        CHECK(t.shape() == sizeVec{ 4 });
        CHECK(t.strides() == sizeVec{ 1 });
        CHECK(t.size() == std::size_t{ 4 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == true);

        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(t[i] == i + 1);
            CHECK(t[i] == x[i]);
            CHECK(&t[i] == &x[i]);
        }

        // mutation works, both have same underlying data
        x[1] = 99;
        CHECK(t[1] == 99);
    }

    SUBCASE("Transpose view 2D") {
        auto x = Tensor<int>::from({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }, true);
        auto t = x.transpose();

        CHECK(t.shape() == sizeVec{ 3, 2 }); // transposed shape
        CHECK(t.strides() == sizeVec{ 1, 3 }); // transposed strides
        CHECK(t.size() == std::size_t{ 6 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == true);

        CHECK(t[0, 0] == x[0, 0]); CHECK(&t[0, 0] == &x[0, 0]);
        CHECK(t[0, 1] == x[1, 0]); CHECK(&t[0, 1] == &x[1, 0]);
        CHECK(t[1, 0] == x[0, 1]); CHECK(&t[1, 0] == &x[0, 1]);
        CHECK(t[1, 1] == x[1, 1]); CHECK(&t[1, 1] == &x[1, 1]);
        CHECK(t[2, 0] == x[0, 2]); CHECK(&t[2, 0] == &x[0, 2]);
        CHECK(t[2, 1] == x[1, 2]); CHECK(&t[2, 1] == &x[1, 2]);

        // mutation works, both have same underlying data
        x[1, 2] = 99;
        CHECK(t[2, 1] == 99);
    }

    SUBCASE("Transpose view 3D") {
        auto x = Tensor<float>::from({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, { 3, 2, 2 }, true);

        CHECK_THROWS_AS(x.transpose(), std::runtime_error); // cant transpose 3D tensor, need permute

        auto t = x.permute(0, 2);
      
        CHECK(t.shape() == sizeVec{ 2, 2, 3 }); // transposed shape
        CHECK(t.strides() == sizeVec{ 1, 2, 4 }); // transposed strides
        CHECK(t.size() == std::size_t{ 12 });
        CHECK(t.offset() == std::size_t{ 0 });
        CHECK(t.requires_grad() == true);

        // swapped dim 0 with dim 2 -> [i, j, k] to [k, j, i]
        CHECK(t[0,0,0] == x[0,0,0]); CHECK(&t[0,0,0] == &x[0,0,0]);
        CHECK(t[1,0,0] == x[0,0,1]); CHECK(&t[1,0,0] == &x[0,0,1]);
        CHECK(t[0,1,0] == x[0,1,0]); CHECK(&t[0,1,0] == &x[0,1,0]);
        CHECK(t[1,1,0] == x[0,1,1]); CHECK(&t[1,1,0] == &x[0,1,1]);
        CHECK(t[0,0,1] == x[1,0,0]); CHECK(&t[0,0,1] == &x[1,0,0]);
        CHECK(t[1,0,1] == x[1,0,1]); CHECK(&t[1,0,1] == &x[1,0,1]);
        CHECK(t[0,1,1] == x[1,1,0]); CHECK(&t[0,1,1] == &x[1,1,0]);
        CHECK(t[1,1,1] == x[1,1,1]); CHECK(&t[1,1,1] == &x[1,1,1]);
        CHECK(t[0,0,2] == x[2,0,0]); CHECK(&t[0,0,2] == &x[2,0,0]);
        CHECK(t[1,0,2] == x[2,0,1]); CHECK(&t[1,0,2] == &x[2,0,1]);
        CHECK(t[0,1,2] == x[2,1,0]); CHECK(&t[0,1,2] == &x[2,1,0]);
        CHECK(t[1,1,2] == x[2,1,1]); CHECK(&t[1,1,2] == &x[2,1,1]);
    }
}



TEST_CASE("Tensor operations views") {

    SUBCASE("row view exception") {
        auto x = Tensor<float>::from({ 1, 2, 3 }, { 3 }, true);
        CHECK_THROWS_AS(x.row(0), std::runtime_error);
    }

    SUBCASE("col view exception") {
        auto x = Tensor<float>::from({ 1, 2, 3 }, { 3 }, true);
        CHECK_THROWS_AS(x.col(0), std::runtime_error);
    }

    SUBCASE("row view") {
        auto x = Tensor<float>::from({ 1, 2, 3, 4, 5, 6 }, { 3, 2 }, true);

        auto r = x.row(1); // returns view of second row, idx = 1

        CHECK(r.shape() == sizeVec{ 2 });
        CHECK(r.strides() == sizeVec{ 1 });
        CHECK(r.size() == std::size_t{ 2 });
        CHECK(r.offset() == std::size_t{ 2 });
        CHECK(r.requires_grad() == true);

        CHECK(r[0] == x[1, 0]); CHECK(&r[0] == &x[1, 0]);
        CHECK(r[1] == x[1, 1]); CHECK(&r[1] == &x[1, 1]);

        auto x2 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }, true);

        auto r2 = x2.row(0); // returns view of first row, idx = 0
        
        CHECK(r2.shape() == sizeVec{ 3 });
        CHECK(r2.strides() == sizeVec{ 1 });
        CHECK(r2.size() == std::size_t{ 3 });
        CHECK(r2.offset() == std::size_t{ 0 });
        CHECK(r2.requires_grad() == true);

        CHECK(r2[0] == x2[0, 0]); CHECK(&r2[0] == &x2[0, 0]);
        CHECK(r2[1] == x2[0, 1]); CHECK(&r2[1] == &x2[0, 1]);
        CHECK(r2[2] == x2[0, 2]); CHECK(&r2[2] == &x2[0, 2]);
    }

    SUBCASE("col view") {
        auto x = Tensor<float>::from({ 1, 2, 3, 4, 5, 6 }, { 3, 2 }, true);

        auto c = x.col(1); // returns view of second col, idx = 1

        CHECK(c.shape() == sizeVec{ 3 });
        CHECK(c.strides() == sizeVec{ 2 });
        CHECK(c.size() == std::size_t{ 3 });
        CHECK(c.offset() == std::size_t{ 1 });
        CHECK(c.requires_grad() == true);

        CHECK(c[0] == x[0, 1]); CHECK(&c[0] == &x[0, 1]);
        CHECK(c[1] == x[1, 1]); CHECK(&c[1] == &x[1, 1]);
        CHECK(c[2] == x[2, 1]); CHECK(&c[2] == &x[2, 1]);

        auto x2 = Tensor<float>::from({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }, true);

        auto c2 = x2.col(0); // returns view of first col, idx = 0
        
        CHECK(c2.shape() == sizeVec{ 2 });
        CHECK(c2.strides() == sizeVec{ 3 });
        CHECK(c2.size() == std::size_t{ 2 });
        CHECK(c2.offset() == std::size_t{ 0 });
        CHECK(c2.requires_grad() == true);

        CHECK(c2[0] == x2[0, 0]); CHECK(&c2[0] == &x2[0, 0]);
        CHECK(c2[1] == x2[1, 0]); CHECK(&c2[1] == &x2[1, 0]);
    }

    SUBCASE("slice view 1d") {

    }
    
    SUBCASE("slice view 3d") {

    }
}




TEST_CASE("Tensor utilities") {

    SUBCASE("item exception") {
        auto x = Tensor<float>::from({ 1, 2, 3 }, { 3 }, true);
        CHECK_THROWS_AS(x.item(), std::runtime_error);
    }

    SUBCASE("item") {
        auto t = Tensor<float>::from({ 1 }, { 1 }, true);
        auto e = t.item();
        CHECK(e == t[0]);
    }

    SUBCASE("cast") {
        auto t = Tensor<float>::from({ 1, 2, 3, 4 }, { 2, 2 }, true);
        auto t2 = t.cast<int>(true);

        CHECK(t.shape() == t2.shape());
        CHECK(t.strides() == t2.strides());
        CHECK(t.size() == t2.size());
        CHECK(t.offset() == t2.offset());
        CHECK(t2.requires_grad() == true);
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                CHECK(t[i, j] == static_cast<float>(t2[i, j]));
            }
        }
    }
}



TEST_CASE("Backward - simple addition") {
    auto a = Tensor<float>::from({1, 2, 3}, {3}, true);
    auto b = Tensor<float>::from({4, 5, 6}, {3}, true);
    auto c = a + b;
    
    c.sum().backward();  // ← FIXED: reduce to scalar first
    
    auto* ga = a.grad_ptr();
    auto* gb = b.grad_ptr();
        
    // For addition: grad flows through as 1
    CHECK((*ga)[0] == doctest::Approx(1.0f));
    CHECK((*ga)[1] == doctest::Approx(1.0f));
    CHECK((*ga)[2] == doctest::Approx(1.0f));
    CHECK((*gb)[0] == doctest::Approx(1.0f));
    CHECK((*gb)[1] == doctest::Approx(1.0f));
    CHECK((*gb)[2] == doctest::Approx(1.0f));
}
TEST_CASE("Backward - broadcast accumulation") {
    auto a = Tensor<float>::from({1, 2, 3}, {3, 1}, true);
    auto b = Tensor<float>::from({4, 5, 6, 7}, {1, 4}, true);
    auto c = a + b;
    
    c.sum().backward();  // ← FIXED
    
    auto* ga = a.grad_ptr();
    auto* gb = b.grad_ptr();
    
    // a used 4 times per row → grad = 4
    CHECK((*ga)[0] == doctest::Approx(4.0f));
    CHECK((*ga)[1] == doctest::Approx(4.0f));
    CHECK((*ga)[2] == doctest::Approx(4.0f));
    
    // b used 3 times per column → grad = 3
    CHECK((*gb)[0] == doctest::Approx(3.0f));
    CHECK((*gb)[1] == doctest::Approx(3.0f));
    CHECK((*gb)[2] == doctest::Approx(3.0f));
    CHECK((*gb)[3] == doctest::Approx(3.0f));
}
TEST_CASE("Backward - multiplication") {
    auto a = Tensor<float>::from({2, 3}, {2}, true);
    auto b = Tensor<float>::from({4, 5}, {2}, true);
    auto c = a * b;  // [8, 15]
    
    c.sum().backward();  // ← FIXED
    
    auto* ga = a.grad_ptr();
    auto* gb = b.grad_ptr();
    
    // d(sum(a*b))/da = b (element-wise)
    CHECK((*ga)[0] == doctest::Approx(4.0f));
    CHECK((*ga)[1] == doctest::Approx(5.0f));
    
    // d(sum(a*b))/db = a (element-wise)
    CHECK((*gb)[0] == doctest::Approx(2.0f));
    CHECK((*gb)[1] == doctest::Approx(3.0f));
}
TEST_CASE("Backward - chain rule") {
    auto a = Tensor<float>::from({2}, {1}, true);
    auto b = Tensor<float>::from({3}, {1}, true);
    auto c = a * b;  // 6
    auto d = c + a;  // 6 + 2 = 8
    
    d.backward();  // ← This one is OK! d is shape {1} = scalar
    
    auto* ga = a.grad_ptr();
    auto* gb = b.grad_ptr();
    
    // d = a*b + a
    // dd/da = b + 1 = 3 + 1 = 4
    CHECK((*ga)[0] == doctest::Approx(4.0f));
    
    // dd/db = a = 2
    CHECK((*gb)[0] == doctest::Approx(2.0f));
}




TEST_CASE("View operations") {
    SUBCASE("slice 2D dim 0") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4});
        // t = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        
        auto s = t.slice(0, 1);  // Second row: [5,6,7,8]
        
        CHECK(s.shape() == sizeVec{4});
        CHECK(s.strides() == sizeVec{1});
        CHECK(s.size() == 4);
        CHECK(s.offset() == 4);
        
        CHECK(s[0] == doctest::Approx(5.0f));
        CHECK(s[1] == doctest::Approx(6.0f));
        CHECK(s[2] == doctest::Approx(7.0f));
        CHECK(s[3] == doctest::Approx(8.0f));
        
        // Pointer equality - shares storage
        CHECK(&s[0] == &t[1, 0]);
        CHECK(&s[2] == &t[1, 2]);
    }
    SUBCASE("slice 2D dim 1") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4});
        
        auto s = t.slice(1, 2);  // Third column: [3, 7, 11]
        
        CHECK(s.shape() == sizeVec{3});
        CHECK(s.strides() == sizeVec{4});
        CHECK(s.size() == 3);
        CHECK(s.offset() == 2);
        
        CHECK(s[0] == doctest::Approx(3.0f));
        CHECK(s[1] == doctest::Approx(7.0f));
        CHECK(s[2] == doctest::Approx(11.0f));
        
        CHECK(&s[0] == &t[0, 2]);
        CHECK(&s[1] == &t[1, 2]);
        CHECK(&s[2] == &t[2, 2]);
    }
    SUBCASE("slice 3D") {
        auto t = Tensor<float>::from({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {2, 3, 4});
        // Shape: 2 "matrices" of 3x4
        
        auto s = t.slice(0, 1);  // Second "matrix" → shape {3, 4}
        
        CHECK(s.shape() == sizeVec{3, 4});
        CHECK(s.strides() == sizeVec{4, 1});
        CHECK(s.size() == 12);
        CHECK(s.offset() == 12);
        
        CHECK(s[0, 0] == doctest::Approx(13.0f));
        CHECK(s[0, 3] == doctest::Approx(16.0f));
        CHECK(s[2, 3] == doctest::Approx(24.0f));
        
        auto s2 = t.slice(1, 1);  // Second row of each matrix → shape {2, 4}
        
        CHECK(s2.shape() == sizeVec{2, 4});
        CHECK(s2.strides() == sizeVec{12, 1});
        CHECK(s2.size() == 8);
        CHECK(s2.offset() == 4);
        
        CHECK(s2[0, 0] == doctest::Approx(5.0f));
        CHECK(s2[0, 3] == doctest::Approx(8.0f));
        CHECK(s2[1, 0] == doctest::Approx(17.0f));
        CHECK(s2[1, 3] == doctest::Approx(20.0f));
    }
    SUBCASE("slice out of bounds throws") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        
        CHECK_THROWS_AS(t.slice(0, 5), std::out_of_range);  // idx out of shape[0]=2
        CHECK_THROWS_AS(t.slice(3, 0), std::out_of_range);  // dim out of range
    }
    SUBCASE("row view") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
        
        auto r0 = t.row(0);
        CHECK(r0.shape() == sizeVec{3});
        CHECK(r0.strides() == sizeVec{1});
        CHECK(r0.offset() == 0);
        CHECK(r0[0] == doctest::Approx(1.0f));
        CHECK(r0[1] == doctest::Approx(2.0f));
        CHECK(r0[2] == doctest::Approx(3.0f));
        
        auto r2 = t.row(2);
        CHECK(r2.offset() == 6);
        CHECK(r2[0] == doctest::Approx(7.0f));
        CHECK(r2[1] == doctest::Approx(8.0f));
        CHECK(r2[2] == doctest::Approx(9.0f));
        
        // Mutation propagates
        r0[1] = 99.0f;
        CHECK(t[0, 1] == doctest::Approx(99.0f));
    }
    SUBCASE("row view exception on non-2D") {
        auto t1 = Tensor<float>::from({1, 2, 3}, sizeVec{3});
        CHECK_THROWS_AS(t1.row(0), std::runtime_error);
        
        auto t3 = Tensor<float>::from({1,2,3,4,5,6,7,8}, {2, 2, 2});
        CHECK_THROWS_AS(t3.row(0), std::runtime_error);
    }
    SUBCASE("col view") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
        
        auto c0 = t.col(0);
        CHECK(c0.shape() == sizeVec{3});
        CHECK(c0.strides() == sizeVec{3});
        CHECK(c0.offset() == 0);
        CHECK(c0[0] == doctest::Approx(1.0f));
        CHECK(c0[1] == doctest::Approx(4.0f));
        CHECK(c0[2] == doctest::Approx(7.0f));
        
        auto c2 = t.col(2);
        CHECK(c2.offset() == 2);
        CHECK(c2[0] == doctest::Approx(3.0f));
        CHECK(c2[1] == doctest::Approx(6.0f));
        CHECK(c2[2] == doctest::Approx(9.0f));
        
        // Mutation propagates
        c0[2] = 77.0f;
        CHECK(t[2, 0] == doctest::Approx(77.0f));
    }
    SUBCASE("col view exception on non-2D") {
        auto t1 = Tensor<float>::from({1, 2, 3}, sizeVec{3});
        CHECK_THROWS_AS(t1.col(0), std::runtime_error);
    }
    SUBCASE("transpose 1D returns same") {
        auto t = Tensor<float>::from({1, 2, 3, 4}, sizeVec{4});
        auto tT = t.transpose();
        
        CHECK(tT.shape() == sizeVec{4});
        CHECK(tT.strides() == sizeVec{1});
        
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK(&tT[i] == &t[i]);
        }
    }
    SUBCASE("transpose 2D") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        // t = [[1,2,3], [4,5,6]]
        
        auto tT = t.transpose();
        
        CHECK(tT.shape() == sizeVec{3, 2});
        CHECK(tT.strides() == sizeVec{1, 3});
        
        // tT[i,j] == t[j,i]
        CHECK(tT[0, 0] == doctest::Approx(1.0f)); CHECK(&tT[0, 0] == &t[0, 0]);
        CHECK(tT[0, 1] == doctest::Approx(4.0f)); CHECK(&tT[0, 1] == &t[1, 0]);
        CHECK(tT[1, 0] == doctest::Approx(2.0f)); CHECK(&tT[1, 0] == &t[0, 1]);
        CHECK(tT[1, 1] == doctest::Approx(5.0f)); CHECK(&tT[1, 1] == &t[1, 1]);
        CHECK(tT[2, 0] == doctest::Approx(3.0f)); CHECK(&tT[2, 0] == &t[0, 2]);
        CHECK(tT[2, 1] == doctest::Approx(6.0f)); CHECK(&tT[2, 1] == &t[1, 2]);
        
        // Mutation propagates
        tT[1, 1] = 55.0f;
        CHECK(t[1, 1] == doctest::Approx(55.0f));
    }
    SUBCASE("transpose 3D throws") {
        auto t = Tensor<float>::from({1,2,3,4,5,6,7,8}, {2, 2, 2});
        CHECK_THROWS_AS(t.transpose(), std::runtime_error);
    }
    SUBCASE("permute 2D") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto p = t.permute(0, 1);  // Same as transpose for 2D
        
        CHECK(p.shape() == sizeVec{3, 2});
        CHECK(p.strides() == sizeVec{1, 3});
        
        CHECK(p[0, 0] == t[0, 0]);
        CHECK(p[0, 1] == t[1, 0]);
        CHECK(p[2, 1] == t[1, 2]);
    }
    SUBCASE("permute 3D") {
        auto t = Tensor<float>::from({1,2,3,4,5,6,7,8,9,10,11,12}, {3, 2, 2});
        // Shape {3, 2, 2}, strides {4, 2, 1}
        
        auto p = t.permute(0, 2);  // Swap dim 0 and dim 2
        
        CHECK(p.shape() == sizeVec{2, 2, 3});
        CHECK(p.strides() == sizeVec{1, 2, 4});
        
        // p[k, j, i] == t[i, j, k]
        CHECK(p[0, 0, 0] == t[0, 0, 0]);
        CHECK(p[1, 0, 0] == t[0, 0, 1]);
        CHECK(p[0, 1, 0] == t[0, 1, 0]);
        CHECK(p[0, 0, 1] == t[1, 0, 0]);
        CHECK(p[1, 1, 2] == t[2, 1, 1]);
    }
    SUBCASE("permute same dim returns same") {
        auto t = Tensor<float>::from({1, 2, 3, 4}, {2, 2});
        auto p = t.permute(0, 0);
        
        CHECK(p.shape() == t.shape());
        CHECK(p.strides() == t.strides());
        CHECK(&p[0, 0] == &t[0, 0]);
    }
    SUBCASE("permute out of bounds throws") {
        auto t = Tensor<float>::from({1, 2, 3, 4}, {2, 2});
        CHECK_THROWS_AS(t.permute(0, 5), std::out_of_range);
        CHECK_THROWS_AS(t.permute(3, 1), std::out_of_range);
    }
    SUBCASE("chained views: transpose then row") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        // t = [[1,2,3], [4,5,6]]
        // t.T = [[1,4], [2,5], [3,6]]
        
        auto tT = t.transpose();
        auto r1 = tT.row(1);  // [2, 5]
        
        CHECK(r1.shape() == sizeVec{2});
        CHECK(r1[0] == doctest::Approx(2.0f));
        CHECK(r1[1] == doctest::Approx(5.0f));
        
        CHECK(&r1[0] == &t[0, 1]);
        CHECK(&r1[1] == &t[1, 1]);
    }
    SUBCASE("chained views: col then operation") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {3, 2});
        // t = [[1,2], [3,4], [5,6]]
        
        auto c0 = t.col(0);  // [1, 3, 5]
        auto c1 = t.col(1);  // [2, 4, 6]
        auto sum = c0 + c1;  // [3, 7, 11]
        
        CHECK(sum[0] == doctest::Approx(3.0f));
        CHECK(sum[1] == doctest::Approx(7.0f));
        CHECK(sum[2] == doctest::Approx(11.0f));
    }
}


TEST_CASE("Scalar binary operations") {
    SUBCASE("addition 1D") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5}, sizeVec{5});
        auto r = t + 10.0f;
        
        CHECK(r.shape() == sizeVec{5});
        CHECK(r.strides() == sizeVec{1});
        CHECK(r.size() == 5);
        
        CHECK(r[0] == doctest::Approx(11.0f));
        CHECK(r[1] == doctest::Approx(12.0f));
        CHECK(r[2] == doctest::Approx(13.0f));
        CHECK(r[3] == doctest::Approx(14.0f));
        CHECK(r[4] == doctest::Approx(15.0f));
        
        // Original unchanged
        CHECK(t[0] == doctest::Approx(1.0f));
    }
    SUBCASE("addition 2D") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto r = t + 100.0f;
        
        CHECK(r.shape() == sizeVec{2, 3});
        CHECK(r.strides() == sizeVec{3, 1});
        
        CHECK(r[0, 0] == doctest::Approx(101.0f));
        CHECK(r[0, 1] == doctest::Approx(102.0f));
        CHECK(r[0, 2] == doctest::Approx(103.0f));
        CHECK(r[1, 0] == doctest::Approx(104.0f));
        CHECK(r[1, 1] == doctest::Approx(105.0f));
        CHECK(r[1, 2] == doctest::Approx(106.0f));
    }
    SUBCASE("addition 3D") {
        auto t = Tensor<float>::from({1,2,3,4,5,6,7,8}, {2, 2, 2});
        auto r = t + 1000.0f;
        
        CHECK(r.shape() == sizeVec{2, 2, 2});
        CHECK(r[0, 0, 0] == doctest::Approx(1001.0f));
        CHECK(r[0, 1, 1] == doctest::Approx(1004.0f));
        CHECK(r[1, 1, 1] == doctest::Approx(1008.0f));
    }
    SUBCASE("subtraction 1D") {
        auto t = Tensor<float>::from({10, 20, 30, 40}, sizeVec{4});
        auto r = t - 5.0f;
        
        CHECK(r[0] == doctest::Approx(5.0f));
        CHECK(r[1] == doctest::Approx(15.0f));
        CHECK(r[2] == doctest::Approx(25.0f));
        CHECK(r[3] == doctest::Approx(35.0f));
    }
    SUBCASE("subtraction 2D") {
        auto t = Tensor<float>::from({100, 200, 300, 400}, {2, 2});
        auto r = t - 50.0f;
        
        CHECK(r[0, 0] == doctest::Approx(50.0f));
        CHECK(r[0, 1] == doctest::Approx(150.0f));
        CHECK(r[1, 0] == doctest::Approx(250.0f));
        CHECK(r[1, 1] == doctest::Approx(350.0f));
    }
    SUBCASE("multiplication 1D") {
        auto t = Tensor<float>::from({1, 2, 3, 4}, sizeVec{4});
        auto r = t * 3.0f;
        
        CHECK(r[0] == doctest::Approx(3.0f));
        CHECK(r[1] == doctest::Approx(6.0f));
        CHECK(r[2] == doctest::Approx(9.0f));
        CHECK(r[3] == doctest::Approx(12.0f));
    }
    SUBCASE("multiplication 2D") {
        auto t = Tensor<float>::from({2, 4, 6, 8, 10, 12}, {2, 3});
        auto r = t * 0.5f;
        
        CHECK(r[0, 0] == doctest::Approx(1.0f));
        CHECK(r[0, 1] == doctest::Approx(2.0f));
        CHECK(r[0, 2] == doctest::Approx(3.0f));
        CHECK(r[1, 0] == doctest::Approx(4.0f));
        CHECK(r[1, 1] == doctest::Approx(5.0f));
        CHECK(r[1, 2] == doctest::Approx(6.0f));
    }
    SUBCASE("multiplication 3D") {
        auto t = Tensor<float>::from({1,2,3,4,5,6,7,8}, {2, 2, 2});
        auto r = t * 10.0f;
        
        CHECK(r[0, 0, 0] == doctest::Approx(10.0f));
        CHECK(r[0, 0, 1] == doctest::Approx(20.0f));
        CHECK(r[1, 1, 1] == doctest::Approx(80.0f));
    }
    SUBCASE("division 1D") {
        auto t = Tensor<float>::from({10, 20, 30, 40}, sizeVec{4});
        auto r = t / 10.0f;
        
        CHECK(r[0] == doctest::Approx(1.0f));
        CHECK(r[1] == doctest::Approx(2.0f));
        CHECK(r[2] == doctest::Approx(3.0f));
        CHECK(r[3] == doctest::Approx(4.0f));
    }
    SUBCASE("division 2D") {
        auto t = Tensor<float>::from({4, 8, 12, 16, 20, 24}, {2, 3});
        auto r = t / 4.0f;
        
        CHECK(r[0, 0] == doctest::Approx(1.0f));
        CHECK(r[0, 1] == doctest::Approx(2.0f));
        CHECK(r[0, 2] == doctest::Approx(3.0f));
        CHECK(r[1, 0] == doctest::Approx(4.0f));
        CHECK(r[1, 1] == doctest::Approx(5.0f));
        CHECK(r[1, 2] == doctest::Approx(6.0f));
    }
    SUBCASE("division 3D") {
        auto t = Tensor<float>::from({100, 200, 300, 400, 500, 600, 700, 800}, {2, 2, 2});
        auto r = t / 100.0f;
        
        CHECK(r[0, 0, 0] == doctest::Approx(1.0f));
        CHECK(r[0, 0, 1] == doctest::Approx(2.0f));
        CHECK(r[1, 1, 1] == doctest::Approx(8.0f));
    }
    SUBCASE("requires_grad propagates") {
        auto t1 = Tensor<float>::from({1, 2, 3}, {3}, true);
        auto r1 = t1 + 1.0f;
        CHECK(r1.requires_grad() == true);
        
        auto t2 = Tensor<float>::from({1, 2, 3}, {3}, false);
        auto r2 = t2 * 2.0f;
        CHECK(r2.requires_grad() == false);
    }
    SUBCASE("chained scalar operations") {
        auto t = Tensor<float>::from({1, 2, 3, 4}, {2, 2});
        auto r = (t + 10.0f) * 2.0f - 5.0f;
        
        // (1+10)*2-5 = 17, (2+10)*2-5 = 19, (3+10)*2-5 = 21, (4+10)*2-5 = 23
        CHECK(r[0, 0] == doctest::Approx(17.0f));
        CHECK(r[0, 1] == doctest::Approx(19.0f));
        CHECK(r[1, 0] == doctest::Approx(21.0f));
        CHECK(r[1, 1] == doctest::Approx(23.0f));
    }
}

TEST_CASE("Scalar operations on views") {
    SUBCASE("scalar add on row view") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto row = t.row(1);  // View of [4, 5, 6]
        auto r = row + 10.0f;
        
        CHECK(r.size() == 3);
        CHECK(r[0] == doctest::Approx(14.0f));
        CHECK(r[1] == doctest::Approx(15.0f));
        CHECK(r[2] == doctest::Approx(16.0f));
        
        // Original unchanged
        CHECK(t[1, 0] == doctest::Approx(4.0f));
    }
    
    SUBCASE("scalar mul on col view") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {3, 2});
        auto col = t.col(0);  // View of [1, 3, 5] with stride 2
        auto r = col * 2.0f;
        
        CHECK(r[0] == doctest::Approx(2.0f));
        CHECK(r[1] == doctest::Approx(6.0f));
        CHECK(r[2] == doctest::Approx(10.0f));
    }
    
    SUBCASE("scalar add on transposed view") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto tT = t.transpose();  // shape {3, 2}
        auto r = tT + 100.0f;
        
        CHECK(r.shape() == sizeVec{3, 2});
        CHECK(r[0, 0] == doctest::Approx(101.0f));  // was t[0,0]=1
        CHECK(r[0, 1] == doctest::Approx(104.0f));  // was t[1,0]=4
        CHECK(r[1, 0] == doctest::Approx(102.0f));  // was t[0,1]=2
        CHECK(r[2, 1] == doctest::Approx(106.0f));  // was t[1,2]=6
    }
}

TEST_CASE("In-place scalar operations") {
    SUBCASE("in-place add") {
        auto t = Tensor<float>::from({1, 2, 3, 4}, {2, 2}, false);  // No grad!
        t += 10.0f;
        CHECK(t[0, 0] == doctest::Approx(11.0f));
        CHECK(t[1, 1] == doctest::Approx(14.0f));
    }
    
    SUBCASE("in-place on view modifies only view elements") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3}, false);
        auto row = t.row(1);
        row += 100.0f;
        
        // Row 1 modified
        CHECK(t[1, 0] == doctest::Approx(104.0f));
        CHECK(t[1, 1] == doctest::Approx(105.0f));
        CHECK(t[1, 2] == doctest::Approx(106.0f));
        
        // Row 0 unchanged
        CHECK(t[0, 0] == doctest::Approx(1.0f));
        CHECK(t[0, 1] == doctest::Approx(2.0f));
    }
    
    SUBCASE("in-place throws with requires_grad") {
        auto t = Tensor<float>::from({1, 2, 3}, {3}, true);
        CHECK_THROWS_AS(t += 1.0f, std::runtime_error);
    }
}

TEST_CASE("Tensor-tensor operations") {
    SUBCASE("addition same shape") {
        auto a = Tensor<float>::from({1, 2, 3, 4}, {2, 2});
        auto b = Tensor<float>::from({10, 20, 30, 40}, {2, 2});
        auto c = a + b;
        
        CHECK(c[0, 0] == doctest::Approx(11.0f));
        CHECK(c[0, 1] == doctest::Approx(22.0f));
        CHECK(c[1, 0] == doctest::Approx(33.0f));
        CHECK(c[1, 1] == doctest::Approx(44.0f));
    }
    
    SUBCASE("subtraction same shape") {
        auto a = Tensor<float>::from({10, 20, 30}, sizeVec{3});
        auto b = Tensor<float>::from({1, 2, 3}, sizeVec{3});
        auto c = a - b;
        CHECK(c[0] == doctest::Approx(9.0f));
        CHECK(c[1] == doctest::Approx(18.0f));
        CHECK(c[2] == doctest::Approx(27.0f));
    }
    
    SUBCASE("multiplication same shape") {
        auto a = Tensor<float>::from({2, 3, 4}, sizeVec{3});
        auto b = Tensor<float>::from({5, 6, 7}, sizeVec{3});
        auto c = a * b;
        CHECK(c[0] == doctest::Approx(10.0f));
        CHECK(c[1] == doctest::Approx(18.0f));
        CHECK(c[2] == doctest::Approx(28.0f));
    }
    
    SUBCASE("division same shape") {
        auto a = Tensor<float>::from({10, 20, 30}, sizeVec{3});
        auto b = Tensor<float>::from({2, 4, 5}, sizeVec{3});
        auto c = a / b;
        CHECK(c[0] == doctest::Approx(5.0f));
        CHECK(c[1] == doctest::Approx(5.0f));
        CHECK(c[2] == doctest::Approx(6.0f));
    }
}

TEST_CASE("Tensor-tensor broadcast operations") {
    SUBCASE("broadcast row vector") {
        // {3,1} + {1,4} → {3,4}
        auto a = Tensor<float>::from({1, 2, 3}, {3, 1});
        auto b = Tensor<float>::from({10, 20, 30, 40}, {1, 4});
        auto c = a + b;
        
        CHECK(c.shape() == sizeVec{3, 4});
        CHECK(c[0, 0] == doctest::Approx(11.0f));  // 1 + 10
        CHECK(c[0, 3] == doctest::Approx(41.0f));  // 1 + 40
        CHECK(c[2, 0] == doctest::Approx(13.0f));  // 3 + 10
        CHECK(c[2, 3] == doctest::Approx(43.0f));  // 3 + 40
    }
    
    SUBCASE("broadcast scalar tensor") {
        auto a = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto b = Tensor<float>::from({10}, sizeVec{1});
        auto c = a + b;
        
        CHECK(c.shape() == sizeVec{2, 3});
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                CHECK(c[i, j] == doctest::Approx(a[i, j] + 10.0f));
            }
        }
    }
    
    SUBCASE("broadcast multiplication") {
        auto a = Tensor<float>::from({1, 2, 3}, {3, 1});
        auto b = Tensor<float>::from({2, 3}, {1, 2});
        auto c = a * b;  // {3, 2}
        
        CHECK(c[0, 0] == doctest::Approx(2.0f));   // 1*2
        CHECK(c[0, 1] == doctest::Approx(3.0f));   // 1*3
        CHECK(c[1, 0] == doctest::Approx(4.0f));   // 2*2
        CHECK(c[2, 1] == doctest::Approx(9.0f));   // 3*3
    }
    
    SUBCASE("incompatible shapes throw") {
        auto a = Tensor<float>::from({1, 2, 3}, sizeVec{3});
        auto b = Tensor<float>::from({1, 2, 3, 4}, sizeVec{4});
        CHECK_THROWS_AS(a + b, std::invalid_argument);
    }
}

TEST_CASE("Tensor operations on views") {
    SUBCASE("row + row") {
        auto t = Tensor<float>::from({1,2,3,4,5,6}, {2, 3});
        auto r0 = t.row(0);  // {1, 2, 3}
        auto r1 = t.row(1);  // {4, 5, 6}
        auto sum = r0 + r1;
        
        CHECK(sum[0] == doctest::Approx(5.0f));
        CHECK(sum[1] == doctest::Approx(7.0f));
        CHECK(sum[2] == doctest::Approx(9.0f));
    }
    
    SUBCASE("transposed tensor + tensor") {
        auto t = Tensor<float>::from({1,2,3,4}, {2, 2});
        auto tT = t.transpose();
        auto r = t + tT;  // t + t^T (should be symmetric)
        
        CHECK(r[0, 1] == r[1, 0]);  // Symmetric
        CHECK(r[0, 0] == doctest::Approx(2.0f));   // 1+1
        CHECK(r[0, 1] == doctest::Approx(5.0f));   // 2+3
        CHECK(r[1, 1] == doctest::Approx(8.0f));   // 4+4
    }
    
    SUBCASE("col * scalar tensor") {
        auto t = Tensor<float>::from({1,2,3,4,5,6}, {3, 2});
        auto col = t.col(1);  // {2, 4, 6}
        auto s = Tensor<float>::from({10}, sizeVec{1});
        auto r = col * s;
        
        CHECK(r[0] == doctest::Approx(20.0f));
        CHECK(r[1] == doctest::Approx(40.0f));
        CHECK(r[2] == doctest::Approx(60.0f));
    }
}

TEST_CASE("In-place tensor operations") {
    SUBCASE("in-place add same shape") {
        auto a = Tensor<float>::from({1, 2, 3, 4}, {2, 2}, false);
        auto b = Tensor<float>::from({10, 20, 30, 40}, {2, 2}, false);
        a += b;
        
        CHECK(a[0, 0] == doctest::Approx(11.0f));
        CHECK(a[1, 1] == doctest::Approx(44.0f));
    }
    
    SUBCASE("in-place add with broadcast") {
        auto a = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3}, false);
        auto b = Tensor<float>::from({10, 20, 30}, {1, 3}, false);
        a += b;
        
        CHECK(a[0, 0] == doctest::Approx(11.0f));
        CHECK(a[0, 2] == doctest::Approx(33.0f));
        CHECK(a[1, 0] == doctest::Approx(14.0f));
    }
    
    SUBCASE("in-place on view") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3}, false);
        auto b = Tensor<float>::from({100, 200, 300}, {3}, false);
        auto row = t.row(0);
        row += b;
        
        CHECK(t[0, 0] == doctest::Approx(101.0f));
        CHECK(t[0, 1] == doctest::Approx(202.0f));
        CHECK(t[1, 0] == doctest::Approx(4.0f));  // Unchanged
    }
    
    SUBCASE("throws when broadcast changes shape") {
        auto a = Tensor<float>::from({1, 2, 3}, {3, 1}, false);
        auto b = Tensor<float>::from({1, 2, 3, 4}, {1, 4}, false);
        CHECK_THROWS_AS(a += b, std::runtime_error);  // Would become {3,4}
    }
}

TEST_CASE("Reduction operations") {
    SUBCASE("sum total") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto s = t.sum();
        CHECK(s.shape() == sizeVec{1});
        CHECK(s.item() == doctest::Approx(21.0f));
    }
    
    SUBCASE("sum axis 0") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto s = t.sum(0);  // Sum over rows
        CHECK(s.shape() == sizeVec{3});
        CHECK(s[0] == doctest::Approx(5.0f));   // 1+4
        CHECK(s[1] == doctest::Approx(7.0f));   // 2+5
        CHECK(s[2] == doctest::Approx(9.0f));   // 3+6
    }
    
    SUBCASE("sum axis 1 keepdim") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto s = t.sum(1, true);
        CHECK(s.shape() == sizeVec{2, 1});
        CHECK(s[0, 0] == doctest::Approx(6.0f));   // 1+2+3
        CHECK(s[1, 0] == doctest::Approx(15.0f));  // 4+5+6
    }
    
    SUBCASE("prod") {
        auto t = Tensor<float>::from({2, 3, 4}, sizeVec{3});
        auto p = t.prod();
        CHECK(p.item() == doctest::Approx(24.0f));
    }
    
    SUBCASE("min/max") {
        auto t = Tensor<float>::from({3, 1, 4, 1, 5, 9}, {2, 3});
        CHECK(t.min().item() == doctest::Approx(1.0f));
        CHECK(t.max().item() == doctest::Approx(9.0f));
    }
    
    SUBCASE("mean") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        CHECK(t.mean().item() == doctest::Approx(3.5f));
    }
    
    SUBCASE("mean axis") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto m = t.mean(1);  // Mean over cols
        CHECK(m[0] == doctest::Approx(2.0f));  // (1+2+3)/3
        CHECK(m[1] == doctest::Approx(5.0f));  // (4+5+6)/3
    }
}

TEST_CASE("Reduction operations on views") {
    SUBCASE("sum on row view") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto row = t.row(1);  // {4, 5, 6}
        auto s = row.sum();
        CHECK(s.item() == doctest::Approx(15.0f));
    }
    
    SUBCASE("sum on transposed tensor") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto tT = t.transpose();  // {3, 2}
        auto s = tT.sum(0);  // Sum over first dim → {2}
        CHECK(s[0] == doctest::Approx(6.0f));   // 1+2+3
        CHECK(s[1] == doctest::Approx(15.0f));  // 4+5+6
    }
    
    SUBCASE("mean on col view") {
        auto t = Tensor<float>::from({10, 20, 30, 40, 50, 60}, {3, 2});
        auto col = t.col(1);  // {20, 40, 60}
        auto m = col.mean();
        CHECK(m.item() == doctest::Approx(40.0f));
    }
}

TEST_CASE("Gradient: scalar operations") {
    using makemore::Tensor;
    SUBCASE("scalar add gradient") {
        auto t = Tensor<float>::from({2, 3, 4, 5}, {4}, true);
        auto r = t + 10.0f;
        auto loss = r.sum();
        loss.backward();
        
        // d(sum(t + c))/dt_i = 1 for all i
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        CHECK((*g)[0] == doctest::Approx(1.0f));
        CHECK((*g)[1] == doctest::Approx(1.0f));
        CHECK((*g)[2] == doctest::Approx(1.0f));
        CHECK((*g)[3] == doctest::Approx(1.0f));
    }
    SUBCASE("scalar sub gradient") {
        auto t = Tensor<float>::from({2, 3, 4}, {3}, true);
        auto r = t - 5.0f;
        r.sum().backward();
        
        // d(sum(t - c))/dt_i = 1 for all i
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        CHECK((*g)[0] == doctest::Approx(1.0f));
        CHECK((*g)[1] == doctest::Approx(1.0f));
        CHECK((*g)[2] == doctest::Approx(1.0f));
    }
    SUBCASE("scalar mul gradient") {
        auto t = Tensor<float>::from({2, 3, 4}, {3}, true);
        auto r = t * 5.0f;
        r.sum().backward();
        
        // d(sum(t * c))/dt_i = c = 5
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        CHECK((*g)[0] == doctest::Approx(5.0f));
        CHECK((*g)[1] == doctest::Approx(5.0f));
        CHECK((*g)[2] == doctest::Approx(5.0f));
    }
    SUBCASE("scalar div gradient") {
        auto t = Tensor<float>::from({10, 20, 30}, {3}, true);
        auto r = t / 2.0f;
        r.sum().backward();
        
        // d(sum(t / c))/dt_i = 1/c = 0.5
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        CHECK((*g)[0] == doctest::Approx(0.5f));
        CHECK((*g)[1] == doctest::Approx(0.5f));
        CHECK((*g)[2] == doctest::Approx(0.5f));
    }
    SUBCASE("chained scalar operations gradient") {
        auto t = Tensor<float>::from({1, 2, 3}, {3}, true);
        auto r = (t * 2.0f) + 1.0f;  // 2t + 1
        r.sum().backward();
        
        // d(sum(2t + 1))/dt_i = 2
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        CHECK((*g)[0] == doctest::Approx(2.0f));
        CHECK((*g)[1] == doctest::Approx(2.0f));
        CHECK((*g)[2] == doctest::Approx(2.0f));
    }
}
TEST_CASE("Gradient: tensor-tensor operations") {
    using makemore::Tensor;
    SUBCASE("add gradient same shape") {
        auto a = Tensor<float>::from({1, 2, 3}, {3}, true);
        auto b = Tensor<float>::from({4, 5, 6}, {3}, true);
        auto c = a + b;
        c.sum().backward();
        
        // d(sum(a+b))/da_i = 1, d(sum(a+b))/db_i = 1
        auto* ga = a.grad_ptr();
        auto* gb = b.grad_ptr();
        REQUIRE(ga != nullptr);
        REQUIRE(gb != nullptr);
        
        CHECK((*ga)[0] == doctest::Approx(1.0f));
        CHECK((*ga)[1] == doctest::Approx(1.0f));
        CHECK((*ga)[2] == doctest::Approx(1.0f));
        CHECK((*gb)[0] == doctest::Approx(1.0f));
        CHECK((*gb)[1] == doctest::Approx(1.0f));
        CHECK((*gb)[2] == doctest::Approx(1.0f));
    }
    SUBCASE("sub gradient same shape") {
        auto a = Tensor<float>::from({10, 20, 30}, {3}, true);
        auto b = Tensor<float>::from({1, 2, 3}, {3}, true);
        auto c = a - b;
        c.sum().backward();
        
        // d(sum(a-b))/da_i = 1, d(sum(a-b))/db_i = -1
        auto* ga = a.grad_ptr();
        auto* gb = b.grad_ptr();
        
        CHECK((*ga)[0] == doctest::Approx(1.0f));
        CHECK((*ga)[1] == doctest::Approx(1.0f));
        CHECK((*gb)[0] == doctest::Approx(-1.0f));
        CHECK((*gb)[1] == doctest::Approx(-1.0f));
    }
    SUBCASE("mul gradient same shape") {
        auto a = Tensor<float>::from({2, 3}, {2}, true);
        auto b = Tensor<float>::from({4, 5}, {2}, true);
        auto c = a * b;  // {8, 15}
        c.sum().backward();
        
        // d(a*b)/da = b, d(a*b)/db = a
        auto* ga = a.grad_ptr();
        auto* gb = b.grad_ptr();
        
        CHECK((*ga)[0] == doctest::Approx(4.0f));  // b[0]
        CHECK((*ga)[1] == doctest::Approx(5.0f));  // b[1]
        CHECK((*gb)[0] == doctest::Approx(2.0f));  // a[0]
        CHECK((*gb)[1] == doctest::Approx(3.0f));  // a[1]
    }
    SUBCASE("div gradient same shape") {
        auto a = Tensor<float>::from({10, 20}, {2}, true);
        auto b = Tensor<float>::from({2, 4}, {2}, true);
        auto c = a / b;  // {5, 5}
        c.sum().backward();
        
        // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        auto* ga = a.grad_ptr();
        auto* gb = b.grad_ptr();
        
        CHECK((*ga)[0] == doctest::Approx(0.5f));    // 1/2
        CHECK((*ga)[1] == doctest::Approx(0.25f));   // 1/4
        CHECK((*gb)[0] == doctest::Approx(-2.5f));   // -10/4
        CHECK((*gb)[1] == doctest::Approx(-1.25f));  // -20/16
    }
    SUBCASE("broadcast add gradient accumulation") {
        auto a = Tensor<float>::from({1, 2, 3}, {3, 1}, true);
        auto b = Tensor<float>::from({10, 20}, {1, 2}, true);
        auto c = a + b;  // {3, 2}
        c.sum().backward();
        
        // a is used 2 times (broadcast over cols) → each grad = 2
        // b is used 3 times (broadcast over rows) → each grad = 3
        auto* ga = a.grad_ptr();
        auto* gb = b.grad_ptr();
        
        CHECK((*ga)[0] == doctest::Approx(2.0f));
        CHECK((*ga)[1] == doctest::Approx(2.0f));
        CHECK((*ga)[2] == doctest::Approx(2.0f));
        CHECK((*gb)[0] == doctest::Approx(3.0f));
        CHECK((*gb)[1] == doctest::Approx(3.0f));
    }
    SUBCASE("broadcast mul gradient") {
        auto a = Tensor<float>::from({2, 3}, {2, 1}, true);
        auto b = Tensor<float>::from({4, 5, 6}, {1, 3}, true);
        auto c = a * b;  // {2, 3}: [[8,10,12], [12,15,18]]
        c.sum().backward();
        
        // grad_a[0] = sum(b) = 4+5+6 = 15
        // grad_a[1] = sum(b) = 4+5+6 = 15
        // grad_b[0] = sum(a) = 2+3 = 5
        // grad_b[1] = sum(a) = 2+3 = 5
        // grad_b[2] = sum(a) = 2+3 = 5
        auto* ga = a.grad_ptr();
        auto* gb = b.grad_ptr();
        
        CHECK((*ga)[0] == doctest::Approx(15.0f));
        CHECK((*ga)[1] == doctest::Approx(15.0f));
        CHECK((*gb)[0] == doctest::Approx(5.0f));
        CHECK((*gb)[1] == doctest::Approx(5.0f));
        CHECK((*gb)[2] == doctest::Approx(5.0f));
    }
}
TEST_CASE("Gradient: reduction operations") {
    using makemore::Tensor;
    SUBCASE("sum total gradient") {
        auto t = Tensor<float>::from({1, 2, 3, 4}, {2, 2}, true);
        auto s = t.sum();
        s.backward();
        
        // d(sum)/dt_i = 1 for all i
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        CHECK((*g)[0] == doctest::Approx(1.0f));
        CHECK((*g)[1] == doctest::Approx(1.0f));
        CHECK((*g)[2] == doctest::Approx(1.0f));
        CHECK((*g)[3] == doctest::Approx(1.0f));
    }
    SUBCASE("sum axis gradient") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3}, true);
        auto s = t.sum(1);  // Sum over cols → {6, 15}
        s.sum().backward();  // Need scalar for backward
        
        // Each element contributes 1 to its row's sum, then 1 to the total
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK((*g)[i] == doctest::Approx(1.0f));
        }
    }
    SUBCASE("mean total gradient") {
        auto t = Tensor<float>::from({1, 2, 3, 4}, {4}, true);
        auto m = t.mean();
        m.backward();
        
        // d(mean)/dt_i = 1/n = 0.25
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        CHECK((*g)[0] == doctest::Approx(0.25f));
        CHECK((*g)[1] == doctest::Approx(0.25f));
        CHECK((*g)[2] == doctest::Approx(0.25f));
        CHECK((*g)[3] == doctest::Approx(0.25f));
    }
    SUBCASE("mean axis gradient") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3}, true);
        auto m = t.mean(1);  // Mean over cols → {2, 5}
        m.sum().backward();
        
        // d/dt_ij = 1/3 (divided by 3 elements in each row)
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        float expected = 1.0f / 3.0f;
        for (std::size_t i = 0; i < 6; ++i) {
            CHECK((*g)[i] == doctest::Approx(expected));
        }
    }
    SUBCASE("prod gradient (no zeros)") {
        auto t = Tensor<float>::from({2, 3, 4}, {3}, true);
        auto p = t.prod();  // 24
        p.backward();
        
        // d(prod)/dt_i = prod / t_i
        // d/dt[0] = 24/2 = 12
        // d/dt[1] = 24/3 = 8
        // d/dt[2] = 24/4 = 6
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        CHECK((*g)[0] == doctest::Approx(12.0f));
        CHECK((*g)[1] == doctest::Approx(8.0f));
        CHECK((*g)[2] == doctest::Approx(6.0f));
    }
}
TEST_CASE("Gradient: chain rule") {
    using makemore::Tensor;
    SUBCASE("simple composition: d = a*b + a") {
        auto a = Tensor<float>::from({2}, {1}, true);
        auto b = Tensor<float>::from({3}, {1}, true);
        auto c = a * b;    // c = 6
        auto d = c + a;    // d = 8
        
        d.backward();
        
        // d = a*b + a
        // dd/da = b + 1 = 3 + 1 = 4
        // dd/db = a = 2
        auto* ga = a.grad_ptr();
        auto* gb = b.grad_ptr();
        
        CHECK((*ga)[0] == doctest::Approx(4.0f));
        CHECK((*gb)[0] == doctest::Approx(2.0f));
    }
    SUBCASE("longer chain: (a + b) * (a - b)") {
        auto a = Tensor<float>::from({5}, {1}, true);
        auto b = Tensor<float>::from({3}, {1}, true);
        auto sum = a + b;   // 8
        auto diff = a - b;  // 2
        auto prod = sum * diff;  // 16
        
        prod.backward();
        
        // prod = (a+b)(a-b) = a^2 - b^2
        // d/da = 2a = 10
        // d/db = -2b = -6
        auto* ga = a.grad_ptr();
        auto* gb = b.grad_ptr();
        
        CHECK((*ga)[0] == doctest::Approx(10.0f));
        CHECK((*gb)[0] == doctest::Approx(-6.0f));
    }
    SUBCASE("multiple uses of same tensor") {
        auto x = Tensor<float>::from({3}, {1}, true);
        auto y = x * x;  // x^2 = 9
        auto z = y + x;  // x^2 + x = 12
        
        z.backward();
        
        // dz/dx = 2x + 1 = 7
        auto* g = x.grad_ptr();
        CHECK((*g)[0] == doctest::Approx(7.0f));
    }
    SUBCASE("reduction in chain") {
        auto t = Tensor<float>::from({1, 2, 3, 4}, {2, 2}, true);
        auto doubled = t * 2.0f;
        auto sum = doubled.sum();  // sum(2t) = 20
        
        sum.backward();
        
        // d/dt_i = 2
        auto* g = t.grad_ptr();
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK((*g)[i] == doctest::Approx(2.0f));
        }
    }
}
TEST_CASE("Gradient: view operations") {
    using makemore::Tensor;
    SUBCASE("gradient through row view") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3}, true);
        auto row = t.row(0);  // {1, 2, 3}
        auto s = row.sum();   // 6
        
        s.backward();
        
        // Only row 0 elements have gradient 1, row 1 has gradient 0
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        
        // Row 0: indices 0, 1, 2
        CHECK((*g)[0] == doctest::Approx(1.0f));
        CHECK((*g)[1] == doctest::Approx(1.0f));
        CHECK((*g)[2] == doctest::Approx(1.0f));
        
        // Row 1: indices 3, 4, 5
        CHECK((*g)[3] == doctest::Approx(0.0f));
        CHECK((*g)[4] == doctest::Approx(0.0f));
        CHECK((*g)[5] == doctest::Approx(0.0f));
    }
    SUBCASE("gradient through col view") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {3, 2}, true);
        // t = [[1,2], [3,4], [5,6]]
        auto col = t.col(1);  // {2, 4, 6}
        auto s = col.sum();   // 12
        
        s.backward();
        
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        
        // Col 0: indices 0, 2, 4 → grad 0
        CHECK((*g)[0] == doctest::Approx(0.0f));
        CHECK((*g)[2] == doctest::Approx(0.0f));
        CHECK((*g)[4] == doctest::Approx(0.0f));
        
        // Col 1: indices 1, 3, 5 → grad 1
        CHECK((*g)[1] == doctest::Approx(1.0f));
        CHECK((*g)[3] == doctest::Approx(1.0f));
        CHECK((*g)[5] == doctest::Approx(1.0f));
    }
    SUBCASE("gradient through transpose") {
        auto t = Tensor<float>::from({1, 2, 3, 4}, {2, 2}, true);
        auto tT = t.transpose();
        auto s = tT.sum();
        
        s.backward();
        
        // All elements contribute → all have gradient 1
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        CHECK((*g)[0] == doctest::Approx(1.0f));
        CHECK((*g)[1] == doctest::Approx(1.0f));
        CHECK((*g)[2] == doctest::Approx(1.0f));
        CHECK((*g)[3] == doctest::Approx(1.0f));
    }
    SUBCASE("gradient with view operation on result") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3}, true);
        auto r = t * 2.0f;       // {2,4,6,8,10,12}
        auto row = r.row(1);     // {8, 10, 12}
        auto s = row.sum();      // 30
        
        s.backward();
        
        // d/dt for row 0: 0 (not used)
        // d/dt for row 1: 2 (the scalar multiplier)
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        
        CHECK((*g)[0] == doctest::Approx(0.0f));
        CHECK((*g)[1] == doctest::Approx(0.0f));
        CHECK((*g)[2] == doctest::Approx(0.0f));
        CHECK((*g)[3] == doctest::Approx(2.0f));
        CHECK((*g)[4] == doctest::Approx(2.0f));
        CHECK((*g)[5] == doctest::Approx(2.0f));
    }
    SUBCASE("scalar op on view then gradient") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3}, true);
        auto row = t.row(1);      // View: {4, 5, 6}
        auto r = row + 10.0f;     // {14, 15, 16}
        auto s = r.sum();         // 45
        
        s.backward();
        
        auto* g = t.grad_ptr();
        REQUIRE(g != nullptr);
        
        // Row 0: not involved → grad 0
        CHECK((*g)[0] == doctest::Approx(0.0f));
        CHECK((*g)[1] == doctest::Approx(0.0f));
        CHECK((*g)[2] == doctest::Approx(0.0f));
        
        // Row 1: d(sum(t+c))/dt = 1
        CHECK((*g)[3] == doctest::Approx(1.0f));
        CHECK((*g)[4] == doctest::Approx(1.0f));
        CHECK((*g)[5] == doctest::Approx(1.0f));
    }
}
TEST_CASE("Gradient: zero_grad") {
    using makemore::Tensor;
    SUBCASE("zero_grad clears gradients") {
        auto t = Tensor<float>::from({1, 2, 3}, {3}, true);
        auto r = t * 2.0f;
        r.sum().backward();
        
        // Gradients are 2
        auto* g = t.grad_ptr();
        CHECK((*g)[0] == doctest::Approx(2.0f));
        
        // Clear gradients
        t.zero_grad();
        
        CHECK((*g)[0] == doctest::Approx(0.0f));
        CHECK((*g)[1] == doctest::Approx(0.0f));
        CHECK((*g)[2] == doctest::Approx(0.0f));
    }
    SUBCASE("gradients accumulate without zero_grad") {
        auto t = Tensor<float>::from({1, 2, 3}, {3}, true);
        
        // First backward
        auto r1 = t.sum();
        r1.backward();
        CHECK((*t.grad_ptr())[0] == doctest::Approx(1.0f));
        
        // Second backward without zero_grad → accumulates
        auto r2 = t.sum();
        r2.backward();
        CHECK((*t.grad_ptr())[0] == doctest::Approx(2.0f));
        
        // Now zero and run again
        t.zero_grad();
        auto r3 = t.sum();
        r3.backward();
        CHECK((*t.grad_ptr())[0] == doctest::Approx(1.0f));
    }
}

TEST_CASE("Matrix multiplication") {
    using makemore::Tensor;
    using sizeVec = std::vector<std::size_t>;
    SUBCASE("basic 2x3 @ 3x2") {
        auto a = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        // a = [[1,2,3], [4,5,6]]
        
        auto b = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {3, 2});
        // b = [[1,2], [3,4], [5,6]]
        
        auto c = a.matmul(b);
        
        CHECK(c.shape() == sizeVec{2, 2});
        // c[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // c[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // c[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // c[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        CHECK(c[0, 0] == doctest::Approx(22.0f));
        CHECK(c[0, 1] == doctest::Approx(28.0f));
        CHECK(c[1, 0] == doctest::Approx(49.0f));
        CHECK(c[1, 1] == doctest::Approx(64.0f));
    }
    SUBCASE("matmul with transposed input") {
        auto a = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {3, 2});
        auto aT = a.transpose();  // {2, 3}
        
        auto b = Tensor<float>::from({1, 0, 0, 1, 0, 0}, {3, 2});  // Identity-ish
        
        auto c = aT.matmul(b);
        
        CHECK(c.shape() == sizeVec{2, 2});
        // aT = [[1,3,5], [2,4,6]]
        // c[0,0] = 1*1 + 3*0 + 5*0 = 1
        // c[0,1] = 1*0 + 3*1 + 5*0 = 3
        // c[1,0] = 2*1 + 4*0 + 6*0 = 2
        // c[1,1] = 2*0 + 4*1 + 6*0 = 4
        CHECK(c[0, 0] == doctest::Approx(1.0f));
        CHECK(c[0, 1] == doctest::Approx(3.0f));
        CHECK(c[1, 0] == doctest::Approx(2.0f));
        CHECK(c[1, 1] == doctest::Approx(4.0f));
    }
    SUBCASE("matmul gradient") {
        auto a = Tensor<float>::from({1, 2, 3, 4}, {2, 2}, true);
        auto b = Tensor<float>::from({5, 6, 7, 8}, {2, 2}, true);
        auto c = a.matmul(b);
        
        c.sum().backward();
        
        // dL/dA = dL/dC @ B^T, where dL/dC = ones(2,2)
        // B^T = [[5,7], [6,8]]
        // dL/dA[0,0] = 1*5 + 1*6 = 11
        // dL/dA[0,1] = 1*7 + 1*8 = 15
        // dL/dA[1,0] = 1*5 + 1*6 = 11
        // dL/dA[1,1] = 1*7 + 1*8 = 15
        auto* ga = a.grad_ptr();
        CHECK((*ga)[0] == doctest::Approx(11.0f));
        CHECK((*ga)[1] == doctest::Approx(15.0f));
        CHECK((*ga)[2] == doctest::Approx(11.0f));
        CHECK((*ga)[3] == doctest::Approx(15.0f));
        
        // dL/dB = A^T @ dL/dC
        // A^T = [[1,3], [2,4]]
        // dL/dB[0,0] = 1*1 + 3*1 = 4
        // dL/dB[0,1] = 1*1 + 3*1 = 4
        // dL/dB[1,0] = 2*1 + 4*1 = 6
        // dL/dB[1,1] = 2*1 + 4*1 = 6
        auto* gb = b.grad_ptr();
        CHECK((*gb)[0] == doctest::Approx(4.0f));
        CHECK((*gb)[1] == doctest::Approx(4.0f));
        CHECK((*gb)[2] == doctest::Approx(6.0f));
        CHECK((*gb)[3] == doctest::Approx(6.0f));
    }
}

TEST_CASE("dot product") {
    using makemore::Tensor;
    SUBCASE("basic dot product") {
        auto a = Tensor<float>::from({1, 2, 3}, sizeVec{3});
        auto b = Tensor<float>::from({4, 5, 6}, sizeVec{3});
        auto c = a.matmul(b);
        
        CHECK(c.shape() == std::vector<std::size_t>{1});
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        CHECK(c.item() == doctest::Approx(32.0f));
    }
    SUBCASE("dot product on views") {
        auto t = Tensor<float>::from({1, 2, 3, 4, 5, 6}, {2, 3});
        auto row0 = t.row(0);  // [1, 2, 3]
        auto row1 = t.row(1);  // [4, 5, 6]
        auto c = row0.matmul(row1);
        
        // 1*4 + 2*5 + 3*6 = 32
        CHECK(c.item() == doctest::Approx(32.0f));
    }
    SUBCASE("dot product gradient") {
        auto a = Tensor<float>::from({1, 2, 3}, sizeVec{3}, true);
        auto b = Tensor<float>::from({4, 5, 6}, sizeVec{3}, true);
        auto c = a.matmul(b);  // scalar: 32
        
        c.backward();
        
        // ∂c/∂a = b, ∂c/∂b = a
        auto* ga = a.grad_ptr();
        auto* gb = b.grad_ptr();
        
        CHECK((*ga)[0] == doctest::Approx(4.0f));
        CHECK((*ga)[1] == doctest::Approx(5.0f));
        CHECK((*ga)[2] == doctest::Approx(6.0f));
        
        CHECK((*gb)[0] == doctest::Approx(1.0f));
        CHECK((*gb)[1] == doctest::Approx(2.0f));
        CHECK((*gb)[2] == doctest::Approx(3.0f));
    }
    SUBCASE("dot product with scalar multiplication in chain") {
        auto a = Tensor<float>::from({1, 2}, sizeVec{2}, true);
        auto b = Tensor<float>::from({3, 4}, sizeVec{2}, true);
        auto c = a.matmul(b);     // 1*3 + 2*4 = 11
        auto d = c * 2.0f;     // 22
        
        d.backward();
        
        // d = 2 * (a·b)
        // ∂d/∂a = 2 * b = [6, 8]
        // ∂d/∂b = 2 * a = [2, 4]
        auto* ga = a.grad_ptr();
        auto* gb = b.grad_ptr();
        
        CHECK((*ga)[0] == doctest::Approx(6.0f));
        CHECK((*ga)[1] == doctest::Approx(8.0f));
        CHECK((*gb)[0] == doctest::Approx(2.0f));
        CHECK((*gb)[1] == doctest::Approx(4.0f));
    }
}

TEST_CASE("Tensor exp()") {
    using makemore::Tensor;
    using sizeVec = std::vector<std::size_t>;
    SUBCASE("basic exp 1D") {
        auto t = Tensor<float>::from({0.0f, 1.0f, 2.0f}, sizeVec{3});
        auto r = t.exp();
        
        CHECK(r.shape() == sizeVec{3});
        CHECK(r[0] == doctest::Approx(1.0f));           // exp(0) = 1
        CHECK(r[1] == doctest::Approx(2.71828f).epsilon(0.001));  // exp(1) ≈ e
        CHECK(r[2] == doctest::Approx(7.38906f).epsilon(0.001));  // exp(2) ≈ e^2
    }
    SUBCASE("exp 2D") {
        auto t = Tensor<float>::from({0.0f, 1.0f, -1.0f, 2.0f}, {2, 2});
        auto r = t.exp();
        
        CHECK(r.shape() == sizeVec{2, 2});
        CHECK(r[0, 0] == doctest::Approx(1.0f));
        CHECK(r[0, 1] == doctest::Approx(std::exp(1.0f)));
        CHECK(r[1, 0] == doctest::Approx(std::exp(-1.0f)));
        CHECK(r[1, 1] == doctest::Approx(std::exp(2.0f)));
    }
    SUBCASE("exp on view") {
        auto t = Tensor<float>::from({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {2, 3});
        auto row = t.row(1);  // [3, 4, 5]
        auto r = row.exp();
        
        CHECK(r.shape() == sizeVec{3});
        CHECK(r[0] == doctest::Approx(std::exp(3.0f)));
        CHECK(r[1] == doctest::Approx(std::exp(4.0f)));
        CHECK(r[2] == doctest::Approx(std::exp(5.0f)));
    }
    SUBCASE("exp on transposed view") {
        auto t = Tensor<float>::from({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2});
        auto tT = t.transpose();
        auto r = tT.exp();
        
        CHECK(r.shape() == sizeVec{2, 2});
        CHECK(r[0, 0] == doctest::Approx(std::exp(0.0f)));  // t[0,0]
        CHECK(r[0, 1] == doctest::Approx(std::exp(2.0f)));  // t[1,0]
        CHECK(r[1, 0] == doctest::Approx(std::exp(1.0f)));  // t[0,1]
        CHECK(r[1, 1] == doctest::Approx(std::exp(3.0f)));  // t[1,1]
    }
    SUBCASE("exp gradient") {
        auto t = Tensor<float>::from({0.0f, 1.0f, 2.0f}, sizeVec{3}, true);
        auto r = t.exp();
        r.sum().backward();
        
        // d(exp(x))/dx = exp(x)
        auto* g = t.grad_ptr();
        CHECK((*g)[0] == doctest::Approx(std::exp(0.0f)));  // 1
        CHECK((*g)[1] == doctest::Approx(std::exp(1.0f)));  // e
        CHECK((*g)[2] == doctest::Approx(std::exp(2.0f)));  // e^2
    }
    SUBCASE("exp gradient chain rule") {
        auto t = Tensor<float>::from({1.0f, 2.0f}, sizeVec{2}, true);
        auto r = (t * 2.0f).exp();  // exp(2t)
        r.sum().backward();
        
        // d(exp(2t))/dt = 2 * exp(2t)
        auto* g = t.grad_ptr();
        CHECK((*g)[0] == doctest::Approx(2.0f * std::exp(2.0f)));
        CHECK((*g)[1] == doctest::Approx(2.0f * std::exp(4.0f)));
    }
    SUBCASE("original tensor unchanged") {
        auto t = Tensor<float>::from({1.0f, 2.0f, 3.0f}, sizeVec{3});
        auto r = t.exp();
        
        CHECK(t[0] == doctest::Approx(1.0f));
        CHECK(t[1] == doctest::Approx(2.0f));
        CHECK(t[2] == doctest::Approx(3.0f));
    }
}

TEST_CASE("Unary operations: exp") {
    using makemore::Tensor;
    
    SUBCASE("forward") {
        auto t = Tensor<float>::from({0.0f, 1.0f, 2.0f}, sizeVec{3});
        auto r = t.exp();
        CHECK(r[0] == doctest::Approx(1.0f));
        CHECK(r[1] == doctest::Approx(std::exp(1.0f)));
        CHECK(r[2] == doctest::Approx(std::exp(2.0f)));
    }
    
    SUBCASE("gradient") {
        auto t = Tensor<float>::from({0.0f, 1.0f, 2.0f}, sizeVec{3}, true);
        t.exp().sum().backward();
        auto* g = t.grad_ptr();
        CHECK((*g)[0] == doctest::Approx(std::exp(0.0f)));
        CHECK((*g)[1] == doctest::Approx(std::exp(1.0f)));
        CHECK((*g)[2] == doctest::Approx(std::exp(2.0f)));
    }
    
    SUBCASE("on view") {
        auto t = Tensor<float>::from({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2});
        auto r = t.row(1).exp();  // exp([2, 3])
        CHECK(r[0] == doctest::Approx(std::exp(2.0f)));
        CHECK(r[1] == doctest::Approx(std::exp(3.0f)));
    }
}
TEST_CASE("Unary operations: log") {
    using makemore::Tensor;
    
    SUBCASE("forward") {
        auto t = Tensor<float>::from({1.0f, std::exp(1.0f), std::exp(2.0f)}, sizeVec{3});
        auto r = t.log();
        CHECK(r[0] == doctest::Approx(0.0f));
        CHECK(r[1] == doctest::Approx(1.0f));
        CHECK(r[2] == doctest::Approx(2.0f));
    }
    
    SUBCASE("gradient") {
        auto t = Tensor<float>::from({1.0f, 2.0f, 4.0f}, sizeVec{3}, true);
        t.log().sum().backward();
        auto* g = t.grad_ptr();
        CHECK((*g)[0] == doctest::Approx(1.0f));     // 1/1
        CHECK((*g)[1] == doctest::Approx(0.5f));     // 1/2
        CHECK((*g)[2] == doctest::Approx(0.25f));    // 1/4
    }
}
TEST_CASE("Unary operations: sqrt") {
    using makemore::Tensor;
    
    SUBCASE("forward") {
        auto t = Tensor<float>::from({1.0f, 4.0f, 9.0f}, sizeVec{3});
        auto r = t.sqrt();
        CHECK(r[0] == doctest::Approx(1.0f));
        CHECK(r[1] == doctest::Approx(2.0f));
        CHECK(r[2] == doctest::Approx(3.0f));
    }
    
    SUBCASE("gradient") {
        auto t = Tensor<float>::from({1.0f, 4.0f, 9.0f}, sizeVec{3}, true);
        t.sqrt().sum().backward();
        auto* g = t.grad_ptr();
        CHECK((*g)[0] == doctest::Approx(0.5f));      // 1/(2*1)
        CHECK((*g)[1] == doctest::Approx(0.25f));     // 1/(2*2)
        CHECK((*g)[2] == doctest::Approx(1.0f/6.0f)); // 1/(2*3)
    }
}
TEST_CASE("Unary operations: tanh") {
    using makemore::Tensor;
    
    SUBCASE("forward") {
        auto t = Tensor<float>::from({0.0f, 1.0f, -1.0f}, sizeVec{3});
        auto r = t.tanh();
        CHECK(r[0] == doctest::Approx(0.0f));
        CHECK(r[1] == doctest::Approx(std::tanh(1.0f)));
        CHECK(r[2] == doctest::Approx(std::tanh(-1.0f)));
    }
    
    SUBCASE("gradient") {
        auto t = Tensor<float>::from({0.0f, 1.0f}, sizeVec{2}, true);
        t.tanh().sum().backward();
        auto* g = t.grad_ptr();
        // d/dx tanh(x) = 1 - tanh²(x)
        CHECK((*g)[0] == doctest::Approx(1.0f));  // 1 - 0² = 1
        float tanh1 = std::tanh(1.0f);
        CHECK((*g)[1] == doctest::Approx(1.0f - tanh1 * tanh1));
    }
}
TEST_CASE("Unary operations: sigmoid") {
    using makemore::Tensor;
    
    SUBCASE("forward") {
        auto t = Tensor<float>::from({0.0f, 100.0f, -100.0f}, sizeVec{3});
        auto r = t.sigmoid();
        CHECK(r[0] == doctest::Approx(0.5f));
        CHECK(r[1] == doctest::Approx(1.0f).epsilon(0.01));  // ~1
        CHECK(r[2] == doctest::Approx(0.0f).epsilon(0.01));  // ~0
    }
    
    SUBCASE("gradient") {
        auto t = Tensor<float>::from({0.0f}, sizeVec{1}, true);
        t.sigmoid().backward();
        auto* g = t.grad_ptr();
        // σ'(0) = σ(0)(1 - σ(0)) = 0.5 * 0.5 = 0.25
        CHECK((*g)[0] == doctest::Approx(0.25f));
    }
}
TEST_CASE("Unary operations: relu") {
    using makemore::Tensor;
    
    SUBCASE("forward") {
        auto t = Tensor<float>::from({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, sizeVec{5});
        auto r = t.relu();
        CHECK(r[0] == doctest::Approx(0.0f));
        CHECK(r[1] == doctest::Approx(0.0f));
        CHECK(r[2] == doctest::Approx(0.0f));
        CHECK(r[3] == doctest::Approx(1.0f));
        CHECK(r[4] == doctest::Approx(2.0f));
    }
    
    SUBCASE("gradient") {
        auto t = Tensor<float>::from({-1.0f, 0.0f, 1.0f, 2.0f}, sizeVec{4}, true);
        t.relu().sum().backward();
        auto* g = t.grad_ptr();
        CHECK((*g)[0] == doctest::Approx(0.0f));  // x < 0
        CHECK((*g)[1] == doctest::Approx(0.0f));  // x = 0 (convention)
        CHECK((*g)[2] == doctest::Approx(1.0f));  // x > 0
        CHECK((*g)[3] == doctest::Approx(1.0f));  // x > 0
    }
}
TEST_CASE("Unary operations: neg") {
    using makemore::Tensor;
    
    SUBCASE("forward") {
        auto t = Tensor<float>::from({1.0f, -2.0f, 3.0f}, sizeVec{3});
        auto r = t.neg();
        CHECK(r[0] == doctest::Approx(-1.0f));
        CHECK(r[1] == doctest::Approx(2.0f));
        CHECK(r[2] == doctest::Approx(-3.0f));
    }
    
    SUBCASE("gradient") {
        auto t = Tensor<float>::from({1.0f, 2.0f, 3.0f}, sizeVec{3}, true);
        t.neg().sum().backward();
        auto* g = t.grad_ptr();
        CHECK((*g)[0] == doctest::Approx(-1.0f));
        CHECK((*g)[1] == doctest::Approx(-1.0f));
        CHECK((*g)[2] == doctest::Approx(-1.0f));
    }
}
TEST_CASE("Unary operations: chained") {
    using makemore::Tensor;
    
    SUBCASE("exp(log(x)) = x") {
        auto t = Tensor<float>::from({1.0f, 2.0f, 3.0f}, sizeVec{3});
        auto r = t.log().exp();
        CHECK(r[0] == doctest::Approx(1.0f));
        CHECK(r[1] == doctest::Approx(2.0f));
        CHECK(r[2] == doctest::Approx(3.0f));
    }
    
    SUBCASE("chained gradient") {
        auto t = Tensor<float>::from({2.0f}, sizeVec{1}, true);
        auto r = t.exp().log();  // log(exp(x)) = x, so d/dx = 1
        r.backward();
        auto* g = t.grad_ptr();
        CHECK((*g)[0] == doctest::Approx(1.0f));
    }
}